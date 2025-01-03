import os
import argparse
import pickle
from pathlib import Path
from omegaconf import OmegaConf

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score 
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from script.utils.utils import fix_seed, get_date
from script.utils.cls_utils import make_optimizer, make_scheduler, compute_mae, compute_rmse, get_loader, gen_forget_schedule
from script.utils.model_utils import classifier_model

def softCrossEntropy(inputs, target, reduction='none'):
    """
    :param inputs: predictions
    :param target: target labels
    :return: loss
    """
    log_likelihood = - F.log_softmax(inputs, dim=1)
    sample_num, class_num = target.shape

    if reduction == 'sum':
        loss = torch.sum(torch.mul(log_likelihood, target))
    elif reduction == 'mean':
        loss = torch.sum(torch.mul(log_likelihood, target))/sample_num
    elif reduction == 'none':
        loss = torch.sum(torch.mul(log_likelihood, target), dim=1)

    return loss

def soft_label_with_temperetures_batch(hard_labels, temperatures, num_classes=4, eps=1e-8):
    # hard_labelsはバッチサイズの1次元テンソルと仮定します
    batch_size = hard_labels.size(0)

    # num_classesの長さを持つテンソルをバッチサイズ分繰り返す
    label_tensor = torch.arange(num_classes).unsqueeze(0).repeat(batch_size, 1)  # (batch_size, num_classes)

    # hard_labelsをnum_classesの長さを持つテンソルに展開して繰り返す
    hard_label_tensor = hard_labels.unsqueeze(1).repeat(1, num_classes)  # (batch_size, num_classes)

    # ソフトラベルを計算
    soft_label = torch.exp(-torch.abs(label_tensor - hard_label_tensor)/(temperatures+eps))
    soft_label = soft_label / torch.sum(soft_label, dim=1, keepdim=True)  # 各バッチで正規化

    return soft_label

def hard2soft_label_batch(hard_labels, num_classes=4):
    # hard_labelsはバッチサイズの1次元テンソルと仮定します
    batch_size = hard_labels.size(0)

    # num_classesの長さを持つテンソルをバッチサイズ分繰り返す
    label_tensor = torch.arange(num_classes).unsqueeze(0).repeat(batch_size, 1)  # (batch_size, num_classes)

    # hard_labelsをnum_classesの長さを持つテンソルに展開して繰り返す
    hard_label_tensor = hard_labels.unsqueeze(1).repeat(1, num_classes)  # (batch_size, num_classes)

    # ソフトラベルを計算
    soft_label = torch.exp(-torch.abs(label_tensor - hard_label_tensor))
    soft_label = soft_label / torch.sum(soft_label, dim=1, keepdim=True)  # 各バッチで正規化

    return soft_label

def loss_soft_coteaching_plus(logits_1, logits_2, labels, temp, forget_rate, ind, noise_or_not, num_classes, device):
    soft_labels = hard2soft_label_batch(labels, num_classes).to(device)
    hard_labels = labels.to(device)

    loss_1 = F.cross_entropy(logits_1 / temp, hard_labels, reduction='none')
    ind_1_sorted = np.argsort(loss_1.cpu().data).to(device)
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(logits_2 / temp, hard_labels, reduction='none')
    ind_2_sorted = np.argsort(loss_2.cpu().data).to(device)
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    ind_1_update=ind_1_sorted[:num_remember].cpu()
    ind_2_update=ind_2_sorted[:num_remember].cpu()
    if len(ind_1_update) == 0:
        ind_1_update = ind_1_sorted.cpu().numpy()
        ind_2_update = ind_2_sorted.cpu().numpy()
        num_remember = ind_1_update.shape[0]

    pure_ratio_hard = np.sum(noise_or_not[ind[ind_1_sorted.cpu()[:num_remember]]])/float(num_remember)
    pure_ratio_soft = np.sum(noise_or_not[ind[ind_2_sorted.cpu()[:num_remember]]])/float(num_remember)

    loss_1_update = softCrossEntropy(logits_1[ind_2_update], soft_labels[ind_2_update], reduction='none')
    loss_2_update = softCrossEntropy(logits_2[ind_1_update], soft_labels[ind_1_update], reduction='none')
    
    return num_remember, torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, pure_ratio_hard, pure_ratio_soft


def train(loader, epoch, net1, optimizer1, scheduler1, net2, optimizer2, scheduler2, rate_schedule, temp, noise_or_not, num_classes, device):
    net1.train()
    net2.train()

    log_loss = 0
    n_data = 0

    y_pred = {}
    y_true = {}
    log_loss = {}
    pure_ratio_list = {}
    results = {}
    for name in ['net1', 'net2', 'net1&net2']:
        pure_ratio_list[name] = []
        log_loss[name] = 0
        y_pred[name] = []
        y_true[name] = []

    for images, labels, _, indices in loader:
        ind = indices.cpu().numpy().transpose()
        images = images.to(device)
        labels = labels.to(torch.int64)

        logits1 = net1(images)
        logits2 = net2(images)

        num_remember, loss1, loss2, pure_ratio_1, pure_ratio_2 = \
              loss_soft_coteaching_plus(logits1, logits2, labels, temp, rate_schedule[epoch], ind, noise_or_not, num_classes, device)
        
        pure_ratio_list['net1'].append(100*pure_ratio_1)
        pure_ratio_list['net2'].append(100*pure_ratio_2)
        pure_ratio_list['net1&net2'].append((100*pure_ratio_1 + 100*pure_ratio_2)/2)
        
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()

        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        log_loss['net1'] += loss1.item() * num_remember
        log_loss['net2'] += loss2.item() * num_remember
        log_loss['net1&net2'] += (loss1.item() + loss2.item()) * num_remember / 2
        n_data += num_remember

        _, pred1 = logits1.max(1)
        pred1 = pred1.detach().cpu()
        if pred1.size == 1:
            y_pred['net1'].append(pred1.item())
            y_pred['net1&net2'].append(pred1.item())
            y_true['net1'].append(labels.item())
            y_true['net1&net2'].append(labels.item())
        else:
            for d in pred1:
                y_pred['net1'].append(d.item())
                y_pred['net1&net2'].append(d.item())
            for d in labels:
                y_true['net1'].append(d.item())
                y_true['net1&net2'].append(d.item())
    
        _, pred2 = logits2.max(1)
        pred2 = pred2.detach().cpu()
        if pred2.size == 1:
            y_pred['net2'].append(pred2.item())
            y_pred['net1&net2'].append(pred2.item())
            y_true['net2'].append(labels.item())
            y_true['net1&net2'].append(labels.item())
        else:
            for d in pred2:
                y_pred['net2'].append(d.item())
                y_pred['net1&net2'].append(d.item())
            for d in labels:
                y_true['net2'].append(d.item())
                y_true['net1&net2'].append(d.item())
    
    scheduler1.step()
    scheduler2.step()

    for name in ['net1', 'net2', 'net1&net2']:
        acc = accuracy_score(y_true[name], y_pred[name])
        mae = compute_mae(y_true[name], y_pred[name])
        rmse = compute_rmse(y_true[name], y_pred[name])
        f1 = f1_score(y_true[name], y_pred[name], average='macro')
        precision = precision_score(y_true[name], y_pred[name], average='macro')
        recall = recall_score(y_true[name], y_pred[name], average='macro')
        kappa = cohen_kappa_score(y_true[name], y_pred[name], weights='quadratic')
        results[name] = {'loss': log_loss[name]/n_data, 'acc': acc, 'mae': mae, 'rmse': rmse, 'f1': f1, 'precision': precision, 'recall': recall, 'kappa': kappa, 'pure_ratio':np.mean(pure_ratio_list[name])}

    return results

@torch.no_grad()
def eval(loader, net1, net2, num_classes, device):
    net1.eval()
    net2.eval()

    log_loss = 0
    n_data = 0

    y_pred = {}
    y_true = {}
    log_loss = {}
    pure_ratio_list = {}
    results = {}
    for name in ['net1', 'net2', 'net1&net2']:
        pure_ratio_list[name] = []
        log_loss[name] = 0
        y_pred[name] = []
        y_true[name] = []

    for images, labels, _, _ in loader:
        images = images.to(device)
        labels = labels.to(torch.int64)
        soft_labels = hard2soft_label_batch(labels, num_classes).to(device)
        n_data += images.size(0)

        logits1 = net1(images)
        logits2 = net2(images)
        loss1 = softCrossEntropy(logits1, soft_labels, reduction='sum')
        loss2 = softCrossEntropy(logits2, soft_labels, reduction='sum')

        log_loss['net1'] += loss1.item()
        log_loss['net2'] += loss2.item() 
        log_loss['net1&net2'] += (loss1.item() + loss2.item()) / 2

        _, pred1 = logits1.max(1)
        pred1 = pred1.detach().cpu()
        if pred1.size == 1:
            y_pred['net1'].append(pred1.item())
            y_pred['net1&net2'].append(pred1.item())
            y_true['net1'].append(labels.item())
            y_true['net1&net2'].append(labels.item())
        else:
            for d in pred1:
                y_pred['net1'].append(d.item())
                y_pred['net1&net2'].append(d.item())
            for d in labels:
                y_true['net1'].append(d.item())
                y_true['net1&net2'].append(d.item())
    
        _, pred2 = logits2.max(1)
        pred2 = pred2.detach().cpu()
        if pred2.size == 1:
            y_pred['net2'].append(pred2.item())
            y_pred['net1&net2'].append(pred2.item())
            y_true['net2'].append(labels.item())
            y_true['net1&net2'].append(labels.item())
        else:
            for d in pred2:
                y_pred['net2'].append(d.item())
                y_pred['net1&net2'].append(d.item())
            for d in labels:
                y_true['net2'].append(d.item())
                y_true['net1&net2'].append(d.item())

    for name in ['net1', 'net2', 'net1&net2']:
        acc = accuracy_score(y_true[name], y_pred[name])
        mae = compute_mae(y_true[name], y_pred[name])
        rmse = compute_rmse(y_true[name], y_pred[name])
        f1 = f1_score(y_true[name], y_pred[name], average='macro')
        precision = precision_score(y_true[name], y_pred[name], average='macro')
        recall = recall_score(y_true[name], y_pred[name], average='macro')
        kappa = cohen_kappa_score(y_true[name], y_pred[name], weights='quadratic')
        results[name] = {'loss': log_loss[name]/n_data, 'acc': acc, 'mae': mae, 'rmse': rmse, 'f1': f1, 'precision': precision, 'recall': recall, 'kappa': kappa, 'pure_ratio':np.mean(pure_ratio_list[name])}

    return results

def main(args):
    fix_seed(seed=args.seed)

    workdir = Path(args.workdir).joinpath(args.data_name, f'{args.noise_type}-{args.noise_rate}', f'fold{args.fold}')
    config = OmegaConf.load(args.config)
    date = get_date()
    txtfile = workdir / "log.txt"
    if os.path.exists(txtfile):
        os.system(f'mv {txtfile} {txtfile}.bak-{date}')

    workdir.mkdir(parents=True, exist_ok=True)
    os.makedirs(workdir / 'ckpt', exist_ok=True)
    os.makedirs(workdir / 'logs'/ date, exist_ok=True)
    log = f'{date}, {args.data_name}-{args.noise_type}-{args.noise_rate}-fold{args.fold}'
    print(log)
    with open(txtfile, "a") as myfile:
        myfile.write(log + '\n')

    # writer
    writer = SummaryWriter(log_dir=workdir / 'logs'/ date)

    # GPU
    if torch.cuda.is_available():   
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Configure data loader
    csv_root = Path('./dataset').joinpath(args.data_name, '5fold', f'{args.noise_type}-{args.noise_rate}', f'fold{args.fold}')
    train_dataset, train_loader = get_loader(csv_file= csv_root / f'train_fold{args.fold}.csv', batch_size=config.training.batch_size, noisy=True, train=True)
    val_dataset, val_loader = get_loader(csv_file= csv_root / f'val_fold{args.fold}.csv', batch_size=config.training.batch_size, noisy=True, train=False)
    test_dataset, test_loader = get_loader(csv_file= csv_root / f'test_fold{args.fold}.csv', batch_size=config.training.batch_size, noisy=False, train=False)

    noise_or_not = train_dataset.noise_or_not
    forget_rate = args.noise_rate
    rate_schedule = gen_forget_schedule(forget_rate=forget_rate, 
                                    n_epoch=config.training.max_epoch, 
                                    num_gradual=config.training.forget_num_gradual,
                                    )


    # Build net
    log = f'Build model -> {config.model.name}\n'
    net1 = classifier_model(**config.model)
    net1.to(device) 
    net2 = classifier_model(**config.model)
    net2.to(device) 

    # Optimizers
    log += f'Build optimizer -> {config.optimizer.name}\n'
    optimizer1 = make_optimizer(net1.parameters(), **config.optimizer)
    optimizer2 = make_optimizer(net2.parameters(), **config.optimizer)
    log += f'Build scheduler -> {config.scheduler.name}\n'
    scheduler1 = make_scheduler(optimizer1, **config.scheduler)
    scheduler2 = make_scheduler(optimizer2, **config.scheduler)

    print(log)
    with open(txtfile, "a") as myfile:
        myfile.write(log + '\n')

    # Training
    save_results = {'net1':{}, 'net2':{}, 'net1&net2':{}}
    for name in ['net1', 'net2', 'net1&net2']:
        for mode in ['train', 'val', 'test']:
            save_results[name][mode] = {}
            for metric in config.training.metrics:
                    save_results[name][mode][metric] = []

    all_results = {}
    for epoch in range(config.training.max_epoch):

        all_results['train'] = train(
                        loader=train_loader,
                        epoch=epoch,
                        net1=net1,
                        optimizer1=optimizer1,
                        scheduler1=scheduler1,
                        net2=net2,
                        optimizer2=optimizer2,
                        scheduler2=scheduler2,
                        rate_schedule=rate_schedule,
                        temp=config.training.temp,
                        noise_or_not=noise_or_not,
                        num_classes=config.model.num_classes,
                        device=device
                        )
                
        with torch.no_grad():
            all_results['val'] = eval(
                            loader=val_loader,
                            net1=net1,
                            net2=net2,
                            num_classes=config.model.num_classes,
                            device=device,
                            )

            all_results['test'] = eval(
                loader=test_loader,
                net1=net1,
                net2=net2,
                num_classes=config.model.num_classes,
                device=device,
                )
        
        # Log writer
        for mode in ['train', 'val', 'test']:
            for metric in config.training.metrics:
                if mode != 'train' and metric == 'pure_ratio':
                    pass
                else:
                    for name in ['net1', 'net2', 'net1&net2']:
                        writer.add_scalar(f'{name}/{metric}/{mode}_{metric}', all_results[mode][name][metric], epoch)
                        save_results[name][mode][metric].append(all_results[mode][name][metric])

        # Log text
        log = f'Fold{args.fold} [Epoch {epoch+1}/{config.training.max_epoch}]'
        for mode in ['train', 'val', 'test']:
            for metric in config.training.metrics:
                if mode != 'train' and metric == 'pure_ratio':
                    pass
                else:
                    for name in ['net1', 'net2', 'net1&net2']:
                        log += ' ' + f'[{mode}_{metric}_{name}: {all_results[mode][name][metric]:.4f}]'
        print(log)
        with open(txtfile, "a") as myfile:
            myfile.write(log + '\n')
        
    # Save weight of last epoch
    print('Last epoch, Saving model ...')
    torch.save(net1.state_dict(), workdir.joinpath('ckpt','net1_last.ckpt'))
    torch.save(optimizer1.state_dict(), workdir.joinpath('ckpt', 'optim1_last.ckpt'))
    torch.save(net2.state_dict(), workdir.joinpath('ckpt','net2_last.ckpt'))
    torch.save(optimizer2.state_dict(), workdir.joinpath('ckpt', 'optim2_last.ckpt'))
    
   
    result = f'Last ten epochs average results ->'
    for mode in ['train', 'val', 'test']:
        for metric in config.training.metrics:
            if mode != 'train' and metric == 'pure_ratio':
                pass
            else:
                for name in ['net1', 'net2', 'net1&net2']:
                    result += ' ' + f'[{mode}_{metric}_{name}: {np.mean(save_results[name][mode][metric][-10:]):.4f}]' 

    print(result)
    with open(txtfile, "a") as myfile:
        myfile.write(result + '\n')
    
    with open(workdir / 'save_results.pickle', 'wb') as f:
        pickle.dump(save_results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./script/co_teaching_ours/config/co_teaching_ours.yaml', help='(.yaml)')
    parser.add_argument('--workdir', type=str, default='./expr/co_teaching_ours/')
    parser.add_argument('--data_name', type=str, default='limuc')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--noise_type', type=str, default='quasi', choices=['quasi', 'truncated'])
    parser.add_argument('--noise_rate', type=float, default=0.2, choices=[0.2,0.4])
    parser.add_argument('--seed', type=int, default=777)
    args = parser.parse_args()
    main(args)