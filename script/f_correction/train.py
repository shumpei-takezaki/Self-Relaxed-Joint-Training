import os
import argparse
import pickle
import tempfile
from pathlib import Path
from omegaconf import OmegaConf

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from script.utils.utils import fix_seed, get_date
from script.utils.cls_utils import make_optimizer, make_scheduler, compute_mae, compute_rmse, get_loader
from script.utils.model_utils import classifier_model
from dataset.noisy_csv_5fold import make_transition_matrix

class RobustCrossEntropyLoss(nn.Module):
    """CrossEntropyLoss class for noisy labels
    Args:
        T (Tensor): Row-Stochaistic transition matrix for the noise, shape (CxC)
        roobust_method (str): Specifies the method fo the robustness (either "forward" or "backward")
    """
    def __init__(self, T = None,
                 robust_method = "forward"):
        super(RobustCrossEntropyLoss, self).__init__()
        self.robust_method = robust_method
        assert self.robust_method in ["forward", "backward"]
        if self.robust_method == "backward":
            self.register_buffer("T", torch.linalg.inv(T))
        else:
            self.register_buffer("T", T)

    def forward(self, pred, target, reduction = 'mean', eps = 1e-7):
        target = target.type(self.T.dtype)
        pred = torch.clamp(pred.softmax(-1), min = eps, max = 1-eps)

        if self.robust_method == "backward":
            target = torch.inner(target, self.T)
        else:
            pred = torch.inner(pred, self.T)

        pred = torch.log(pred)
        if reduction == 'mean':
            return - torch.mean(torch.sum(target * pred, axis = -1))
        elif reduction == 'sum':
            return - torch.sum(torch.sum(target * pred, axis = -1))

def train(net, loader, optimizer, scheduler, criterion, num_classes, device):
    net.train()
    log_loss = 0
    step = 0
    n_data = 0
    y_pred = []
    y_true = []
    for images, labels, _, _ in loader:
        images = images.to(device)
        labels = labels.to(torch.int64).to(device)
        labels_onehot = F.one_hot(labels, num_classes)
        n_batch = images.size(0)

        outputs = net(images)
        loss = criterion(outputs, labels_onehot, 'sum')
        loss = loss / n_batch

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred = outputs.max(1)
        pred = pred.detach().cpu()
        if pred.size == 1:
            y_pred[pred.item()]
            y_true[labels.item()]
        else:
            for d in pred:
                y_pred.append(d.item())
            for d in labels:
                y_true.append(d.item())

        log_loss += (loss.item() * n_batch)
        step += 1
        n_data += n_batch

    scheduler.step()

    acc = accuracy_score(y_true, y_pred)
    mae = compute_mae(y_true, y_pred)
    rmse = compute_rmse(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')

    return {'loss': log_loss/n_data, 'acc': acc, 'mae': mae, 'rmse': rmse, 'f1': f1, 'precision': precision, 'recall': recall, 'kappa': kappa}

@torch.no_grad()
def eval(net, loader, criterion, num_classes, device):

    net.eval()
    log_loss = 0
    step = 0
    n_data = 0
    y_pred = []
    y_true = []
    for images, labels, _, _ in loader:
        images = images.to(device)
        labels = labels.to(torch.int64).to(device)
        labels_onehot = F.one_hot(labels, num_classes)

        outputs = net(images)
        loss = criterion(outputs, labels_onehot, 'sum')

        _, pred = outputs.max(1)
        pred = pred.detach().cpu().squeeze().numpy()
        if pred.size == 1:
            y_pred.append(pred)
            y_true.append(labels.item())
        else:
            for d in pred:
                y_pred.append(d)
            for d in labels:
                y_true.append(d.item())

        log_loss += loss.item()
        step += 1
        n_data += images.size(0)
    
    acc = accuracy_score(y_true, y_pred)
    mae = compute_mae(y_true, y_pred)
    rmse = compute_rmse(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')

    return {'loss': log_loss/n_data, 'acc': acc, 'mae': mae, 'rmse': rmse, 'f1': f1, 'precision': precision, 'recall': recall, 'kappa': kappa}

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
    
    # Build net
    log = f'Build model -> {config.model.name}\n'
    net = classifier_model(**config.model)
    net.to(device) 

    # Optimizers
    log += f'Build optimizer -> {config.optimizer.name}\n'
    optimizer = make_optimizer(net.parameters(), **config.optimizer)
    log += f'Build scheduler -> {config.scheduler.name}\n'
    scheduler = make_scheduler(optimizer, **config.scheduler)
    print(log)
    with open(txtfile, "a") as myfile:
        myfile.write(log + '\n')

    # Criterion
    transition_matrix = make_transition_matrix(num_classes=config.model.num_classes, noise_rate=args.noise_rate, noise_type=args.noise_type)
    transition_matrix = torch.from_numpy(transition_matrix).to(torch.float32).to(device)
    criterion = RobustCrossEntropyLoss(T=transition_matrix)

    # Training
    save_results = {'train':{}, 'val':{}, 'test':{}}
    best_val_results = {'train': {}, 'val': {}, 'test':{}}
    last_test_ten_epoch_results = {'train': {}, 'val': {}, 'test':{}}
    for metric in config.training.metrics:
        for mode in ['train', 'val', 'test']:
            best_val_results[mode][metric] = 0
            last_test_ten_epoch_results[mode][metric] = []
            save_results[mode][metric] = []
    all_results = {}
    for epoch in range(config.training.max_epoch):

        all_results['train'] = train(
                        net=net,
                        loader=train_loader,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        criterion=criterion,
                        num_classes=config.model.num_classes,
                        device=device
                        )
                
        with torch.no_grad():
            all_results['val'] = eval(
                            net=net,
                            loader=val_loader,
                            criterion=criterion,
                            num_classes=config.model.num_classes,
                            device=device,
                            )

            all_results['test'] = eval(
                            net=net,
                            loader=test_loader,
                            criterion=criterion,
                            num_classes=config.model.num_classes,
                            device=device,
                            )
        
        # Log writer
        for mode in ['train', 'val', 'test']:
            for metric in config.training.metrics:
                writer.add_scalar(f'{metric}/{mode}_{metric}', all_results[mode][metric], epoch)
                save_results[mode][metric].append(all_results[mode][metric])

        # Log text
        log = f'Fold{args.fold} [Epoch {epoch+1}/{config.training.max_epoch}]'
        for mode in ['train', 'val', 'test']:
            for metric in config.training.metrics:
                log += ' ' + f'[{mode}_{metric}: {all_results[mode][metric]:.4f}]'
        print(log)
        with open(txtfile, "a") as myfile:
            myfile.write(log + '\n')
        
        # Save weight        
        if epoch >= (config.training.max_epoch - 10):
            for mode in ['train', 'val', 'test']:
                for metric in config.training.metrics:
                    last_test_ten_epoch_results[mode][metric].append(all_results[mode][metric])
        
    # Save weight of last epoch
    print('Last epoch, Saving model ...')
    torch.save(net.state_dict(), workdir.joinpath('ckpt','net_last.ckpt'))
    torch.save(optimizer.state_dict(), workdir.joinpath('ckpt', 'optim_last.ckpt'))
    
    result = f'Best validation results->'
    for mode in ['train', 'val', 'test']:
        for metric in config.training.metrics:
            result += ' ' + f'[{mode}_{metric}: {best_val_results[mode][metric]:.4f}]'
    result += '\n'
    result += f'Last ten epochs average results ->'
    for mode in ['train', 'val', 'test']:
        for metric in config.training.metrics:
            result += ' ' + f'[{mode}_{metric}: {np.mean(last_test_ten_epoch_results[mode][metric]):.4f}]' 
    print(result)
    with open(txtfile, "a") as myfile:
        myfile.write(result + '\n')
    
    with open(workdir / 'save_results.pickle', 'wb') as f:
        pickle.dump(save_results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./script/f_correction/config/f_correction.yaml', help='(.yaml)')
    parser.add_argument('--workdir', type=str, default='./expr/f_correction/')
    parser.add_argument('--data_name', type=str, default='limuc')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--noise_type', type=str, default='quasi', choices=['quasi', 'truncated'])
    parser.add_argument('--noise_rate', type=float, default=0.2, choices=[0.2,0.4])
    parser.add_argument('--seed', type=int, default=777)
    args = parser.parse_args()
    main(args)