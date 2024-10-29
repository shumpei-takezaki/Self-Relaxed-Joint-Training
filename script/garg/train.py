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
from torchvision.models import resnet18
from dataset.noisy_csv_5fold import make_transition_matrix

class OR_Model(nn.Module):
    def __init__(self, name='resnet18', pretrained=True,  num_classes=3, **kwargs) -> None:
        super().__init__()
        self.model = resnet18(pretrained=pretrained)
        in_features = self.model.fc.in_features 
        self.model.fc = nn.Linear(in_features, 1, bias=False)
        self.linear_bias = nn.Parameter(torch.zeros(num_classes).float())
    
    def forward(self, x):
        x = self.model(x)
        x = x + self.linear_bias
        # probas = torch.sigmoid(x)
        return x

def label_from_logits(logits):
    """Converts logits to class labels.
    This is function is specific to CORAL.
    """
    probas = torch.sigmoid(logits)
    predict_levels = probas > 0.5
    predicted_labels = torch.sum(predict_levels, dim=1)
    return predicted_labels

def coral_loss(logits, levels):
    val = (-torch.sum((F.logsigmoid(logits)*levels
                        + (F.logsigmoid(logits) - logits)*(1-levels)),
            dim=2))
    
    return val

class GargLoss(nn.Module):
    """CrossEntropyLoss class for noisy labels
    Args:
        T (Tensor): Row-Stochaistic transition matrix for the noise, shape (CxC)
    """
    def __init__(self, T, num_classes = 4) -> None:
        super(GargLoss, self).__init__()
        self.T = T

        Z = [[1]*int(label) + [0]*(num_classes - 1 - int(label)) for label in range(num_classes)]
        self.Z = torch.tensor(Z).to(torch.float32).to(self.T.device)
        self.num_classes = num_classes

    def forward(self, logits, target, reduction = 'mean'):
        batch_z = torch.stack([self.Z]*logits.size(0), dim=0)
        batch_logits = torch.stack([logits]*self.num_classes, dim=1) 

        batch_loss = coral_loss(batch_logits, batch_z)
        batch_loss = batch_loss.unsqueeze(dim=-1)
        
        l =  self.T @ batch_loss
        loss = torch.sum(target * l.squeeze(), dim=1)

        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'sum':
            return torch.sum(loss)
        elif reduction == 'none':
            return loss 

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
        loss = criterion(outputs, labels_onehot, reduction='sum')
        loss = loss / n_batch

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = label_from_logits(outputs)
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
        loss = criterion(outputs, labels_onehot, reduction='sum')

        pred = label_from_logits(outputs)
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
    net = OR_Model(**config.model)
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
    transition_matrix = make_transition_matrix(num_classes=config.model.num_classes+1, noise_rate=args.noise_rate, noise_type=args.noise_type)
    inverse_matrix = np.linalg.inv(transition_matrix)
    inverse_matrix = torch.from_numpy(inverse_matrix).to(torch.float32).to(device)
    criterion = GargLoss(T=inverse_matrix, num_classes=config.model.num_classes+1)

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
                        num_classes=config.model.num_classes+1,
                        device=device
                        )
                
        with torch.no_grad():
            all_results['val'] = eval(
                            net=net,
                            loader=val_loader,
                            criterion=criterion,
                            num_classes=config.model.num_classes+1,
                            device=device,
                            )

            all_results['test'] = eval(
                            net=net,
                            loader=test_loader,
                            criterion=criterion,
                            num_classes=config.model.num_classes+1,
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
    parser.add_argument('--config', type=str, default='./script/garg/config/garg.yaml', help='(.yaml)')
    parser.add_argument('--workdir', type=str, default='./expr/garg/')
    parser.add_argument('--data_name', type=str, default='limuc')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--noise_type', type=str, default='quasi', choices=['quasi', 'truncated'])
    parser.add_argument('--noise_rate', type=float, default=0.2, choices=[0.2,0.4])
    parser.add_argument('--seed', type=int, default=777)
    args = parser.parse_args()
    main(args)