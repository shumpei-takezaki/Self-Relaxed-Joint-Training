from collections import OrderedDict
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torch.nn.functional as F
from torchvision.transforms import transforms as T


def load_weight(net, path,  parallel=False):
    if parallel:
        state_dict = torch.load(path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
        return net
    else:
        module_dict = torch.load(path)
        net.load_state_dict(module_dict)
        return net

def make_optimizer(params, name, **kwargs):
    # Optimizer
    return optim.__dict__[name](params, **kwargs)

def make_scheduler(optimize, name, **kwargs):
    # Optimizer
    return scheduler.__dict__[name](optimize, **kwargs)

def compute_mae(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = np.mean(np.abs(y_pred - y_true))
    return mae

def compute_rmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    return rmse

def worker_init_fn(seed):
    random.seed(seed)
    np.random.seed(seed)

def get_loader(csv_file, batch_size, noisy=True, train=True, seed=777):
    ds = NoisyDataset(csv_file=csv_file, noisy=noisy, train=train)
    dl = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=True if train else False,
        num_workers=8,
        pin_memory=True,
        drop_last=True if train else False,
        worker_init_fn=worker_init_fn(seed)
    )
    return ds, dl

class NoisyDataset(Dataset):
    def __init__(self, csv_file, noisy=True, train=True):
        super().__init__()
        if train:
            self.transforms = T.Compose([
                T.Resize(256),
                T.RandomCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transforms = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
                ])

        self.csv_file = pd.read_csv(csv_file)
        self.img_path = []
        self.patient_id = []
        self.patient_to_path = {}
        self.noisy_label = []
        self.clean_label = []
        if noisy:
            for path, clean_label, noisy_label in zip(self.csv_file['path'], self.csv_file['label'], self.csv_file['noisy_label']):
                
                self.noisy_label.append(torch.tensor(noisy_label).to(torch.int64))
                self.clean_label.append(torch.tensor(clean_label).to(torch.int64))
                self.img_path.append(path)
        else:
            for path, clean_label in zip(self.csv_file['path'], self.csv_file['label']):
                
                self.clean_label.append(torch.tensor(clean_label).to(torch.int64))
                self.img_path.append(path)
                
            self.noisy_label = self.clean_label

        self.noise_or_not = np.transpose(self.noisy_label)==np.transpose(self.clean_label)
    
    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, index):
        image = Image.open(self.img_path[index]).convert('RGB')
        noisy_label = self.noisy_label[index]
        clean_label = self.clean_label[index]
        image = self.transforms(image)
        return image, noisy_label, clean_label, index

def kl_loss_compute(pred, soft_targets, reduce=False):

    kl = F.kl_div(F.log_softmax(pred, dim=1),F.softmax(soft_targets, dim=1), reduction='none')

    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)

def js_loss_compute(pred, soft_targets, reduce=False):
    
    pred_softmax = F.softmax(pred, dim=1)
    targets_softmax = F.softmax(soft_targets, dim=1)
    mean = (pred_softmax + targets_softmax) / 2
    kl_1 = F.kl_div(F.log_softmax(pred, dim=1), mean, reduction='none')
    kl_2 = F.kl_div(F.log_softmax(soft_targets, dim=1), mean, reduction='none')
    js = (kl_1 + kl_2) / 2 
    
    if reduce:
        return torch.mean(torch.sum(js, dim=1))
    else:
        return torch.sum(js, 1)

def gen_forget_rate(noise_type, noise_level):
    if noise_type == 'inverse':
        if noise_level == 0.1:
            return 0.2
        elif noise_level == 0.2:
            return 0.4
        elif noise_level == 0.15:
            return 0.3
        elif noise_level == 0.075:
            return 0.15
        
    elif noise_type == 'neighbor':
        if noise_level == 0.15:
            return 0.2
        elif noise_level == 0.3:
            return 0.4
        elif noise_level == 0.2:
            return 0.3
        elif noise_level == 0.1:
            return 0.15
        
    elif noise_type == 'ideal':
        return 0.4

def gen_forget_schedule(forget_rate, n_epoch, num_gradual):
    rate_schedule = np.ones(n_epoch) * forget_rate
    rate_schedule[:num_gradual] = np.linspace(0, forget_rate, num_gradual)
    return rate_schedule

def compute_mae(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = np.mean(np.abs(y_pred - y_true))
    return mae

def compute_rmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    return rmse
