import os
from datetime import datetime
import random
import json

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import torch

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_date():
    now = datetime.now()
    date = now.strftime('%Y%m%d_%H%M%S')
    return date

def save_args(args, folder):
    with open(f"{folder}/args.json", mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

def plot_cm(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cmn = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
    sns.heatmap(cmn, square=True, cbar=True, annot=True, cmap='Blues', fmt='.2f', vmin=0, vmax=1)
    plt.xlabel("Prediction", fontsize=13)
    plt.ylabel("Ground Truth", fontsize=13)
    plt.savefig(save_path, bbox_inches='tight')
    plt.clf()
    plt.close()