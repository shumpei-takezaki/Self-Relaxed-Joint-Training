import os
import argparse

import numpy as np
import pandas as pd

from script.utils.utils import fix_seed

def multiclass_noisify(y, P, random_state=1):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
#    print (np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    np.testing.assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]
    return new_y

def eta_ij(i,j,noise_level=0.15):
    return noise_level / np.absolute(i-j)

def quasi_noise(num_classes=4, noise_level=0.15):
    """
    Refference: Robust Deep Ordinal Regression under Label Noise
    URL: https://proceedings.mlr.press/v129/garg20a/garg20a.pdf
    Uniformly inversely decaying noise
    """
    transition_matrix = np.eye(num_classes)
    for i in range(num_classes):
        for j in range(num_classes):
            if i==j:
                transition_matrix[i,j] = 1. - np.sum([eta_ij(i,k,noise_level) if i!=k else 0. for k in range(num_classes)])
            else:
                transition_matrix[i,j] = eta_ij(i,j,noise_level)
    
    return transition_matrix 

def truncated_noise(num_classes=4, noise_level=0.1):
    transition_matrix = np.eye(num_classes)
    for i in range(num_classes):
        for j in range(num_classes):
            if i==j:
                if (i==0) or (i==(num_classes-1)):
                    transition_matrix[i,j] = 1. - noise_level
                else:
                    transition_matrix[i,j] = 1. - noise_level * 2.
            else:
                if (i==0) or (i==(num_classes-1)):
                    if np.absolute(i-j) == 1:
                        transition_matrix[i,j] = noise_level
                    else:
                        transition_matrix[i,j] = 0.
                else:
                    if np.absolute(i-j) == 1:
                        transition_matrix[i,j] = noise_level
                    else:
                        transition_matrix[i,j] = 0.
    return transition_matrix

    return soft_label
def make_transition_matrix(num_classes=4, noise_rate=0.2, noise_type='quasi'):
    if noise_type == 'quasi':
        if noise_rate == 0.2:
            noise_level = 0.1
        elif noise_rate == 0.4:
            noise_level = 0.2
        return quasi_noise(num_classes=num_classes, noise_level=noise_level)
    elif noise_type == 'truncated':
        if noise_rate == 0.2:
            noise_level = 0.15
        elif noise_rate == 0.4:
            noise_level = 0.3
        return truncated_noise(num_classes=num_classes, noise_level=noise_level)

def main(args):
    fix_seed(seed=args.seed)
    for k in range(5):
        save_dir = os.path.join(args.save_dir, f"{args.noise_type}-{args.noise_rate}", f"fold{k+1}")
        os.makedirs(save_dir, exist_ok=True)
        for t in ['train', 'val', 'test']:
            csv_path = os.path.join(args.root_dir, f"fold{k+1}", f"{t}_fold{k+1}.csv")
            csv_file = pd.read_csv(csv_path, index_col=0)
            if (t == 'train') or (t=='val'):
                transition_matrix = make_transition_matrix(num_classes=args.num_classes, noise_rate=args.noise_rate, noise_type=args.noise_type)
                print(transition_matrix)
                clean_label = csv_file['label'].values
                noisy_label = multiclass_noisify(clean_label, transition_matrix, args.seed)
                csv_file['noisy_label'] = np.array(noisy_label)
            csv_file.to_csv(os.path.join(save_dir, f'{t}_fold{k+1}.csv'))
            noise_rate = np.sum(np.transpose(noisy_label)!=np.transpose(clean_label)) / len(noisy_label)
            print(f'Acctually nois rate -> {noise_rate}')
            print(f'Save fold{k+1} {t}')
    print('finish!!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='./dataset/limuc/5fold/clean/')
    parser.add_argument('--save_dir', type=str, default='./dataset/limuc/5fold/')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--noise_type', type=str, default='quasi', choices=['quasi', 'truncated'])
    parser.add_argument('--noise_rate', type=float, default=0.2, choices=[0.2, 0.4])
    parser.add_argument('--seed', type=int, default=777)
    args = parser.parse_args()

    main(args)