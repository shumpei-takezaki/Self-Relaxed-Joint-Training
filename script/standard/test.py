import argparse
import pickle
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np

def main(args):
    workdir = Path(args.workdir).joinpath(args.data_name, f'{args.noise_type}-{args.noise_rate}')
    config = OmegaConf.load(args.config)

    results = {'train': {}, 'val': {}, 'test':{}}
    for mode in ['train','val','test']:
        for metric in config.training.metrics:
                results[mode][metric] = []
    
    for k in range(5):
        k_workdir = workdir.joinpath(f'fold{k+1}')    
        with open(k_workdir / 'save_results.pickle', 'rb') as f:
            k_results = pickle.load(f)

        for mode in ['train','val','test']:
            for metric in config.training.metrics:
                results[mode][metric].append(np.mean(k_results[mode][metric][-10:]))

        log = f'{args.data_name}-{args.noise_type}-{args.noise_rate}\n'
        for mode in ['train','val','test']:
            log += f'{mode} results ->'
            for metric in config.training.metrics:
                mean = np.mean(results[mode][metric])
                std = np.std(results[mode][metric])
                log += ' ' + f'[{metric}] {mean:.3f}±{std:.3f}'
            log += '\n'
        
        print(log) 
        with open(workdir / 'results.txt', "w") as myfile:
            myfile.write(log)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./script/standard/config/standard.yaml', help='(.yaml)')
    parser.add_argument('--workdir', type=str, default='./expr/standard/')
    parser.add_argument('--data_name', type=str, default='limuc')
    parser.add_argument('--noise_type', type=str, default='quasi', choices=['quasi', 'truncated'])
    parser.add_argument('--noise_rate', type=float, default=0.2, choices=[0.2,0.4])
    args = parser.parse_args()
    main(args)