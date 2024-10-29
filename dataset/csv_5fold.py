import os

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from script.utils.utils import fix_seed

if __name__ == '__main__':
    data_name = 'limuc'
    save_folder = f'./dataset/{data_name}/5fold/clean'
    os.makedirs(save_folder, exist_ok=True)

    seed = 777
    fix_seed(seed=seed)

    csv_file = pd.read_csv(f'./dataset/{data_name}/dataset_info_{data_name}.csv', index_col=0)

    paitients_with_max_Mayo = []
    patients_with_images_path = []

    for s_n in range(csv_file['sequence_num'].max()):
        if s_n+1 in csv_file['sequence_num'].values:
            patients_info = csv_file[csv_file['sequence_num']==s_n+1]

            patients_with_images_path.append(list(patients_info['path']))

            label = patients_info['label'].max()
            paitients_with_max_Mayo.append(label)
        else:
            pass

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    k_fold_index = []
    for train_index, test_index in skf.split(X=patients_with_images_path, y=paitients_with_max_Mayo):
        k_fold_index.append(test_index)
    
    train = [[0,1,2], [1,2,3], [2,3,4], [0,3,4], [0,1,4]]
    val = [3,4,0,1,2]
    test = [4,0,1,2,3]
    for k in range(5):
        os.makedirs(f'{save_folder}/fold{k+1}/', exist_ok=True)
        tmp_pd = []
        for i in train[k]:
            tmp_pd.append(csv_file[csv_file['sequence_num'].isin(k_fold_index[i]+1)])
        train_pd = pd.concat(tmp_pd)
        train_pd.to_csv(f'{save_folder}/fold{k+1}/train_fold{k+1}.csv')

        val_pd = csv_file[csv_file['sequence_num'].isin(k_fold_index[val[k]]+1)]
        val_pd.to_csv(f'{save_folder}/fold{k+1}/val_fold{k+1}.csv')

        test_pd = csv_file[csv_file['sequence_num'].isin(k_fold_index[test[k]]+1)]
        test_pd.to_csv(f'{save_folder}/fold{k+1}/test_fold{k+1}.csv')
    
    print('finish')