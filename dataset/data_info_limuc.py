import os
import glob
import pandas as pd
import natsort

def is_folder_empty(folder_path):
    return not bool(os.listdir(folder_path))

def main():
    if not os.path.exists('./dataset/limuc/dataset_info_limuc.csv'):
        root = './dataset/limuc/patient_based_classified_images'
        patient_folder_list = os.listdir(root)
        patient_folder_list = natsort.natsorted(patient_folder_list)
        data = []
        for patient_folder in patient_folder_list:
            frame_num = 1
            sequence_num = int(patient_folder.split('/')[-1])
            for mayo in range(4):
                class_name = f'Mayo {mayo}'
                class_folder =  os.path.join(root, patient_folder, class_name)
                if is_folder_empty(class_folder):
                    continue
                else:
                    path_list = glob.glob(class_folder+'/**')
                    for filename in path_list:
                        data.append([filename, sequence_num, frame_num, mayo])
                        frame_num += 1

        columns = ["path","sequence_num","frame_num", "label"]
        df = pd.DataFrame(data=data, columns=columns)
        df.to_csv('./dataset/limuc/dataset_info_limuc.csv')
        
    print('finish')

if __name__ == '__main__':
    main()