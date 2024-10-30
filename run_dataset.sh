NOISE_TYPE=${NOISE_TYPE:-"quasi"}
NOISE_RATE=${NOISE_RATE:-0.2}

python3.9 dataset/download_limuc.py
python3.9 dataset/data_info_limuc.py # Informatin of image path and label for creating noisy dataset 
python3.9 dataset/csv_5fold.py
python3.9 dataset/noisy_csv_5fold.py --noise_type=$NOISE_TYPE --noise_rate=$NOISE_RATE