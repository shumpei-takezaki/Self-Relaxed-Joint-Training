# Self-Relaxed Joint Training: Sample Selection for Severity Estimation with Ordinal Noisy Labels [[S.Takezaki+, WACV2025](https://arxiv.org/submit/5961885)]

Shumpei Takezaki, Kiyohito Tanaka, Seiichi Uchida

![Illustration](./src/overview.png)

>Severity level estimation is a crucial task in medical image diagnosis. However, accurately assigning severity class labels to individual images is very costly and challenging. Consequently, the attached labels tend to be noisy. In this paper, we propose a new framework for training with ``ordinal'' noisy labels. Since severity levels have an ordinal relationship, we can leverage this to train a classifier while mitigating the negative effects of noisy labels. Our framework uses two techniques: clean sample selection and dual-network architecture. A technical highlight of our approach is the use of soft labels derived from noisy hard labels. By appropriately using the soft and hard labels in the two techniques, we achieve more accurate sample selection and robust network training. The proposed method outperforms various state-of-the-art methods in experiments using two endoscopic ulcerative colitis (UC) datasets and a retinal Diabetic Retinopathy (DR) dataset.

## Requirements
- [Ubuntu 20.04 Desktop](https://ubuntu.com/download)
- [Python](https://www.python.org/) and [Pytorch](https://pytorch.org/)
- [Docker](https://www.docker.com/)


## Create a development environment
```bash
$ sh docker/build.sh
$ sh docker/run.sh
$ sh docker/exec.sh
```

## Download clean dataset ([LIMUC](https://zenodo.org/records/5827695#.YuNBddLP1hH)) and create noisy dataset
Please see sample code `run_dataset.sh`

### Download clean dataset
```bash
$ python3.9 dataset/download_limuc.py
$ python3.9 dataset/data_info_limuc.py # Informatin of image path and label for creating noisy dataset 
```

### Create noisy dataset
Parameters:
- `<noise type>`: [quasi, truncated]
- `<noise rate>`: [0.2, 0.4]

```bash
$ python3.9 dataset/csv_5fold.py
$ python3.9 dataset/noisy_csv_5fold.py --noise_type=<noise type> --noise_rate=<noise rate> 
```

## Experiments (5-fold-validation)
Please see sample code `run_train.sh` \

Parameters:
- `<noise type>`: [quasi, truncated]
- `<noise rate>`: [0.2, 0.4]
- `<method>`: [standard, sord, f_correction, reweight, mixup, cdr, garg, co_teaching, co_teaching_ours, co_teaching_abl, jocor, jocor_ours, codis, codis_ours]

```bash
python3.9 script/method/train.py --workdir=./expr/method --data_name=limuc --config=./script/method/config/method.yaml --noise_type=noise_type --noise_rate=noise_rate --fold=1 
python3.9 script/method/train.py --workdir=./expr/method --data_name=limuc --config=./script/method/config/method.yaml --noise_type=noise_type --noise_rate=noise_rate --fold=2 
python3.9 script/method/train.py --workdir=./expr/method --data_name=limuc --config=./script/method/config/method.yaml --noise_type=noise_type --noise_rate=noise_rate --fold=3 
python3.9 script/method/train.py --workdir=./expr/method --data_name=limuc --config=./script/method/config/method.yaml --noise_type=noise_type --noise_rate=noise_rate --fold=4
python3.9 script/method/train.py --workdir=./expr/method --data_name=limuc --config=./script/method/config/method.yaml --noise_type=noise_type --noise_rate=noise_rate --fold=5
python3.9 script/method/test.py --workdir=./expr/method --data_name=limuc --config=./script/method/config/method.yaml --noise_type=noise_type --noise_rate=noise_rate
```