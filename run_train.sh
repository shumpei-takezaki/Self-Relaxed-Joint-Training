python3.9 script/co_teaching_ours/train.py --workdir=./expr/co_teaching_ours --data_name=limuc --config=./script/co_teaching_ours/config/co_teaching_ours.yaml --noise_type=quasi --noise_rate=0.2 --fold=1 
python3.9 script/co_teaching_ours/train.py --workdir=./expr/co_teaching_ours --data_name=limuc --config=./script/co_teaching_ours/config/co_teaching_ours.yaml --noise_type=quasi --noise_rate=0.2 --fold=2 
python3.9 script/co_teaching_ours/train.py --workdir=./expr/co_teaching_ours --data_name=limuc --config=./script/co_teaching_ours/config/co_teaching_ours.yaml --noise_type=quasi --noise_rate=0.2 --fold=3 
python3.9 script/co_teaching_ours/train.py --workdir=./expr/co_teaching_ours --data_name=limuc --config=./script/co_teaching_ours/config/co_teaching_ours.yaml --noise_type=quasi --noise_rate=0.2 --fold=4
python3.9 script/co_teaching_ours/train.py --workdir=./expr/co_teaching_ours --data_name=limuc --config=./script/co_teaching_ours/config/co_teaching_ours.yaml --noise_type=quasi --noise_rate=0.2 --fold=5
python3.9 script/co_teaching_ours/test.py --workdir=./expr/co_teaching_ours --data_name=limuc --config=./script/co_teaching_ours/config/co_teaching_ours.yaml --noise_type=quasi --noise_rate=0.2