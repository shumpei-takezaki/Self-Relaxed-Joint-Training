NOISE_TYPE=${NOISE_TYPE:-"quasi"}
NOISE_RATE=${NOISE_RATE:-0.2}
METHOD=${METHOD:-"co_teaching_ours"}

# train
for FOLD in 1 2 3 4 5
do
    python3.9 src/$METHOD/train.py --workdir=./expr/$METHOD --data_name=limuc --config=./script/$METHOD/config/$METHOD.yaml --noise_type=$NOISE_TYPE --noise_rate=$NOISE_RATE --fold=$FOLD
done

# test
python3.9 src/$METHOD/test.py --workdir=./expr/$METHOD --data_name=limuc --config=./script/$METHOD/config/$METHOD.yaml --noise_type=$NOISE_TYPE --noise_rate=$NOISE_RATE