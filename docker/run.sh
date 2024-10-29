#!/bin/bash
docker run \
    -d \
    --init \
    --rm \
    -p 15000:5000 \
    -p 16006:6006 \
    -p 18501-18511:8501-8511 \
    -p 18888:8888 \
    -it \
    --gpus=all \
    --ipc=host \
    --name=srjt \
    --env-file=.env \
    --volume=$PWD:/workspace \
    ordinal_noisy:latest \
    ${@-fish}
