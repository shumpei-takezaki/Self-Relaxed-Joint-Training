#!/bin/bash
docker exec -itd srjt tensorboard --logdir=. --host=0.0.0.0 --port=${@-6006}
