#!/bin/bash

# Configuration for Accelerate
# This assumes the user has already run 'accelerate config'
# or we can provide a default config for 1-4 GPUs.

# To run on 4 GPUs:
# accelerate launch --num_processes 4 train.py --batch_size 64 --epochs 10 --lr 3e-5

# Default execution (will use whatever is configured via 'accelerate config')
accelerate launch train.py \
    --batch_size 64 \
    --epochs 5 \
    --lr 2e-5 \
    --alpha 1.0 \
    --beta 1.0 \
    --gamma 0.1 \
    --output_dir "./checkpoints"
