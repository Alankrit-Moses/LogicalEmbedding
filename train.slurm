#!/bin/bash
#SBATCH --job-name=plm_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:2             # Request 2 H100s (adjust 1-4 as needed)
#SBATCH --time=12:00:00
#SBATCH --partition=gpu               # Replace with your cluster's partition name

# 1. Load Modules (if required by your cluster)
# module load cuda/12.1

# 2. Activate Conda Environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate torch-env

# 3. Create Logs Directory
mkdir -p logs

# 4. Distributed Training with Accelerate
# Accelerate automatically detects SLURM environment variables
# We use --num_processes to match the number of GPUs requested in --gres
export NUM_GPUS=$(echo $SLURM_STEP_GPUS | tr "," "\n" | wc -l)
if [ -z "$SLURM_STEP_GPUS" ]; then
    # Fallback if SLURM_STEP_GPUS is not set (depends on SLURM version/config)
    export NUM_GPUS=4 
fi

echo "Running on $NUM_GPUS GPUs"

# Using 'srun' to launch the training process
# Accelerate handles the multi-process coordination
srun accelerate launch \
    --multi_gpu \
    --num_processes $NUM_GPUS \
    --mixed_precision fp16 \
    train.py \
    --batch_size 64 \
    --epochs 5 \
    --lr 2e-5 \
    --output_dir "./checkpoints"

# Note: If you haven't run 'accelerate config' on the head node, 
# accelerate will use default distributed settings which work well with SLURM.
