#!/bin/bash
#SBATCH --job-name=srm_diag_enwik8
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:0
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# 1. Load Modules (if required)
# module load cuda/12.1

# 2. Activate Conda Environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate torch-env

# 3. Distributed Environment Variables
export GPUS_PER_NODE=4
export NNODES=$SLURM_NNODES
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12345

# Fix NCCL and GPU isolation
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
# Ensure each process only sees the GPUs allocated to it
# On many clusters, this is handled by SLURM, but explicitly setting it helps torchrun
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 4. Launch Training
# Using your preferred srun + torchrun pattern
srun --mpi=none torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    train.py \
    --batch_size 64 \
    --epochs 5 \
    --lr 2e-5 \
    --output_dir "./checkpoints"
