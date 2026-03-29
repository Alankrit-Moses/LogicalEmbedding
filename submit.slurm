#!/bin/bash
#SBATCH --job-name=plm_train
#SBATCH --output=logs/plm_%j.out
#SBATCH --error=logs/plm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

# Safety check: Ensure directories exist so Slurm doesn't crash silently
mkdir -p logs checkpoints

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

export MASTER_ADDR=$head_node_ip
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS

echo "Starting distributed training on $WORLD_SIZE GPUs..."
srun python train.py