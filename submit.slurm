#!/bin/bash
#SBATCH --job-name=plm_train
#SBATCH --output=logs/plm_%j.out
#SBATCH --error=logs/plm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

# 1. Activate Conda Environment safely inside a Slurm batch script
eval "$(conda shell.bash hook)"
conda activate torch-env

# Safety check: Ensure directories exist
mkdir -p logs checkpoints

# 2. Extract Master Node IP (Simplified for Single Node)
# Since you are requesting exactly 1 node, we don't need srun to ask for the IP.
# We can just ask the machine directly, bypassing the first MPI error.
head_node_ip=$(hostname -I | awk '{print $1}')

export MASTER_ADDR=$head_node_ip
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS

# 3. Disable MPI to fix the 'pmix_v3' crash
export SLURM_MPI_TYPE=none

echo "Starting distributed training on $WORLD_SIZE GPUs at $head_node_ip..."

# 4. Launch the Training Script using srun
# Adding --mpi=none forces srun to launch the processes nakedly, 
# letting PyTorch's NCCL handle the communication.
srun --mpi=none python train.py