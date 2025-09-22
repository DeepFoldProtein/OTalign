#!/bin/bash

# =================================================================================
# SLURM Script for Distributed Data Parallel (DDP) Training with PyTorch
#
# Usage: sbatch scripts/slurm_ssp_train.sh [args...]
#
# Example:
# sbatch scripts/slurm_ssp_train.sh \
#    --model work/checkpoints/esm1b-lora-finetune-5/checkpoint-epoch-10 \
#    --base_model_for_checkpoint ESM1b_33_650M \
#    --batch_size 4 \
#    --epoch 10 --lr 1e-3 \
#    --outdir work/ssp/esm1b-lora-finetune-5
# =================================================================================

# --- SLURM JOB CONFIGURATION ---
#SBATCH --job-name=ssp_task           # Name for the job
#SBATCH --nodes=1                     # Number of nodes to request
#SBATCH --ntasks-per-node=2           # Number of processes per node (should equal --gres=gpu)
#SBATCH --gres=gpu:2                  # Number of GPUs to request per node
#SBATCH --cpus-per-task=8             # Number of CPU cores per process (for data loading, etc.)
#SBATCH --time=24:00:00               # Maximum runtime for the job (DD:HH:MM:SS)
#SBATCH --partition=normal            # SLURM partition to submit the job to (e.g., gpu, volta, etc.)
#SBATCH --output=work/ssp_job_%j.out  # Path for standard output log
#SBATCH --error=work/ssp_job_%j.err   # Path for standard error log

# --- ENVIRONMENT SETUP ---

# Load necessary modules for your cluster environment (e.g., anaconda, cuda)
# This part is cluster-specific. You might need to uncomment and modify.
# echo "Loading modules..."
# module load anaconda3
# module load cuda/11.8
export CUDA_HOME="/store/deepfold/apps/cuda-12.8.1"

# Activate your Python/Conda environment
source .venv/bin/activate

# --- DDP ENVIRONMENT VARIABLES ---
# PyTorch DDP requires these environment variables to be set for process group initialization.
# MASTER_ADDR: The IP address of the master node (rank 0).
# MASTER_PORT: A free port on the master node.
# WORLD_SIZE: The total number of processes across all nodes.
# RANK: The global rank of the current process.

# SLURM provides the list of nodes. We'll use the first node as the master.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# We'll use a random free port.
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo "=========================================================="
echo "SLURM JOB ID: $SLURM_JOBID"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "Number of nodes: $SLURM_NNODES"
echo "Number of tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total number of tasks: $SLURM_NTASKS"
echo "=========================================================="

# --- RUN THE TRAINING SCRIPT ---
# 'srun' will launch 'ntasks-per-node' copies of the command on each of the 'nodes'.
# The 'train.py' script is expected to handle DDP initialization internally
# using the environment variables (MASTER_ADDR, etc.) and SLURM variables (SLURM_PROCID, etc.).

echo "[$(date)] Starting SSP training..."
echo "Arguments passed to script: $@"
accelerate launch --config_file accelerate_config.yaml scripts/run_secondary_structure_prediction.py "$@"
echo "[$(date)] Training finished."
