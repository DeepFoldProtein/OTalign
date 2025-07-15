#!/bin/bash
#SBATCH --job-name=train               # Job display name
#SBATCH --partition=normal             # Queue / partition on your cluster
#SBATCH --nodes=1                      # Always 1 node for single-GPU jobs
#SBATCH --gres=gpu:1                   # Number of GPUs (change if you need more)
#SBATCH --cpus-per-task=16             # Data-loader workers + PyTorch ops
#SBATCH --mem=48G                      # System RAM
#SBATCH --time=48:00:00                # HH:MM:SS – adjust to your run
#SBATCH --output=logs/%x-%j.out        # Stdout  (%x=job-name, %j=job-ID)
#SBATCH --error=logs/%x-%j.err         # Stderr

#
source .venv/bin/activate

# CLI
CONFIG_YAML=${1:-config.yaml}
if [[ ! -f "${CONFIG_YAML}" ]]; then
    echo "[ERROR] Config file '${CONFIG_YAML}' not found." >&2
    exit 1
fi
echo "[INFO] Using config: ${CONFIG_YAML}"

# Safety: fail on any error after this point
set -euo pipefail

#
# export TORCH_HOME=$SCRATCH/torch_cache # avoid repeated downloads
# export TRANSFORMERS_OFFLINE=1          # prevent HF hub calls
export WANDB_SERVICE_WAIT=300 # wandb robustness in Slurm

#
python -u train.py --config "${CONFIG_YAML}"
