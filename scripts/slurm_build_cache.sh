#!/usr/bin/env bash
#SBATCH --job-name=build_npz_cache
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu           # change to your cluster's GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=8                # total ranks (world size)
#SBATCH --gpus-per-task=1         # one GPU per rank
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=16G

# ---- user config (edit as needed) ----
export HF_DATASET="DeepFoldProtein/SABmark"
export HF_NAME="twi"
export HF_SPLIT="test"

export MODEL="AnkhCL"          # choices: ESM2 | AnkhCL
export DTYPE="fp32"               # fp16 | fp32
export OUTPUT_ROOT=".cache"   # directory where shards will be saved

export BATCH_SIZE=8               # sequences per forward pass
export SHARD_SIZE=2000            # rows per NPZ shard (per rank)
# --------------------------------------

set -euo pipefail

mkdir -p logs

# Activate your environment
source "../.venv/bin/activate"

echo "[info] Launching with ${SLURM_NTASKS} ranks"

srun --label \
  python build_cache_npz_ranked.py \
    --dataset "${HF_DATASET}" \
    --name "${HF_NAME}" \
    --split "${HF_SPLIT}" \
    --model "${MODEL}" \
    --dtype "${DTYPE}" \
    --output_root "${OUTPUT_ROOT}" \
    --batch_size "${BATCH_SIZE}" \
    --shard_size "${SHARD_SIZE}"
