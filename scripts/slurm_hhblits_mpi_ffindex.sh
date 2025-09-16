#!/usr/bin/env bash
#SBATCH --job-name=hhblits_mpi_ffidx
#SBATCH --output=work/%x_%j.out
#SBATCH --error=work/%x_%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1                   # adjust
#SBATCH --ntasks-per-node=8         # MPI ranks per node
#SBATCH --cpus-per-task=4           # threads per rank (hhblits -cpu)
#SBATCH --time=48:00:00

set -euo pipefail

# MPI Environment Setup
export I_MPI_FABRICS=shm:ofi
export FI_PROVIDER=tcp
export SLURM_MPI_TYPE=pmi2
module purge > /dev/null
module load gnu12 > /dev/null
module load intel/mpi > /dev/null

# Python Setup
source .venv/bin/activate
# pyenv activate plm-bench || true
echo "[info] python: $(which python)"; python --version || true

# --- user config ---
FF="${FF:-work/queries}"
OUT_DIR="${OUT_DIR:-work/a3m_mpi}"
HHDM="${HHDM:-/shared/db/uniclust30_2018_08/uniclust30_2018_08}"
CPU_PER_TASK="${SLURM_CPUS_PER_TASK:-4}"
EVALUE="${EVALUE:-1e-3}"
ROUNDS="${ROUNDS:-2}"
COV="${COV:-0.0}"
MAXSEQ="${MAXSEQ:-65535}"
HHBLITS_MPI="${HHBLITS_MPI:-hhblits_mpi}"

mkdir -p "$OUT_DIR" logs

# Stage DB to node-local storage for I/O bandwidth
#NODE_DB="$SLURM_TMPDIR/db"
#if [[ -z "${SLURM_TMPDIR:-}" ]]; then
#  NODE_DB="$TMPDIR/db"
#fi
#mkdir -p "$NODE_DB"
#echo "[info] staging DB to $NODE_DB ..."
#rsync -a --delete "${HHDM%/}/" "$NODE_DB/"

# Each rank writes outputs into OUT_DIR; many builds of hhblits-mpi can produce per-query files under -oa3m-dir
# If your build does not support that, see plan B below.
OA3M_DIR_OPT=("-oa3m" "$OUT_DIR")
CMD_BASE=(
  "$HHBLITS_MPI"
  -i "$FF"
  -d "$HHDM"   # adjust prefix; same as hhblits -d <prefix>
  "${OA3M_DIR_OPT[@]}"
  -e "$EVALUE" -cov "$COV" -n "$ROUNDS" -cpu "$CPU_PER_TASK" -maxseq "$MAXSEQ"
)

echo "[info] running: srun ${CMD_BASE[*]}"
srun "${CMD_BASE[@]}"

echo "[ok] done; outputs in $OUT_DIR"

