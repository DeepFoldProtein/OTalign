#!/usr/bin/env bash
#SBATCH --job-name=msa_hhsuite
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --array=0-9999   # set after creating the file list

set -euo pipefail

# ---- user config ----
FASTA_DIR="${FASTA_DIR:-work/fasta}"          # input single-sequence FASTA files (*.fasta)
A3M_DIR="${A3M_DIR:-work/a3m}"
HHM_DIR="${HHM_DIR:-work/hhm}"
FILELIST="${FILELIST:-work/fasta.list}"       # list of absolute paths to fasta files
# HHsuite database root (prefix without _a3m/_cs219 suffix, as per your installation)
HHDB="${HHDB:-/path/to/uniclust30/uniclust30_2018_08/uniclust30_2018_08}" 
NCPU="${NCPU:-8}"
ROUNDS="${ROUNDS:-2}"
EVALUE="${EVALUE:-1e-3}"
MAXSEQ="${MAXSEQ:-65535}"   # cap sequences in MSA to keep size reasonable
# ---------------------

mkdir -p "$(dirname "$FILELIST")" "$A3M_DIR" "$HHM_DIR" logs

# Build file list once (on array index 0), if not exists
if [[ ! -s "$FILELIST" ]]; then
  find "$FASTA_DIR" -type f -name "*.fasta" | sort > "$FILELIST"
  LINES=$(wc -l < "$FILELIST")
  echo "[info] Wrote $LINES entries to $FILELIST"
  echo "[warn] Update your #SBATCH --array=0-$(($LINES-1)) and resubmit."
  exit 0
fi

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$FILELIST")
if [[ -z "${LINE:-}" ]]; then
  echo "[warn] No line for index $SLURM_ARRAY_TASK_ID"
  exit 0
fi

IN="$LINE"
BASENAME="$(basename "$IN" .fasta)"
A3M="$A3M_DIR/$BASENAME.a3m"
HHM="$HHM_DIR/$BASENAME.hhm"

echo "[info] ($SLURM_ARRAY_TASK_ID) input=$IN -> $A3M | $HHM"

# 1) hhblits -> A3M
if [[ ! -s "$A3M" ]]; then
  hhblits -i "$IN" -d "$HHDB" -oa3m "$A3M" \
          -e "$EVALUE" -cov 0 -qid 0 -n "$ROUNDS" -cpu "$NCPU" -maxseq "$MAXSEQ"
fi

# 2) hhmake -> HHM
if [[ ! -s "$HHM" ]]; then
  hhmake -i "$A3M" -o "$HHM" -cpu "$NCPU"
fi

echo "[ok] ($SLURM_ARRAY_TASK_ID) done."
