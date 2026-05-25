#!/usr/bin/env bash
# Submit hhblits+hhmake slurm array for the ECOD30 hard benchmark.
#
# Prereq:
#   python scripts/prepare_ecod30_hhblits_inputs.py
#
# Outputs:
#   data/hhsuite/ecod30_hard/a3m/<id>.a3m
#   data/hhsuite/ecod30_hard/hhm/<id>.hhm

set -euo pipefail

# Load .env (HHDB, HHSUITE_SIF, SING_BINDS, optional dir overrides)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$HERE/.env" ]]; then
  set -a; source "$HERE/.env"; set +a
else
  echo "ERROR: $HERE/.env not found. Copy .env.example and fill in HHDB/HHSUITE_SIF/SING_BINDS." >&2
  exit 1
fi

ROOT="data/hhsuite/ecod30_hard"
FASTA_DIR="${FASTA_DIR:-$ROOT/fasta}"
A3M_DIR="${A3M_DIR:-$ROOT/a3m}"
HHM_DIR="${HHM_DIR:-$ROOT/hhm}"
FILELIST="${FILELIST:-$ROOT/fasta.list}"

: "${HHDB:?HHDB must be set in .env}"
: "${HHSUITE_SIF:?HHSUITE_SIF must be set in .env}"

# Slurm needs the log directory to exist before it can open --output/--error files.
mkdir -p logs

# Resource per array task (override at the shell):
#   CPUS_PER_TASK=24 MEM=24G bash scripts/submit_ecod30_hhblits.sh
# Node has 96 cores / 1 TB; with 16 cores/16 GB → 6 tasks concurrent.
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM="${MEM:-16G}"

if [[ ! -s "$FILELIST" ]]; then
  echo "ERROR: $FILELIST is missing or empty. Run scripts/prepare_ecod30_hhblits_inputs.py first." >&2
  exit 1
fi

N=$(wc -l < "$FILELIST")
LAST=$((N - 1))

# MaxArraySize (slurm.conf) typically caps an array submission. Auto-chunk.
# Override with CHUNK env var if needed (e.g. CHUNK=500 bash ...).
MAX_ARRAY=$(scontrol show config 2>/dev/null | awk -F= '/MaxArraySize/ {gsub(/ /,"",$2); print $2}')
MAX_ARRAY="${MAX_ARRAY:-1000}"
CHUNK="${CHUNK:-$MAX_ARRAY}"
# Throttle: max concurrent tasks per chunk (e.g. CONCURRENT=200)
CONCURRENT="${CONCURRENT:-}"

echo "Total tasks: $N (indices 0-$LAST). MaxArraySize=$MAX_ARRAY, chunk=$CHUNK"

# MaxArraySize caps both the chunk size AND each array index, so every
# submission has to use 0-based indices. We pass ARRAY_OFFSET so the slurm
# script can map back to the right line in FILELIST.
i=0
while (( i <= LAST )); do
  end=$((i + CHUNK - 1))
  (( end > LAST )) && end=$LAST
  count=$((end - i + 1))
  spec="0-$((count - 1))"
  [[ -n "$CONCURRENT" ]] && spec+="%${CONCURRENT}"
  echo "  -> sbatch --array=$spec  (lines $i-$end, ARRAY_OFFSET=$i)"
  sbatch --array="$spec" \
    --cpus-per-task="$CPUS_PER_TASK" --mem="$MEM" \
    --export=ALL,FASTA_DIR="$FASTA_DIR",A3M_DIR="$A3M_DIR",HHM_DIR="$HHM_DIR",FILELIST="$FILELIST",HHDB="$HHDB",HHSUITE_SIF="$HHSUITE_SIF",SING_BINDS="${SING_BINDS:-}",ARRAY_OFFSET="$i" \
    scripts/slurm_hhblits_hhmake_array.sh
  i=$((end + 1))
done
