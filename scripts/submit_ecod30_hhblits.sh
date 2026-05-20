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

i=0
while (( i <= LAST )); do
  end=$((i + CHUNK - 1))
  (( end > LAST )) && end=$LAST
  spec="$i-$end"
  [[ -n "$CONCURRENT" ]] && spec+="%${CONCURRENT}"
  echo "  -> sbatch --array=$spec"
  sbatch --array="$spec" \
    --export=ALL,FASTA_DIR="$FASTA_DIR",A3M_DIR="$A3M_DIR",HHM_DIR="$HHM_DIR",FILELIST="$FILELIST",HHDB="$HHDB",HHSUITE_SIF="$HHSUITE_SIF",SING_BINDS="${SING_BINDS:-}" \
    scripts/slurm_hhblits_hhmake_array.sh
  i=$((end + 1))
done
