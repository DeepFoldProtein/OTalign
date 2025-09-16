# Benchmark Manual

## Build Embedding Cache

```bash
python scripts/build_cache.py \
  --dataset DeepFoldProtein/SABmark-dataset \
  --name twi --split test \
  --model AnkhCL \
  --output_root .cache
```

## NWalign

See [Zhang Lab](https://zhanggroup.org/NW-align/).

```bash
# HF SABmark (twilight) -> NWalign predictions
python scripts/run_nwalign_on_dataset.py \
  --hf_dataset DeepFoldProtein/SABmark --name twi --split test \
  --nwalign_bin NWalign --glocal 0 \
  --output out/nwalign_sabmark_twi.jsonl

# JSONL pairs you curated yourself
python scripts/run_nwalign_on_dataset.py \
  --jsonl data/SABmark/twi.jsonl \
  --nwalign_bin /path/to/NWalign \
  --output out/nwalign_twi.jsonl
```

## HH-suite

See [repo](https://github.com/soedinglab/hh-suite).

### Array

```bash
export FASTA_DIR=work/fasta
export FILELIST=work/fasta.list
export A3M_DIR=work/a3m
export HHM_DIR=work/hhm
export HHDB=/path/to/hhsuite/db/uniclust30_2018_08/uniclust30_2018_08

python scripts/make_filelist_from_hf.py \
  --dataset DeepFoldProtein/SABmark-dataset \
  --name twi \
  --split test \
  --out_dir $FASTA_DIR \
  --filelist $FILELIST

sbatch scripts/slurm_hhblits_hhmake_array.sh
```

### MPI

```bash
python scripts/make_ffindex_from_hf.py \
  --dataset DeepFoldProtein/malidup-dataset \
  --name all --split test \
  --out_prefix work/queries

sbatch scripts/slurm_hhblits_mpi_ffindex.sh
```

### Final

```bash
# Local HMM-HMM (default)
python scripts/run_hhalign_on_dataset.py \
  --hf_dataset DeepFoldProtein/SABmark --name twi --split test \
  --hhm_dir work/hhm \
  --mode local \
  --output out/hhalign_local.jsonl

# Global-ish attempt
python scripts/run_hhalign_on_dataset.py \
  --jsonl data/SABmark/twi.jsonl \
  --hhm_dir work/hhm \
  --mode global \
  --output out/hhalign_global.jsonl

# Glocal attempt + custom flags (example)
python scripts/run_hhalign_on_dataset.py \
  --jsonl data/SABmark/twi.jsonl \
  --hhm_dir work/hhm \
  --mode glocal \
  --extra_args "-Z" "1" "-B" "1" \
  --output out/hhalign_glocal.jsonl
```
