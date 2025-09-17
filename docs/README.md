# Benchmark

## Databases

### MALIDUP vs MALISAM

| Dataset     | Purpose                         | Examples                             | Which is better?    |
| ----------- | ------------------------------- | ------------------------------------ | ------------------- |
| **MALIDUP** | **True homologs**, low identity | domain duplication, same fold        | High recall         |
| **MALISAM** | **Non-homologous** (analogs)    | structural analogy, different origin | Low false alignment |

* [**MALIDUP**](http://prodata.swmed.edu/malidup/)
* [**MALISAM**](http://prodata.swmed.edu/malisam/)

### SABmark

**SABmark** is a benchmark dataset designed to evaluate sequence alignment methods on **remote homologs**. It contains protein pairs grouped by **SCOP superfamilies**, with structural alignments as ground truth. The dataset includes challenging **low sequence identity** cases and is commonly used to assess alignment **recall and accuracy**.

* [**SABmark**](https://doi.org/10.1093/bioinformatics/bth493)

## Results

### Accuarcy (Recall)

| Method | MALIDUP ↑ | MALISAM ↓ | SABmark (sup) ↑ | SABmark (twi) ↑ |
|---|---|---|---|---|
| Needleman-Wunsch | 0.3733 | 0.0749 | 0.3861 | 0.1496 |
| HHalign | 0.4523 | 0.0093 | | |
| OTalign (AnkhCL) | | | | |
| OTalign (ESM-1b) | | | | |
| OTalign (ESM-2) | | | | |
| OTalign (ProtT5) | | | | |

## Reproduction

### Build Embedding Cache

```bash
python scripts/build_cache.py \
  --dataset DeepFoldProtein/{SABmark-dataset,malisam-dataset,malidup-dataset} \
  --name {all,sup,twi} --split test \
  --model {AnkhCL,ESM1b,ESM2,ProtT5} \
  --output_root .cache
  --dtype fp32 --batch_size 4 --device cuda:0 --shard_size 100
```

### Needleman-Wunsch

See [Zhang Lab](https://zhanggroup.org/NW-align/).

```bash
# HF SABmark (twilight) -> NWalign predictions
python scripts/run_nwalign_on_dataset.py \
  --hf_dataset DeepFoldProtein/SABmark-dataset --name twi --split test \
  --nwalign_bin NWalign --glocal 0 \
  --output out/nwalign_sabmark-twi.jsonl

# JSONL pairs you curated yourself
python scripts/run_nwalign_on_dataset.py \
  --jsonl data/pairs.jsonl \
  --nwalign_bin NWalign \
  --output out/nwalign_pairs.jsonl
```

### HH-suite

See [repo](https://github.com/soedinglab/hh-suite).

```bash
export HHDB=/path/to/hhsuite/db/uniclust/UniRef30_2023_02

python scripts/make_ffindex_from_hf.py \
  --dataset <DATASET> \
  --name all --split test \
  --out_prefix work/queries

sbatch scripts/slurm_hhblits_mpi_ffindex.sh
```

```bash
ffindex_unpack work/a3m.ffdata work/a3m.ffindex work/a3m

find work/a3m -type f -name "*.fasta" -print0 | while IFS= read -r -d '' file; do 
  mv -- "$file" "$(echo "$file" | sed 's/\.fasta$/.a3m/')"
done

mkdir -p work/hhm
for f in $(cat work/queries.names); do
  hhmake -i work/a3m/$f.a3m -o work/hhm/$f.hhm -seq 2
done
```

```bash
# Global HMM-HMM
python scripts/run_hhalign_on_dataset.py \
  --hf_dataset <DATASET> --name all --split test \
  --hhm_dir work/hhm \
  --mode global \
  --output out/hhalign_local.jsonl

# Local HMM-HMM
python scripts/run_hhalign_on_dataset.py \
  --jsonl data/pairs.jsonl \
  --hhm_dir work/hhm \
  --mode local \
  --output out/hhalign_global.jsonl

# Glocal attempt + custom flags (example)
python scripts/run_hhalign_on_dataset.py \
  --jsonl data/pairs.jsonl \
  --hhm_dir work/hhm \
  --mode glocal \
  --extra_args "-Z" "1" "-B" "1" \
  --output out/hhalign_glocal.jsonl
```
