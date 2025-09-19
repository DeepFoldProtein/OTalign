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

### F1

| Method            | MALIDUP ↑   | MALISAM ↑   |
| ----------------- | ----------- | ----------- |
| Needleman-Wunsch  | 0.3492      | 0.0662      |
| HHalign           | 0.3825      | 0.0092      |
| OTalign (AnkhCL)  | **0.6396**  | 0.1911      |
| OTalign (ProtT5)  | 0.5904      | **0.2011**  |
| OTalign (ESM-2)   | 0.5391      | 0.1133      |
| OTalign (ESM-1b)  | 0.4149      | 0.0633      |

### Accuarcy (Recall)

| Method            | SABmark (sup) ↑ | SABmark (twi) ↑ |
| ----------------- | --------------- | --------------- |
| Needleman-Wunsch  | 0.3861          | 0.1496          |
| HHalign           | 0.3507          | 0.1596          |
| OTalign (AnkhCL)  | **0.7139**      | **0.4660**      |
| OTalign (ProtT5)  | 0.6783          | 0.4313          |
| OTalign (ESM-2)   | 0.6499          | 0.3824          |
| OTalign (ESM-1b)  | 0.3298          | 0.2492          |

## Reproduction

### Build Embedding Cache

```bash
python scripts/build_cache.py \
  --dataset DeepFoldProtein/malidup-dataset \
  --name all --split test \
  --model AnkhCL \
  --output_root .cache \
  --device cuda:2 --batch_size 8 \
  --cache_type lmdb
```

### OTalign

```bash
python scripts/run_otalign_on_dataset.py \
  --dp_mode global \
  --device cuda --align_batch_size 16 \
  --output out/global.jsonl \
  --hf_dataset DeepFoldProtein/SABmark-dataset --name sup --split test \
  --model AnkhCL \
  --cache_dir CACHE_DIR
```

### Needleman-Wunsch

A dynamic programming-based global aligner using substitution matrices. Included as a classic baseline for raw sequence alignment.

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

A state-of-the-art profile-profile aligner that leverages MSAs. Serves as a strong upper baseline using evolutionary information.

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
find work/a3m -type f -exec mv -- {} {}.a3m \;

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
  --output work/hhalign_global.jsonl

# Local HMM-HMM
python scripts/run_hhalign_on_dataset.py \
  --jsonl data/pairs.jsonl \
  --hhm_dir work/hhm \
  --mode local \
  --output work/hhalign_local.jsonl

# Glocal attempt + custom flags (example)
python scripts/run_hhalign_on_dataset.py \
  --jsonl data/pairs.jsonl \
  --hhm_dir work/hhm \
  --mode glocal \
  --extra_args "-Z" "1" "-B" "1" \
  --output work/hhalign_glocal.jsonl
```
