# Reproduction

## Build Embedding Cache

You can build the cache from a base model:

```bash
python scripts/build_cache.py \
  --dataset DeepFoldProtein/malidup-dataset,all,test \
  --model Ankh-Large \
  --output_root .cache \
  --device cuda --batch_size 8 \
  --cache_type lmdb
```

Or from a fine-tuned checkpoint:

```bash
python scripts/build_cache.py \
  --dataset DeepFoldProtein/malidup-dataset,all,test \
  --model /path/to/your/checkpoint \
  --base_model_for_checkpoint Ankh-Large \
  --output_root .cache-finetuned \
  --device cuda --batch_size 8 \
  --cache_type lmdb
```

## OTalign

Run OTalign with a base model:

```bash
python scripts/run_otalign_on_dataset.py \
  --dataset DeepFoldProtein/malidup-dataset,all,test \
  --model Ankh-Large \
  --cache_dir /path/to/Ankh-Large/cache \
  --dp_mode global \
  --device cuda --align_batch_size 16 \
  --output out/global_Ankh-Large.jsonl
```

## Needleman-Wunsch

A dynamic programming-based global aligner using substitution matrices. Included as a classic baseline for raw sequence alignment.

See [Zhang Lab](https://zhanggroup.org/NW-align/).

From a Hugging Face dataset:

```bash
python scripts/run_nwalign_on_dataset.py \
  --dataset DeepFoldProtein/SABmark-dataset,twi,test \
  --nwalign_bin NWalign --glocal 0 \
  --output out/nwalign_sabmark-twi.jsonl
```

From a local JSONL file:

```bash
python scripts/run_nwalign_on_dataset.py \
  --dataset data/pairs.jsonl \
  --nwalign_bin NWalign \
  --output out/nwalign_pairs.jsonl
```

## HH-suite

A state-of-the-art profile-profile aligner that leverages MSAs. Serves as a strong upper baseline using evolutionary information. See the [HH-suite repo](https://github.com/soedinglab/hh-suite).

The overall workflow is:

1. Generate a sequence database from your dataset.
2. Run `hhblits` to generate Multiple Sequence Alignments (MSAs).
3. Run `hhmake` to create HMM profiles from MSAs.
4. Run `hhalign` using the HMM profiles.

### 1. Generate Sequence Database

Create an `ffindex` database of your sequences. This is used as input for `hhblits_mpi`.

```bash
python scripts/make_ffindex_from_hf.py \
  --dataset DeepFoldProtein/SABmark-dataset,twi,test \
  --out_prefix work/queries
```

This creates `work/queries.ffdata`, `work/queries.ffindex`, and `work/queries.names`.

### 2. Generate MSAs

This is a computationally intensive step. We provide a SLURM script to run `hhblits` in parallel with MPI.

```bash
# Set the path to your HH-suite database
export HHDB=/path/to/hhsuite/db/uniclust/UniRef30_2023_02

# Submit the SLURM job
sbatch scripts/slurm_hhblits_mpi_ffindex.sh
```

This script will generate MSA files in A3M format inside `work/a3m/`. Each file will be named `<sequence_id>.a3m`.

*Note: The `slurm_hhblits_mpi_ffindex.sh` script assumes your `hhblits_mpi` version supports the `-oa3m <dir>` option to write individual A3M files. If not, `hhblits_mpi` might produce an `ffindex` database of MSAs. In that case, you would need to unpack it using `ffindex_unpack`.*

### 3. Create HMM profiles

Convert the A3M files into HMM profiles (`.hhm`).

```bash
mkdir -p work/hhm
for f in $(cat work/queries.names); do
  hhmake -i work/a3m/$f.a3m -o work/hhm/$f.hhm
done
```

*Note: The original example included `-seq 2`, which severely truncates the MSA. We've removed it for a more general-purpose profile generation.*

### 4. Run HHalign

Now you can run `hhalign` on your dataset using the generated HMMs.

From a Hugging Face dataset:

```bash
# Global HMM-HMM alignment
python scripts/run_hhalign_on_dataset.py \
  --dataset DeepFoldProtein/SABmark-dataset,twi,test \
  --hhm_dir work/hhm \
  --mode global \
  --output work/hhalign_global.jsonl
```

From a local JSONL file:

```bash
# Local HMM-HMM alignment
python scripts/run_hhalign_on_dataset.py \
  --dataset data/pairs.jsonl \
  --hhm_dir work/hhm \
  --mode local \
  --output work/hhalign_local.jsonl
```

With custom flags:

```bash
# Glocal alignment with extra arguments
python scripts/run_hhalign_on_dataset.py \
  --dataset data/pairs.jsonl \
  --hhm_dir work/hhm \
  --mode glocal \
  --extra_args -Z 1 -B 1 \
  --output work/hhalign_glocal.jsonl
```
