# OTalign: Protein Alignment with Optimal Transport

<div align="center">
  <img src="assets/logo_with_text.png" alt="OTalign Logo" width="400">
</div>

[![Paper](https://img.shields.io/badge/paper-coming_soon-B31B1B.svg)](https://www.biorxiv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**OTalign** is a novel protein sequence alignment method that leverages the power of protein language models (PLMs) and the mathematical rigor of optimal transport (OT). It reframes the alignment task as a distribution matching problem, enabling robust and accurate alignments even for remote homologs.

## Key Features

- **PLM-Powered**: Utilizes rich, contextual embeddings from state-of-the-art PLMs like ESM, Ankh, and ProtT5.
- **Optimal Transport Core**: Employs Unbalanced Optimal Transport (UOT) with the Sinkhorn algorithm to find an optimal residue-level correspondence (a "transport plan").
- **Fine-Tuning Framework**: Includes a complete framework for fine-tuning PLMs on alignment tasks using a custom KL-Divergence-based loss and Parameter-Efficient Fine-Tuning (PEFT) with LoRA.
- **High Performance**: Achieves state-of-the-art results on challenging benchmarks, including SABmark (Superfamily, Twilight), MALIDUP, and MALISAM.

## How OTalign Works

OTalign performs alignment in a three-stage process:

1. **Embedding**: Protein sequences are fed into a PLM to generate high-dimensional embeddings for each residue. These embeddings capture structural and functional context.
2. **Optimal Transport**: The two sets of residue embeddings are treated as empirical distributions. OTalign then computes an optimal transport plan that minimizes the "cost" (based on cosine distance) of transforming one distribution into the other. This plan represents a soft, many-to-many mapping between residues.
3. **Dynamic Programming**: The soft transport plan is converted into a discrete, one-to-one gapped alignment using a dynamic programming algorithm, yielding the final alignment.

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/DeepFoldProtein/OTalign.git
   cd OTalign
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --editable .
   ```

## Usage: Running Alignments

You can run OTalign on a dataset using the `run_otalign_on_dataset.py` script. The dataset can be a local JSONL file or a Hugging Face dataset identifier.

```bash
python scripts/run_otalign_on_dataset.py \
  --dataset DeepFoldProtein/malidup-dataset,all,test \
  --model AnkhCL \
  --cache_dir /path/to/embedding_cache \
  --dp_mode global \
  --device cuda \
  --align_batch_size 16 \
  --output out/malidup_predictions.jsonl
```

- `--dataset`: Specifies the dataset. For Hugging Face datasets, the format is `user/dataset,config,split`.
- `--model`: The name of the base PLM (e.g., `AnkhCL`, `ESM2_33_650M`) or the path to a fine-tuned checkpoint.
- `--cache_dir`: Path to a directory for caching embeddings to accelerate subsequent runs.
- `--dp_mode`: The dynamic programming mode (`global`, `local`, or `glocal`).

## Documentation

For more detailed information, please refer to the following documents:

- **[Benchmarking Guide](./docs/benchmarking.md)**: Instructions on how to reproduce our benchmark results and run baseline models.
- **[Training Details](./docs/training_details.md)**: An in-depth explanation of the training process, loss functions, and model configuration.
- **[Dataset Generation](./docs/dataset_generation.md)**: A guide on how the CATH-based training dataset was constructed.

## Training: Fine-Tuning a Model

OTalign allows you to fine-tune PLMs to improve their alignment capabilities. The training process uses LoRA (Low-Rank Adaptation) for efficiency.

The core of the training is a composite loss function:

- **For homologous pairs (positives)**: A **KL-Divergence loss** pushes the model's predicted transport plan to match the ground-truth alignment.
- **For non-homologous pairs (negatives)**: An **emptiness loss** encourages the model to produce a near-zero transport plan, teaching it not to align unrelated proteins.
- **Masked Language Modeling (MLM)**: An auxiliary loss that helps the model retain its general protein knowledge during fine-tuning.

To start training:

1. Configure your training run in a YAML file. See `configs/train_config.yaml` for an example.
2. Launch the training using `accelerate`:

   ```bash
   accelerate launch scripts/train.py configs/train_config.yaml
   ```

   The script supports multi-GPU training with DDP. A sample SLURM script is provided at `scripts/slurm_ddp_train.sh`.

## Benchmark Results

OTalign demonstrates superior performance compared to traditional and other deep learning-based methods.

### F1 Score (Homology/Analogy)

| Method               | MALIDUP ↑  | MALISAM ↑  |
| -------------------- | ---------- | ---------- |
| Needleman-Wunsch     | 0.3492     | 0.0662     |
| HHalign              | 0.3825     | 0.0092     |
| **OTalign (AnkhCL)** | **0.6396** | 0.1911     |
| **OTalign (ProtT5)** | 0.5904     | **0.2011** |

### Accuracy (Recall on Remote Homologs)

| Method               | SABmark (sup) ↑ | SABmark (twi) ↑ |
| -------------------- | --------------- | --------------- |
| Needleman-Wunsch     | 0.3861          | 0.1496          |
| HHalign              | 0.3507          | 0.1596          |
| **OTalign (AnkhCL)** | **0.7139**      | **0.4660**      |
| **OTalign (ProtT5)** | 0.6783          | 0.4313          |

## Citation

If you use OTalign in your research, please cite our paper (link will be available soon).
