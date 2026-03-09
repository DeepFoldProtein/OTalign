# **OTalign: Optimal Transport Based Protein Sequence Alignment**

<div align="center"> <img src="assets/logo_with_text.png" alt="OTalign Logo" width="400"></div>

**OTalign** is a new method that applies Optimal Transport (OT) theory to sequence alignment, providing a mathematically principled
framework for modeling residue matches and gaps. It integrates protein language model embeddings, enabling accurate alignment even for remote homologs in the low-identity ("twilight") zone.

## **Key Features**

* **Optimal Transport Core**: Employs **Unbalanced Optimal Transport (UOT)** with the Sinkhorn algorithm to find an optimal residue-level correspondence (a "transport plan").
* **Position-Specific Gap Penalties**: Introduces adaptive gap penalties derived from OT dual potentials, a principled alternative to fixed-cost models.
* **Fine-Tuning Framework**: Includes a complete, differentiable framework for fine-tuning PLMs on alignment tasks using a custom KL-Divergence-based loss and Parameter-Efficient Fine-Tuning (PEFT) with LoRA.
* **High Performance**: Achieves state-of-the-art results on challenging remote homolog benchmarks (SABmark, MALIDUP, MALISAM).
* **PLM-Powered**: Utilizes rich, contextual embeddings from state-of-the-art PLMs like ESM, Ankh, and ProtT5.
* **PLM Probing Tool**: Provides a quantitative framework to evaluate how well PLM embeddings capture structural and functional relationships.

## **How OTalign Works**

OTalign performs alignment in a three-stage process:

1. **Embedding**: Protein sequences are fed into a PLM to generate high-dimensional embeddings for each residue. These embeddings capture structural and functional context.
2. **Optimal Transport**: The two sets of residue embeddings are treated as empirical distributions. OTalign computes an optimal **entropy-regularized unbalanced optimal transport (UOT)** plan that minimizes the "cost" (based on cosine distance) of transforming one distribution into the other. This plan represents a soft, many-to-many mapping between residues.
3. **Dynamic Programming**: The soft transport plan is used to derive **position-specific match scores** (from Pointwise Mutual Information) and **position-specific gap penalties** (from UOT dual potentials). These parameters guide a standard Dynamic Programming algorithm to produce the final, discrete gapped alignment.

## **Prerequisites**

Before you begin, make sure you have the following installed:

* **Python 3.11+** (OTalign requires Python >= 3.11)
* **pip** (Python package manager, included with Python)
* **Git** (for cloning the repository)
* **CUDA** (optional but recommended for GPU acceleration; OTalign runs on CPU but is significantly faster on GPU)

You can verify your Python version by running:

```bash
python --version   # Should print Python 3.11.x or higher
```

## **Setup**

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

   On Windows, activate the virtual environment with `.venv\Scripts\activate` instead.

## **Quick Start**

The fastest way to get started is with the interactive Jupyter notebook. It walks through a complete alignment example step by step.

1. Make sure you have completed the [Setup](#setup) steps above.

2. Launch the example notebook:

   ```bash
   jupyter notebook example_alignment.ipynb
   ```

3. Follow the cells in the notebook to:
   - Load a protein language model
   - Encode two protein sequences into residue embeddings
   - Compute an optimal transport plan between the two sequences
   - Extract a discrete alignment from the transport plan
   - Visualize the results

If you prefer working from the command line, you can align two sequences directly from a FASTA file:

```bash
python scripts/align_fasta_with_otalign.py \
  --fasta input.fasta \
  --model AnkhCL \
  --device cuda
```

Replace `cuda` with `cpu` if you do not have a GPU.

## **Usage: Running Alignments**

### Aligning a Dataset

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

**Key arguments:**

* `--dataset`: Specifies the dataset. For Hugging Face datasets, the format is `user/dataset,config,split`. You can also pass a local JSONL file path.
* `--model`: The name of the base PLM (e.g., `AnkhCL`, `ESM2_33_650M`) or the path to a fine-tuned checkpoint directory.
* `--cache_dir`: Path to a directory for caching embeddings to accelerate subsequent runs.
* `--dp_mode`: The dynamic programming mode (`global`, `local`, or `glocal`).
* `--device`: `cuda` for GPU or `cpu` for CPU.

### Supported PLM Backends

OTalign supports multiple protein language models:

| Model Name | Parameters | Description |
|------------|-----------|-------------|
| `ESM1b_33_650M` | 650M | Meta AI ESM-1b |
| `ESM2_33_650M` | 650M | Meta AI ESM-2 |
| `Ankh-Large` | 780M | Ankh encoder model |
| `AnkhCL` | ~780M | Ankh contrastive learning variant |
| `ProtT5-XL` | 3B | ProtTrans T5-XL encoder |

You can also pass a **fine-tuned checkpoint directory** as `--model` together with `--base_model_for_checkpoint` to specify the base architecture:

```bash
python scripts/run_otalign_on_dataset.py \
  --dataset DeepFoldProtein/SABmark-dataset,sup,test \
  --model work/checkpoints/esm1b-lora-finetune-2/checkpoint-epoch-3 \
  --base_model_for_checkpoint ESM1b_33_650M \
  --cache_dir .cache \
  --dp_mode glocal \
  --device cuda \
  --output out/sabmark_sup_finetuned.jsonl
```

### Building an Embedding Cache

Building an embedding cache avoids redundant PLM forward passes across runs:

```bash
python scripts/build_cache.py \
  --dataset DeepFoldProtein/malidup-dataset,all,test \
  --model Ankh-Large \
  --output_root .cache \
  --device cuda --batch_size 8 \
  --cache_type lmdb
```

Then pass `--cache_dir .cache` to `run_otalign_on_dataset.py`.

## **Understanding the Output**

OTalign produces a JSONL file where each line contains the alignment result for one sequence pair:

```json
{
  "pair_id": "1a00A-1b00B",
  "seq1_id": "1a00A",
  "seq2_id": "1b00B",
  "pred_alignment": [[0, 0], [1, 1], [2, 3]],
  "metrics": { "precision": 0.85, "recall": 0.80, "f1": 0.82 }
}
```

* **`pred_alignment`**: List of `[query_idx, template_idx]` matched residue pairs (0-based indices).
* **`metrics`**: Alignment quality scores computed against a reference alignment (if provided).

The alignment is also represented as a **CIGAR string** (e.g., `5M2I10M1D3M`):
* `M` = Match (aligned pair), `I` = Insertion (gap in template), `D` = Deletion (gap in query)

## **Common Issues and Troubleshooting**

* **`ModuleNotFoundError: No module named 'otalign'`**: Make sure you installed with `pip install --editable .` from within the project directory, and that your virtual environment is activated.
* **CUDA out of memory**: Try reducing `--align_batch_size` (e.g., to 4 or 1), or use `--device cpu` if you do not have sufficient GPU memory.
* **Slow alignment without GPU**: Install `numba` (`pip install numba`) for a ~5-10x speedup on the dynamic programming step when using CPU.
* **Dependency resolution fails**: Ensure you are using Python >= 3.11. Check with `python --version`.

## **Project Structure**

```
OTalign/
├── otalign/                  # Core library
│   ├── align/                # Cost matrices, DP alignment, UOT alignment
│   ├── functional/           # Sinkhorn algorithm implementations
│   ├── models/               # PLM adaptors and embedding utilities
│   ├── cache/                # Embedding cache (LMDB, NPZ)
│   └── utils/                # Display, quantization, visualization
├── scripts/                  # CLI tools for alignment, training, benchmarking
├── configs/                  # Training configuration files
├── docs/                     # Detailed documentation
├── benchmark/                # Benchmark evaluation framework
└── example_alignment.ipynb   # Interactive tutorial notebook
```

## **EBA Benchmark (MALIDUP, MALISAM, SABmark-sup, SABmark-twi)**

To run the [EBA](https://github.com/DeepFoldProtein/EBA) (Embedding-based alignment) model **eba_prott5** on all four benchmarks and generate plots, use the EBA uv environment and the following commands (from project root):

```bash
# 1. Sync submodule and EBA env (once)
git submodule update --init third_party/eba
uv sync --directory .uv/eba

# 2. Run eba_prott5 on MALIDUP, MALISAM, SABmark-sup, SABmark-twi
.uv/eba/.venv/bin/python -m benchmark run --tests malidup malisam sabmark-sup sabmark-twi --models eba_prott5 --update

# 3. Generate plots for all four benchmarks
.uv/eba/.venv/bin/python -m benchmark plot --tests malidup malisam sabmark-sup sabmark-twi
```

Results are written to `out/results/<test>/eba_prott5/`; plots to `out/plots/`.

## **Training: Fine-Tuning a Model**

OTalign allows you to fine-tune PLMs to improve their alignment capabilities. The training process uses LoRA (Low-Rank Adaptation) for efficiency.

The core of the training is a composite loss function, which adapts based on whether a pair is homologous (positive) or non-homologous (negative):

* **Alignment Loss** (for positives): A **Generalized Kullback-Leibler (KL) Divergence** pushes the model's predicted transport plan ($\Gamma$) to match the ground-truth plan ($T$) derived from a structural alignment.
* **Sparsity Loss** (for positives): An L1-norm regularization that encourages a sharp, sparse alignment path.
* **Emptiness Loss** (for negatives): An L1-norm regularization that forces the total mass of the transport plan towards zero, teaching the model not to align unrelated proteins.

To start training:

1. Configure your training run in a YAML file. See `configs/finetune_config.yaml` for an example.
2. Launch the training using accelerate:

   ```bash
   accelerate launch scripts/finetune.py configs/finetune_config.yaml
   ```

   The script supports multi-GPU training with DDP. A sample SLURM script is provided at `scripts/slurm_ddp_train.sh`.

## **Benchmark Results**

OTalign demonstrates superior performance compared to traditional and other deep learning-based methods, especially on remote homolog benchmarks. Results below are F1-Scores.

> **Note:** Benchmark results will be updated.

| Method | SABmark (sup) F1 | SABmark (twi) F1 | MALIDUP F1 | MALISAM F1 |
| :---- | :---- | :---- | :---- | :---- |
| Needleman-Wunsch | | | | |
| HHalign | | | | |
| DeepBLAST (ProtT5-XL) | | | | |
| PLMAlign (ProtT5-XL) | | | | |
| OTalign (ProtT5-XL) | | | | |
| OTalign (ESM-1b) | | | | |
| OTalign (PSM-2 650M) | | | | |
| OTalign (Ankh-Large) | | | | |

## **Interactive Leaderboard**

We provide a comprehensive online leaderboard that enables systematic evaluation and comparison of alignment methods across multiple benchmark datasets. The platform serves as both a performance evaluation tool and a probe for assessing the structural fidelity of protein language model representations.

**[Access the OTalign Leaderboard](https://otalign.deepfold.org)**

* **Comprehensive Benchmarking**: Compare OTalign variants against traditional methods (Needleman-Wunsch, HHalign) and recent PLM-based approaches (PLMAlign, DeepBLAST) across challenging remote homolog datasets.
* **PLM Representation Analysis**: Evaluate how different protein language models (ESM, ProtT5, Ankh families) perform under the OTalign framework.
* **Scaling Behavior Visualization**: Interactive plots showing the relationship between model parameters and alignment performance.
* **Dataset Documentation**: Detailed descriptions of benchmark datasets (SABmark, MALIDUP, MALISAM).
* **Community Contributions**: Submit your own methods for standardized benchmarking and reproducible evaluation.

## **Documentation**

For more detailed information, please refer to the following documents:

* [**Usage Guide**](docs/usage_guide.md): Comprehensive guide covering parameter configuration, output formats, and interpreting results.
* [**Reproduction Guide**](docs/reproduction.md): Instructions on how to reproduce our benchmark results and run baseline models.
* [**Training Details**](docs/training_details.md): An in-depth explanation of the training process, loss functions, and model configuration.
* [**Dataset Generation**](docs/dataset_generation.md): A guide on how the CATH-based training dataset was constructed.
* [**ECOD Homolog Detection Benchmark**](docs/ecod_benchmark.md): Guide for the ECOD-based homolog detection benchmark with ROC/PR curve evaluation.

## **Citation**

If you use OTalign in your research, please cite our paper:

```bibtex
@article{minsoo2025,
      title={OTalign: Optimal Transport Alignment for Remote Protein Homologs Using Protein Language Model Embeddings},
      author={Minsoo Kim, Hanjin Bae, Gyeongpil Jo, Kunwoo Kim, Jejoong Yoo, and Keehyoung Joo},
      volume={},
      ISSN={},
      doi={},
      number={},
      journal={under review},
      publisher={},
      author={},
      year={},
      pages={}
```

<!-- OTalign: Optimal Transport Alignment for Remote Protein Homologs Using Protein Language Model Embeddings
Minsoo Kim, Hanjin Bae, Gyeongpil Jo, Kunwoo Kim, Jejoong Yoo, and Keehyoung Joo.
Bioinformatics (2025) -->
