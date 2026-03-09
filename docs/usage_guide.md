# OTalign Usage Guide

This guide covers installation, running alignments, configuring parameters, and interpreting results.

## Installation

```bash
git clone https://github.com/DeepFoldProtein/OTalign.git
cd OTalign
python -m venv .venv
source .venv/bin/activate
pip install --editable .
```

**Optional dependencies:**

- `numba`: Accelerates the dynamic programming alignment step (~5-10x speedup).
- `matplotlib`: Required for transport plan visualization.

## Running Alignments

### Single-Pair Alignment

See [`example_alignment.ipynb`](../example_alignment.ipynb) for a step-by-step walkthrough. The key steps are:

1. **Load a PLM** via `get_plm_adaptor_and_configs()` (optionally with a fine-tuned LoRA checkpoint).
2. **Encode sequences** to get residue embeddings.
3. **Compute the cost matrix** using `pairwise_cosine()`.
4. **Run UOT Sinkhorn** via `unbalanced_sinkhorn()` to get the transport plan and dual potentials.
5. **Extract the hard alignment** with `hard_alignment_from_transport()`.

```python
from otalign.align.cost import pairwise_cosine
from otalign.align.uot_alignment import hard_alignment_from_transport
from otalign.functional.sinkhorn_uot import unbalanced_sinkhorn
from otalign.models.plm_adaptors import get_plm_adaptor_and_configs

# Load model
plm_adaptor, _, _ = get_plm_adaptor_and_configs("AnkhCL", for_masked_lm=True)
model = plm_adaptor.model.to("cuda").eval()

# Encode two sequences
emb_out = plm_adaptor.encode(["AGLPV...", "DGLVH..."], device="cuda", disable_grad=True)

# Compute cost and transport plan
cost = pairwise_cosine(emb1, emb2)
plan, u, v = unbalanced_sinkhorn(cost, a, b, num_iter=1000, reg=0.1, reg_m1=1.0, reg_m2=1.0)

# Get hard alignment
import numpy as np
plan_np = plan[0].cpu().numpy()
f_np = np.log(u[0].cpu().numpy())
g_np = np.log(v[0].cpu().numpy())
result = hard_alignment_from_transport(plan_np, f=f_np, g=g_np, mode="glocal")
print(result["cigar"])  # e.g., "5M2I10M1D3M"
```

### Batch Alignment on a Dataset

Use `scripts/run_otalign_on_dataset.py` for batch evaluation:

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

**Using a fine-tuned checkpoint:**

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

**Dataset format:** The `--dataset` argument accepts either:

- A **Hugging Face dataset** in the format `user/dataset,config,split`
- A **local JSONL file** where each line contains `seq1`, `seq2`, `seq1_id`, `seq2_id`, and optionally `ref_alignment`.

### Saving Transport Plans

To save transport plans for downstream analysis (e.g., ablation studies):

```bash
python scripts/run_otalign_on_dataset.py \
  --dataset DeepFoldProtein/SABmark-dataset,sup,test \
  --model AnkhCL \
  --dp_mode glocal \
  --device cuda \
  --save_transport_plan_dir out/transport_plans \
  --output out/predictions.jsonl
```

### FASTA Alignment from Sequences

For aligning sequences directly from FASTA files:

```bash
python scripts/align_fasta_with_otalign.py \
  --fasta input.fasta \
  --model AnkhCL \
  --device cuda
```

## UOT Parameters

These parameters control the Unbalanced Optimal Transport step (Sinkhorn algorithm):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `reg` | 0.1 | **Entropy regularization** (epsilon). Lower values produce sharper transport plans but may slow convergence. Higher values spread mass more evenly. |
| `reg_m` / `lambda1`, `lambda2` | 1.0 | **Marginal relaxation** (KL penalty). Controls how strictly the row/column marginals must match the uniform distribution. Lower values allow more mass to be "destroyed" (useful for sequences of different lengths or with unaligned regions). |
| `num_iter` | 1000 | Number of Sinkhorn iterations. Usually 500-1000 is sufficient for convergence. |

## Dynamic Programming Parameters

These parameters control how the transport plan is converted into a discrete alignment:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dp_mode` | `global` | Alignment mode: `global` (end-to-end), `glocal` (free terminal gaps on both ends), `q2t` (free template terminals), `t2q` (free query terminals), `local` (free on all ends, no negative scores). |
| `go_base` | 8.0 | **Base gap open penalty**. Higher values discourage opening new gaps. |
| `ge_base` | 1.0 | **Base gap extend penalty**. Higher values discourage long gaps. |
| `gamma` | 1.0 | **Mass sensitivity exponent**. Controls how strongly the transport plan marginal mass modulates gap penalties. `gamma=0` disables mass-based modulation. |
| `k_f`, `k_g` | 0.75 | **Dual potential sensitivity**. Controls how the UOT dual potentials (`f`, `g`) influence gap penalties via a sigmoid function. `k=0` disables dual potential modulation. |
| `score_scale` | 1.0 | **Match score multiplier**. Scales the PMI-based match scores from the transport plan. |
| `eta` | 0.25 | Minimum ratio for gap extend penalties relative to `ge_base`. |
| `band` | None | Optional diagonal band width constraint for the DP matrix. |

### Choosing `dp_mode`

- **`global`**: Use when both sequences should be fully aligned end-to-end.
- **`glocal`**: Recommended default for benchmarks. Allows partial overlaps at both terminals.
- **`local`**: Use when you expect only a sub-region to align (e.g., domain-level matching).

## PLM Backends

OTalign supports multiple protein language models. Specify the model name via the `--model` argument:

| Model Name | Parameters | Notes |
|------------|-----------|-------|
| `ESM1b_33_650M` | 650M | Meta AI ESM-1b |
| `ESM2_33_650M` | 650M | Meta AI ESM-2 |
| `Ankh-Large` | 780M | Ankh encoder model |
| `AnkhCL` | ~780M | Ankh contrastive learning variant |
| `ProtT5-XL` | 3B | ProtTrans T5-XL encoder |

You can also pass a **fine-tuned checkpoint directory** as `--model` together with `--base_model_for_checkpoint` to specify the base architecture.

## Understanding Output Formats

### JSONL Output

Each line in the output JSONL file contains:

```json
{
  "pair_id": "1a00A-1b00B",
  "seq1_id": "1a00A",
  "seq2_id": "1b00B",
  "pred_alignment": [[0, 0], [1, 1], [2, 3], ...],
  "metrics": {
    "precision": 0.85,
    "recall": 0.80,
    "f1": 0.82,
    "num_predicted": 45,
    "num_reference": 50
  },
  "ot_metrics": {
    "transport_cost": 1.23,
    "mean_cosine": 0.78,
    "mass_total": 0.95,
    "sinkhorn_divergence": 0.15
  }
}
```

- **`pred_alignment`**: List of `[query_idx, template_idx]` matched residue pairs (0-based).
- **`metrics`**: Standard alignment quality metrics against the reference alignment.
- **`ot_metrics`**: Transport plan quality metrics (cosine similarity, mass coverage, etc.).

### CIGAR String

The alignment is also represented as a CIGAR string (e.g., `5M2I10M1D3M`):

- `M`: Match (aligned pair)
- `I`: Insertion (gap in template)
- `D`: Deletion (gap in query)

### Transport Plans

Saved transport plans (`.npz` files) contain:

- `data`, `scale`, `zero_point`: Quantized transport plan matrix (dequantize with `otalign.quantize.dequantize()`).
- `f`, `g`: Log-space dual potentials (UOT scaling vectors).

## Interpreting Transport Plans

The transport plan is a matrix `P[i, j]` representing the amount of "mass" transported from query residue `i` to template residue `j`.

- **Diagonal patterns** indicate well-aligned regions.
- **Off-diagonal mass** suggests insertions, deletions, or structural rearrangements.
- **Low marginal mass** (row/column sums near zero) indicates residues that are poorly matched — these receive higher gap penalties in the DP step.
- **Dual potentials** (`f`, `g`) encode residue-level alignment confidence from the Sinkhorn algorithm. Negative values indicate residues likely to be gapped.

Use `otalign.viz.plot_plan_with_domains()` to visualize transport plans with marginal mass bars:

```python
from otalign.viz import plot_plan_with_domains
fig, ax = plot_plan_with_domains(plan, f=f, g=g, cmap="terrain_r")
```

## Ablation Study

To run the ablation study on gap penalty components:

```bash
python scripts/run_ablation_study.py \
  --dataset DeepFoldProtein/SABmark-dataset,sup,test \
  --transport_plan_dir out/transport_plans \
  --output out/ablation_results.csv
```

See `scripts/run_ablation_study.py --help` for all options.

## Embedding Cache

Building an embedding cache avoids redundant PLM forward passes:

```bash
python scripts/build_cache.py \
  --dataset DeepFoldProtein/malidup-dataset,all,test \
  --model Ankh-Large \
  --output_root .cache \
  --device cuda --batch_size 8 \
  --cache_type lmdb
```

Then pass `--cache_dir .cache` to `run_otalign_on_dataset.py`.

## Training (Fine-Tuning)

To fine-tune a PLM with LoRA:

1. Configure training in a YAML file (see `configs/train_config.yaml`).
2. Launch with accelerate:

```bash
accelerate launch scripts/finetune.py configs/finetune_config.yaml
```

See [`docs/training_details.md`](training_details.md) for loss function details and [`docs/dataset_generation.md`](dataset_generation.md) for dataset construction.
