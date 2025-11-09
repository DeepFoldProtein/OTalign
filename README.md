# **OTalign: Optimal Transport Alignment for Remote Protein Homologs Using Protein Language Model Embeddings**

<div align="center"> <img src="assets/logo_with_text.png" alt="OTalign Logo" width="400"></div>

**OTalign** is a new method that applies Optimal Transport (OT) theory to sequence alignment, providing a mathematically principled
framework for modeling residue matches and gaps. It integrates protein language model embeddings to capture evolutionary and structural context, enabling accurate alignment even for remote homologs in the low-identity (“twilight”) zone.

## **Key Features**

* **Optimal Transport Core**: Employs **Unbalanced Optimal Transport (UOT)** with the Sinkhorn algorithm to find an optimal residue-level correspondence (a "transport plan").  
* **Position-Specific Gap Penalties**: Introduces adaptive gap penalties derived from OT dual potentials, a principled alternative to fixed-cost models.  
* **Fine-Tuning Framework**: Includes a complete, differentiable framework for fine-tuning PLMs on alignment tasks using a custom KL-Divergence-based loss and Parameter-Efficient Fine-Tuning (PEFT) with LoRA.  
* **High Performance**: Achieves state-of-the-art results on challenging remote homolog benchmarks (SABmark, MALIDUP, MALISAM).  
* **PLM-Powered**: Utilizes rich, contextual embeddings from state-of-the-art PLMs like ESM, Ankh, and ProtT5.  
* **PLM Probing Tool**: Provides a quantitative framework to evaluate how well PLM embeddings capture structural and functional relationships.

## **How OTalign Works**

OTalign performs alignment in a three-stage process:

1. **Optimal Transport**: The two sets of residue embeddings are treated as empirical distributions. OTalign then computes an optimal **entropy-regularized unbalanced optimal transport (UOT)** plan that minimizes the "cost" (based on cosine distance) of transforming one distribution into the other. This plan represents a soft, many-to-many mapping between residues.  
2. **Embedding**: Protein sequences are fed into a PLM to generate high-dimensional embeddings for each residue. These embeddings capture structural and functional context.  
3. **Dynamic Programming**: The soft transport plan is used to derive **position-specific match scores** (from Pointwise Mutual Information) and **position-specific gap penalties** (from UOT dual potentials). These parameters guide a standard Dynamic Programming algorithm to produce the final, discrete gapped alignment.

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

## **Usage: Running Alignments**

You can run OTalign on a dataset using the run_otalign_on_dataset.py script. The dataset can be a local JSONL file or a Hugging Face dataset identifier.

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

* `--dataset`: Specifies the dataset. For Hugging Face datasets, the format is `user/dataset,config,split`.  
* `--model`: The name of the base PLM (e.g., AnkhCL, ESM2_33_650M) or the path to a fine-tuned checkpoint.  
* `--cache_dir`: Path to a directory for caching embeddings to accelerate subsequent runs.  
* `--dp_mode`: The dynamic programming mode (global, local, or glocal).

## **Documentation**

For more detailed information, please refer to the following documents:

* [**Reproduction Guide**](docs/reproduction.md): Instructions on how to reproduce our benchmark results and run baseline models.  
* [**Training Details**](docs/training_details.md): An in-depth explanation of the training process, loss functions, and model configuration.  
* [**Dataset Generation**](docs/dataset_generation.md): A guide on how the CATH-based training dataset was constructed.

## **Training: Fine-Tuning a Model**

OTalign allows you to fine-tune PLMs to improve their alignment capabilities. The training process uses LoRA (Low-Rank Adaptation) for efficiency.

The core of the training is a composite loss function, which adapts based on whether a pair is homologous (positive) or non-homologous (negative):

* **Alignment Loss** (for positives): A **Generalized Kullback-Leibler (KL) Divergence** pushes the model's predicted transport plan ($\Gamma$) to match the ground-truth plan ($T$) derived from a structural alignment.  
* **Sparsity Loss** (for positives): An L1-norm regularization that encourages a sharp, sparse alignment path.  
* **Emptiness Loss** (for negatives): An L1-norm regularization that forces the total mass of the transport plan towards zero, teaching the model not to align unrelated proteins.

To start training:

1. Configure your training run in a YAML file. See ```configs/finetune_config.yaml``` for an example.  
2. Launch the training using accelerate:  

   ```bash
   accelerate launch scripts/finetune.py configs/finetune_config.yaml
   ```

   The script supports multi-GPU training with DDP. A sample SLURM script is provided at `scripts/slurm_ddp_train.sh`.

## **Benchmark Results**

OTalign demonstrates superior performance compared to traditional and other deep learning-based methods, especially on remote homolog benchmarks. Results below are F1-Scores.

| Method | SABmark (sup) F1 ⬆️ | SABmark (twi) F1 ⬆️ | MALIDUP F1 ⬆️ | MALISAM F1 ⬇️ |
| :---- | :---- | :---- | :---- | :---- |
| Needleman-Wunsch | 0.334 | 0.118 | 0.349 | 0.066 |
| HHalign | 0.454 | 0.196 | 0.491 | 0.011 |
| DeepBLAST (ProtT5-XL) | 0.518 | 0.283 | 0.522 | 0.151 |
| PLMAlign (ProtT5-XL) | 0.469 | 0.253 | 0.507 | 0.168 |
| OTalign (ProtT5-XL) | 0.565 | 0.330 | 0.590 | **0.201** |
| OTalign (ESM-1b) | 0.417 | 0.189 | 0.415 | 0.063 |
| OTalign (PSM-2 650M) | 0.540 | 0.113 | 0.519 | 0.107 |
| **OTalign (Ankh-Large)** | **0.594** | **0.358** | **0.640** | 0.191 |

<!-- These findings confirm that the OTalign alignment framework achieves robust and consistent performance across diverse PLM embeddings and sequence characteristics. -->

## **Interactive Leaderboard**

We provide a comprehensive online leaderboard that enables systematic evaluation and comparison of alignment methods across multiple benchmark datasets. The platform serves as both a performance evaluation tool and a probe for assessing the structural fidelity of protein language model representations.

**🔗 [Access the OTalign Leaderboard](https://otalign.deepfold.org)**

* **Comprehensive Benchmarking**: Compare OTalign variants against traditional methods (Needleman-Wunsch, HHalign) and recent PLM-based approaches (PLMAlign, DeepBLAST) across challenging remote homolog datasets.  
* **PLM Representation Analysis**: Evaluate how different protein language models (ESM, ProtT5, Ankh families) perform under the OTalign framework.  
* **Scaling Behavior Visualization**: Interactive plots showing the relationship between model parameters and alignment performance.  
* **Dataset Documentation**: Detailed descriptions of benchmark datasets (SABmark, MALIDUP, MALISAM).  
* **Community Contributions**: Submit your own methods for standardized benchmarking and reproducible evaluation.

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
