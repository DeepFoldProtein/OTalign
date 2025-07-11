# Optimal Transport Alignment

## Sequence Alignment

* What is sequence alignment?
  * Protein sequence alignment arranges amino acid chains side by side, adding gaps so homologous or similar residues match.

* Why perform sequence alignment?
  * **Reveal evolutionary relationships :** Similar alignments suggest shared ancestry and help build phylogenetic trees.
  * **Predict structure and function :** If an unknown protein aligns well with one whose 3D structure or function is known, you can predict its properties by homology.
  * **Identify conserved regions :** Highly preserved regions often indicate active sites, binding pockets, or other functionally critical motifs.

## Evaluation Metrics

### Pairwise Metrics

```raw
# Prediction (Sample)

MGSIGIIIQVVTEE--LNP (17)
::::::  ::::
MGSIGIV-QVVTFATEGVE (18)
1234567890123456789

# Ground Truth

MGSIGIIIQVVT--EELNP (17)
::::::  ::::   :
MGSIGIV-QVVTFATEGVE (18)
1234567890123456789
```

| Name              | Definition                                                                          |  Worked Example |
| ----------------- | ----------------------------------------------------------------------------------- | --------------: |
| Accuracy          |                                                                                     |                 |
| Alignment length  | Total number of columns in the final alignment                                      |              19 |
| Aligned length    | Number of columns where both sequences have a residue.                              |              16 |
| Identical length  | Among those aligned columns, the count where the two residues are exactly the same. |              10 |
| Sequence identity | Identical length / the length of the longer sequence                                | 0.555 (= 10/18) |

### Statistical Metrics

| Name    | Definition |
| ------- | ---------- |
| e-value |            |
| p-value |            |

## Research Workflow and Methodological Approach

### Training & Evaluation Strategy

* Database
  * SABMark
  * PDB40
  * CASP
  * CATH
* Strategy
  * Train the model on SABMark and evaluate on PDB40.

### Input Data

* Protein Language Model (PLM)
  * ESM-1b
  * ESM-2
* Structure Tokenizer
* Multimodal

### Model Architecture

* Neural Network
  * Baseline
  * Add a trainable neural network instead of using the embedding outputs directly.
* Optimal Transport vs Unbalance Optimal Transport

### Leveraging Outputs from OT

* Direct
* Score Matrix (Dynamic Programming)
  * Local
    * Smith-Waterman
    * Differentiable Smith-Waterman
  * Global
    * Needleman-Wunsch
  * Glocal

### Comparison to Alternative Tools

* NWalign with BLOSUM62
* HHBlits
* HHAlign
* PLMAlign

## References
