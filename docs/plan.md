# Optimal Transport Alignment

## Sequence Alignment

* What is sequence alignment?
  * Protein sequence alignment arranges amino acid chains side by side, adding gaps so homologous or similar residues match.

* Why perform sequence alignment?
  * **Reveal evolutionary relationships :** Similar alignments suggest shared ancestry and help build phylogenetic trees.
  * **Predict structure and function :** If an unknown protein aligns well with one whose 3D structure or function is known, you can predict its properties by homology.
  * **Identify conserved regions :** Highly preserved regions often indicate active sites, binding pockets, or other functionally critical motifs.

## Dataset

* PDB40
  * Protein Data Bank에 저장된 단백질 구조의 서열을 sequence identity를 기준으로 유사도가 40% 이하로 군집화.
  * 군집화 후에 길아가 1,000 이하인 것들을 추출함.
  * One-to-All 계산을 통해 구조기반 정렬(DeepAlign)을 수행함.
  * Ground truth로 사용함.
  * 단, 구조가 유사하지만 filtering 과정에서 누락된 것이 존재할 수 있음.
* SABMark
  * SCOP을 기반으로 twilight zone(twi)과 superfamily(sup) 두 세트로 구성됨.

## Research Workflow and Methodological Approach

### Input Data

* Protein Language Model(PLM)을 이용하여 query와 target의 embedding output을 사용함.
  * ESM-1b (650M), ESM-2 (650M).
* Structure Tokenizer
  * Foldseek[1]처럼 structure token (3Di)을 사용
    * 기준 residue와 가장 근접한 resideu들의 $C_\alpha$ 사이의 각도, 거리, index 차이를 계산
    * VQ-VAE구조의 모델을 학습하여 20개의 structure token으로 tokenize
* Multimodal model
  * DPLM-2
    * 서열 정보와 구조 정보를 동시에 학습

### Optimal Transport

* Neural Network
  * Input 데이터를 바로 사용하지 않고 학습가능한 neural network를 통과시킨 후 optimal transprot를 수행
* Optimal Transport (OT) vs Unbalance Optimal Transport (UOT)
  * UOT는 OT 수행 전과 후 total mass가 보존되지 않는다.
  * Outlier가 있을 때 outlier를 고려하지 않고 정렬할 수 있다.
* Soft-to-Hard Alignment
  * 단백질 서열은 고정된 순서가 존재하기 때문에 정렬 후에도 순서가 유지되어야 한다.
  * Query의 $i$번째 서열이 target의 $j_{i}$번째 서열에 match 되었을 때 $j_{i} < j_{i+1}$

### Leveraging Outputs from OT

* OT Plan을 alignment matrix로 사용
  * query의 길이가 $N$, target의 길이가 $M$. $T \in \mathbb{R}^{N \times M}$
  * Ground Truth: $T_{ij} \in \{0, 1\} \ (\text{match} = 1, \text{otherwise} = 0)$
* Score Matrix로 사용하여 Dynamic Programming을 적용
  * Local alignment
    * Smith-Waterman
    * Differentiable Smith-Waterman
  * Global alignment
    * Needleman-Wunsch
  * Glocal alignment
    * Needleman-Wunsch + 가장자리에 전부 0을 넣기
    * 정렬된 부분만 보기, 나머지는 무시하기 때문에 Gap 페널티가 빠짐

## Evaluation

### Comparison to Alternative Tools

* 기존에 사용하던 alignment tool들과 비교하여 OTalign이 얼마나 효과가 있는지 비교
  * BLOSUM + NWalign, HHBlits, HHAlign, PLMAlign
* Dataset에 대해 alignment 수행 후 evaluation metrics을 평가 지표로 활용

### Evaluation Metrics

* Pairwise Metrics

  ```raw
  # Sequence alignment sample

  MGSIGIIIQVVT--EELNP (17)
  ::::::  ::::   :   
  MGSIGIV-QVVTFATEGVE (18)
  1234567890123456789
  ```

  | Name              | Definition                                                                          |          Sample |
  | ----------------- | ----------------------------------------------------------------------------------- | --------------: |
  | Accuracy          | correct match / total match                                                         |                 |
  | Alignment length  | Total number of columns in the final alignment                                      |              19 |
  | Aligned length    | Number of columns where both sequences have a residue.                              |              16 |
  | Identical length  | Among those aligned columns, the count where the two residues are exactly the same. |              10 |
  | Sequence identity | Identical length / the length of the longer sequence                                | 0.555 (= 10/18) |

* Statistical Metrics

  | Name   | Definition                                                                                                                                                                                  |
  | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | p-value | 주어진 score 이상이 우연히 발생할 정도를 나타낸다. 값이 작을수록 우연히 이런 좋은 정렬이 나올 가능성이 매우 낮아, 두 구조가 실제로 유사하다는 통계적 증거가 강하다는 뜻으로 해석할 수 있다. |

  * 정렬이 잘 되어 score가 높은 경우는 비율 매우 작고 대부분은 score가 낮게 계산될 때 Gumbel distribution을 따른다고 가정함.
  * Parameter $\mu$와 $\beta$로 $\mathrm{Gumbel}(\mu , \beta)$에 fitting함.

## Reference

* [1] [Foldseek](https://doi.org/10.1038/s41587-023-01773-0)
