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
  * Sequence identity가 40%이하인 단백질 Cluster의 중심점들로 구성
  * 단백질 서열의 길이는 1000개 이하
  * One-to-All 계산을 통해 구조기반 정렬(DeepAlign)을 수행
  * Ground truth로 사용 가능
* SABMark
  * SCOP을 기반으로 Twilight Zone과 Superfamilies 두 세트로 구성

## Research Workflow and Methodological Approach

### Input Data

* Protein Language Model(PLM)을 이용하여 query와 targe의 embedding output을 사용
  * ESM-1b, ESM-1v, ESM-2
* Structure Tokenizer
  * Foldseek처럼 구조기반 token을 사용
* Multimodal model
  * DPLM-2

### Optimal Transport

* Neural Network
  * Input 데이터를 바로 사용하지 않고 학습가능한 neural network를 통과시킨 후 optimal transprot에 적용 
* Optimal Transport vs Unbalance Optimal Transport
* Soft-to-Hard Alignment (monotonic constraint)

### Leveraging Outputs from OT

* Direct
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
* dataset에 대해 alignment 수행 후 Evaluation Metrics을 평가 지표로 활용

### Evaluation Metrics

* Pairwise Metrics

  ```
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

  | Name    | Definition |
  | ------- | ---------- |
  | p-value |            |