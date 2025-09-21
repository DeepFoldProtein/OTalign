# Training

## 🧬 CATH-based Training Dataset Construction

### **Step 1: Source Data**

* **CATH Database (v4.4.0)**: The primary source of protein structure information.
* **CATH S40 Dataset**: A subset of CATH where sequences have less than 40% identity, used for non-redundant sampling. Includes the complete list of domains and superfamilies.

### **Step 2: Data Filtering (Optional)**

* **Filter by Length**: To improve computational efficiency, excessively long protein domains can be excluded.
* **Rule**: Domains with lengths exceeding 300 or 1,000 amino acids are removed from the dataset.

### **Step 3: Pair Generation and Balanced Sampling**

* **Positive Samples (Homologous Pairs)**:
  * Domain pairs are sampled from **within the same CATH Homologous Superfamily**. These pairs represent the "ground truth" homologous relationships that the model needs to learn.

* **Negative Samples (Non-homologous Pairs)**:
  * Domain pairs are sampled from **different CATH superfamilies**.
  * To manage the vast number of possible combinations, we **undersample** the negative pairs to maintain a balanced dataset.
  * TM-align is run on these pairs, and only those with a **TM-score below 0.2** are retained as true negative samples.

### **Step 4: Ground Truth Alignment Generation**

* **Run Structure Alignment**: A structure-based alignment tool, such as **TM-align**, is executed on all domain pairs generated in Step 3.
* **Store Results**: The resulting **structural alignment** and the **TM-score** for each pair are saved. This information serves as the ground truth label for model training.

### **Step 5: Dataset Splitting**

To accurately evaluate the model's generalization performance, the dataset is split into training, validation, and test sets.

* **Group-aware Splitting**: To prevent data leakage, domains belonging to the **same superfamily or fold are strictly kept within the same set** (i.e., all in train, all in validation, or all in test).
* **Rationale**: If domains from the same fold were present in both the training and test sets, the model could simply "memorize" the fold's structure, leading to an overestimation of its true performance.

## Training Objective

The training objective uses a conditional loss function that varies depending on whether the input pair is positive (homologous) or negative (non-homologous).

$$
L_{\text{total}} =
\begin{cases}
L_{\text{positive}} & \text{if positive pair} \\
L_{\text{negative}} & \text{if negative pair}
\end{cases}
$$

The individual loss components are defined as follows:

* **For Positive Pairs**:
    $$
    L_{\text{positive}} = L_{\text{alignment}} + \lambda_1 \cdot L_{\text{sparsity}} \quad (\text{where } \lambda_1 = 0.1)
    $$

* **For Negative Pairs**:
    $$
    L_{\text{negative}} = \lambda_2 \cdot L_{\text{emptiness}} \quad (\text{where } \lambda_2 = 1.0)
    $$

### 1. Primary Alignment Loss ($L_{\text{alignment}}$)

This loss is calculated only for positive pairs and measures the discrepancy between the predicted alignment and the ground truth alignment. We use the Generalized Kullback-Leibler (GenKL) divergence.

$$
L_{\text{alignment (GenKL)}} = \sum_{i=1}^{L_A} \sum_{j=1}^{L_B} \left( Q_{ij} \log \frac{Q_{ij}}{P_{ij}} - Q_{ij} + P_{ij} \right)
$$

### 2. Regularization Terms

#### **Sparsity Regularization ($L_{\text{sparsity}}$ for Positive Pairs)**

This term encourages the predicted alignment path for positive pairs to be sharp and sparse.

$$
L_{\text{sparsity}} = \sum_{i=1}^{L_A} \sum_{j=1}^{L_B} |P_{ij}|
$$

#### **Emptiness Regularization ($L_{\text{emptiness}}$ for Negative Pairs)**

For negative pairs, this term forces the total mass of the predicted transport plan towards zero, effectively preventing the model from attempting an alignment.

$$
L_{\text{emptiness}} = \sum_{i=1}^{L_A} \sum_{j=1}^{L_B} |P_{ij}|
$$

### 3. Notation

* $A, B$: Two protein sequences with lengths $L_A$ and $L_B$, respectively.
* $i, j$: Indices for the $i$-th residue of sequence $A$ and the $j$-th residue of sequence $B$.
* $P$: The **unnormalized** transport plan output by the UOT solver. $P_{ij}$ represents the "mass" assigned to the alignment between residues $i$ and $j$.
* $Q$: The ground truth alignment matrix, where $Q_{ij}$ is 1 if residues $i$ and $j$ are aligned, and 0 otherwise.
* $\lambda_1, \lambda_2$: Hyperparameters that control the weight of the regularization terms.
