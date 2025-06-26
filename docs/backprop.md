# Backpropagation

To derive the backpropagation rule for the Sinkhorn iteration, we need to compute the gradients of a loss function that depends on the output of the Sinkhorn algorithm with respect to its inputs. The Sinkhorn iteration is commonly used in entropy-regularized optimal transport to compute a transport plan $\mathbf{P} \in \mathbb{R}^{m \times n}$ between two probability distributions $\mathbf{a} \in \mathbb{R}^m$ and $\mathbf{b} \in \mathbb{R}^n$, given a cost matrix $\mathbf{C} \in \mathbb{R}^{m \times n}$ and a regularization parameter $\epsilon > 0$. Since the Sinkhorn algorithm is iterative, directly differentiating through all iterations can be computationally expensive. Instead, we use implicit differentiation based on the fixed-point conditions at convergence to efficiently compute these gradients.

## Problem Setup

The entropy-regularized optimal transport problem is defined as:

$\min_{\mathbf{P}} \langle \mathbf{C}, \mathbf{P} \rangle - \epsilon H(\mathbf{P}),$

subject to the marginal constraints:

$\mathbf{P} \mathbf{1}_n = \mathbf{a}, \quad \mathbf{P}^\top \mathbf{1}_m = \mathbf{b},$

where $\langle \mathbf{C}, \mathbf{P} \rangle = \sum_{i,j} \mathbf{C}_{ij} \mathbf{P}_{ij}$, $H(\mathbf{P}) = -\sum_{i,j} \mathbf{P}_{ij} \log \mathbf{P}_{ij}$ is the entropy of $\mathbf{P}$, and $\mathbf{1}_k$ denotes a vector of ones in $\mathbb{R}^k$.

The Sinkhorn algorithm solves this by iteratively computing scaling vectors or, equivalently, dual variables $\alpha \in \mathbb{R}^m$ and $\beta \in \mathbb{R}^n$, such that the transport plan takes the form:

$\mathbf{P}_{ij} = e^{(\alpha_i + \beta_j - \mathbf{C}_{ij}) / \epsilon}.$

At convergence, $\mathbf{P}$ satisfies the marginal constraints:

$\sum_j \mathbf{P}_{ij} = a_i, \quad \sum_i \mathbf{P}_{ij} = b_j.$

In practice, the Sinkhorn iterations are often performed in log-space for numerical stability, updating $\alpha$ and $\beta$ until convergence, but for backpropagation, we focus on the fixed-point condition rather than the iteration steps.

Suppose we have a loss function $L = L(\mathbf{P})$ that depends on $\mathbf{P}$, and we wish to compute gradients such as $\frac{\partial L}{\partial \mathbf{C}}$, $\frac{\partial L}{\partial \mathbf{a}}$, and $\frac{\partial L}{\partial \mathbf{b}}$. Since $\mathbf{P}$ is a function of $\mathbf{C}$, $\mathbf{a}$, and $\mathbf{b}$ through the Sinkhorn solution, and $\alpha$ and $\beta$ are implicitly determined by the marginal constraints, we use the implicit function theorem to derive these gradients.

## Fixed-Point Equation

Define the residual function based on the marginal constraints:

$\mathbf{f}(\alpha, \beta; \mathbf{C}, \mathbf{a}, \mathbf{b}) = \begin{pmatrix} \mathbf{f}_\alpha \\ \mathbf{f}_\beta \end{pmatrix} = \begin{pmatrix} \mathbf{a} - \mathbf{P} \mathbf{1}_n \\ \mathbf{b} - \mathbf{P}^\top \mathbf{1}_m \end{pmatrix},$

where $\mathbf{P}_{ij} = e^{(\alpha_i + \beta_j - \mathbf{C}_{ij}) / \epsilon}$, $\mathbf{f}_\alpha \in \mathbb{R}^m$, and $\mathbf{f}_\beta \in \mathbb{R}^n$. At the Sinkhorn solution, after convergence:

$\mathbf{f}(\alpha, \beta; \mathbf{C}, \mathbf{a}, \mathbf{b}) = 0,$

meaning $\mathbf{P} \mathbf{1}_n = \mathbf{a}$ and $\mathbf{P}^\top \mathbf{1}_m = \mathbf{b}$. Here, $\alpha$ and $\beta$ are implicit functions of $\mathbf{C}$, $\mathbf{a}$, and $\mathbf{b}$.

## Implicit Differentiation

To compute the gradients, consider $\alpha = \alpha(\mathbf{C}, \mathbf{a}, \mathbf{b})$ and $\beta = \beta(\mathbf{C}, \mathbf{a}, \mathbf{b})$ as implicitly defined by $\mathbf{f} = 0$. For a small perturbation in the inputs (e.g., $d\mathbf{C}$), the total differential of $\mathbf{f}$ is:

$d \mathbf{f} = \frac{\partial \mathbf{f}}{\partial \alpha} d\alpha + \frac{\partial \mathbf{f}}{\partial \beta} d\beta + \frac{\partial \mathbf{f}}{\partial \mathbf{C}} d\mathbf{C} + \frac{\partial \mathbf{f}}{\partial \mathbf{a}} d\mathbf{a} + \frac{\partial \mathbf{f}}{\partial \mathbf{b}} d\mathbf{b} = 0.$

Since the loss $L$ depends on $\mathbf{P}$, and:

$\mathbf{P} = \mathbf{P}(\alpha, \beta, \mathbf{C}),$

the differential of $L$ is:

$dL = \sum_{i,j} \frac{\partial L}{\partial \mathbf{P}_{ij}} d\mathbf{P}_{ij},$

where:

$d\mathbf{P}_{ij} = \frac{\partial \mathbf{P}_{ij}}{\partial \alpha_i} d\alpha_i + \frac{\partial \mathbf{P}_{ij}}{\partial \beta_j} d\beta_j + \frac{\partial \mathbf{P}_{ij}}{\partial \mathbf{C}_{ij}} d\mathbf{C}_{ij}.$

We need to compute these partial derivatives and relate $d\alpha$ and $d\beta$ to the input differentials.

### Step 1: Compute Partial Derivatives of $\mathbf{P}$

Since $\mathbf{P}_{ij} = e^{(\alpha_i + \beta_j - \mathbf{C}_{ij}) / \epsilon}$:

- $\frac{\partial \mathbf{P}_{ij}}{\partial \alpha_k} = \frac{1}{\epsilon} \mathbf{P}_{ij} \delta_{ik}$, so $\frac{\partial \mathbf{P}_{ij}}{\partial \alpha_i} = \frac{1}{\epsilon} \mathbf{P}_{ij}$, and 0 if $k \neq i$,
- $\frac{\partial \mathbf{P}_{ij}}{\partial \beta_l} = \frac{1}{\epsilon} \mathbf{P}_{ij} \delta_{jl}$, so $\frac{\partial \mathbf{P}_{ij}}{\partial \beta_j} = \frac{1}{\epsilon} \mathbf{P}_{ij}$, and 0 if $l \neq j$,
- $\frac{\partial \mathbf{P}_{ij}}{\partial \mathbf{C}_{kl}} = -\frac{1}{\epsilon} \mathbf{P}_{ij} \delta_{ik} \delta_{jl}$, so $\frac{\partial \mathbf{P}_{ij}}{\partial \mathbf{C}_{ij}} = -\frac{1}{\epsilon} \mathbf{P}_{ij}$, and 0 otherwise.

Thus:

$d\mathbf{P}_{ij} = \frac{1}{\epsilon} \mathbf{P}_{ij} d\alpha_i + \frac{1}{\epsilon} \mathbf{P}_{ij} d\beta_j - \frac{1}{\epsilon} \mathbf{P}_{ij} d\mathbf{C}_{ij}.$

### Step 2: Compute the Jacobian $\frac{\partial \mathbf{f}}{\partial (\alpha, \beta)}$

Define $\mathbf{z} = \begin{pmatrix} \alpha \\ \beta \end{pmatrix}$, so the Jacobian is:

$J = \frac{\partial \mathbf{f}}{\partial \mathbf{z}} = \begin{pmatrix} \frac{\partial \mathbf{f}_\alpha}{\partial \alpha} & \frac{\partial \mathbf{f}_\alpha}{\partial \beta} \\ \frac{\partial \mathbf{f}_\beta}{\partial \alpha} & \frac{\partial \mathbf{f}_\beta}{\partial \beta} \end{pmatrix}.$

- **$\frac{\partial \mathbf{f}_\alpha}{\partial \alpha}$**: For $f_{\alpha,i} = a_i - \sum_j \mathbf{P}_{ij}$,

$\frac{\partial f_{\alpha,i}}{\partial \alpha_k} = -\sum_j \frac{\partial \mathbf{P}_{ij}}{\partial \alpha_k} = -\sum_j \frac{1}{\epsilon} \mathbf{P}_{ij} \delta_{ik} = -\frac{1}{\epsilon} \sum_j \mathbf{P}_{ij} \delta_{ik}.$

If $k = i$, $\frac{\partial f_{\alpha,i}}{\partial \alpha_i} = -\frac{1}{\epsilon} \sum_j \mathbf{P}_{ij} = -\frac{a_i}{\epsilon}$ (since $\sum_j \mathbf{P}_{ij} = a_i$); if $k \neq i$, it’s 0. Thus, $\frac{\partial \mathbf{f}_\alpha}{\partial \alpha} = -\frac{1}{\epsilon} \text{diag}(\mathbf{a})$.

- **$\frac{\partial \mathbf{f}_\alpha}{\partial \beta}$**: $\frac{\partial f_{\alpha,i}}{\partial \beta_l} = -\sum_j \frac{\partial \mathbf{P}_{ij}}{\partial \beta_l} = -\frac{1}{\epsilon} \mathbf{P}_{il} \delta_{ll} = -\frac{1}{\epsilon} \mathbf{P}_{il}$, so $\frac{\partial \mathbf{f}_\alpha}{\partial \beta} = -\frac{1}{\epsilon} \mathbf{P}$.

- **$\frac{\partial \mathbf{f}_\beta}{\partial \alpha}$**: For $f_{\beta,j} = b_j - \sum_i \mathbf{P}_{ij}$, $\frac{\partial f_{\beta,j}}{\partial \alpha_k} = -\frac{1}{\epsilon} \mathbf{P}_{kj}$, so $\frac{\partial \mathbf{f}_\beta}{\partial \alpha} = -\frac{1}{\epsilon} \mathbf{P}^\top$.

- **$\frac{\partial \mathbf{f}_\beta}{\partial \beta}$**: $\frac{\partial f_{\beta,j}}{\partial \beta_l} = -\frac{1}{\epsilon} \sum_i \mathbf{P}_{ij} \delta_{jl} = -\frac{b_j}{\epsilon} \delta_{jl}$, so $\frac{\partial \mathbf{f}_\beta}{\partial \beta} = -\frac{1}{\epsilon} \text{diag}(\mathbf{b})$.

Thus:

$J = -\frac{1}{\epsilon} \begin{pmatrix} \text{diag}(\mathbf{a}) & \mathbf{P} \\ \mathbf{P}^\top & \text{diag}(\mathbf{b}) \end{pmatrix}.$

### Step 3: Differentiate $\mathbf{f} = 0$ with Respect to Inputs

For $d\mathbf{C}$:

$J \begin{pmatrix} d\alpha \\ d\beta \end{pmatrix} + \frac{\partial \mathbf{f}}{\partial \mathbf{C}} d\mathbf{C} = 0.$

Compute $\frac{\partial \mathbf{f}}{\partial \mathbf{C}}$ by considering $\mathbf{P}$’s dependence on $\mathbf{C}$ with $\alpha, \beta$ fixed:

$d\mathbf{P}_{ij} = -\frac{1}{\epsilon} \mathbf{P}_{ij} d\mathbf{C}_{ij},$

so $d\mathbf{P} = -\frac{1}{\epsilon} \mathbf{P} \odot d\mathbf{C}$, where $\odot$ denotes element-wise multiplication. Then:

- $d \mathbf{f}_\alpha = -d\mathbf{P} \mathbf{1}_n = \left( \sum_j \frac{1}{\epsilon} \mathbf{P}_{ij} d\mathbf{C}_{ij} \right)_i$,
- $d \mathbf{f}_\beta = -d\mathbf{P}^\top \mathbf{1}_m = \left( \sum_i \frac{1}{\epsilon} \mathbf{P}_{ij} d\mathbf{C}_{ij} \right)_j$.

Thus:

$\frac{\partial \mathbf{f}}{\partial \mathbf{C}} d\mathbf{C} = \begin{pmatrix} \left( \sum_j \frac{1}{\epsilon} \mathbf{P}_{ij} d\mathbf{C}_{ij} \right)_i \\ \left( \sum_i \frac{1}{\epsilon} \mathbf{P}_{ij} d\mathbf{C}_{ij} \right)_j \end{pmatrix},$

and:

$J \begin{pmatrix} d\alpha \\ d\beta \end{pmatrix} = -\begin{pmatrix} \mathbf{P}^\top d\mathbf{C} \mathbf{1}_n \\ \mathbf{P} d\mathbf{C}^\top \mathbf{1}_m \end{pmatrix}.$

For $d\mathbf{a}$:

$\frac{\partial \mathbf{f}}{\partial \mathbf{a}} = \begin{pmatrix} \mathbf{I}_m \\ \mathbf{0} \end{pmatrix}, \quad J \begin{pmatrix} d\alpha \\ d\beta \end{pmatrix} = -\begin{pmatrix} d\mathbf{a} \\ \mathbf{0} \end{pmatrix}.$

For $d\mathbf{b}$:

$\frac{\partial \mathbf{f}}{\partial \mathbf{b}} = \begin{pmatrix} \mathbf{0} \\ \mathbf{I}_n \end{pmatrix}, \quad J \begin{pmatrix} d\alpha \\ d\beta \end{pmatrix} = -\begin{pmatrix} \mathbf{0} \\ d\mathbf{b} \end{pmatrix}.$

Solve for $\begin{pmatrix} d\alpha \\ d\beta \end{pmatrix}$ using $J \begin{pmatrix} d\alpha \\ d\beta \end{pmatrix} = -\mathbf{h}$, where $\mathbf{h}$ is the right-hand side.

### Step 4: Compute $dL$

$dL = \sum_{i,j} \frac{\partial L}{\partial \mathbf{P}_{ij}} \left( \frac{1}{\epsilon} \mathbf{P}_{ij} d\alpha_i + \frac{1}{\epsilon} \mathbf{P}_{ij} d\beta_j - \frac{1}{\epsilon} \mathbf{P}_{ij} d\mathbf{C}_{ij} \right).$

Rewrite as:

$dL = \sum_i \left( \sum_j \frac{1}{\epsilon} \mathbf{P}_{ij} \frac{\partial L}{\partial \mathbf{P}_{ij}} \right) d\alpha_i + \sum_j \left( \sum_i \frac{1}{\epsilon} \mathbf{P}_{ij} \frac{\partial L}{\partial \mathbf{P}_{ij}} \right) d\beta_j - \sum_{i,j} \frac{1}{\epsilon} \mathbf{P}_{ij} \frac{\partial L}{\partial \mathbf{P}_{ij}} d\mathbf{C}_{ij}.$

Define:

- $\mathbf{w}_\alpha = \frac{1}{\epsilon} \mathbf{P} \left( \frac{\partial L}{\partial \mathbf{P}} \right)^\top \mathbf{1}_n$,
- $\mathbf{w}_\beta = \frac{1}{\epsilon} \mathbf{P}^\top \frac{\partial L}{\partial \mathbf{P}} \mathbf{1}_m$.

Let $\mathbf{w} = \begin{pmatrix} \mathbf{w}_\alpha \\ \mathbf{w}_\beta \end{pmatrix}$, so:

$dL = \mathbf{w}^\top \begin{pmatrix} d\alpha \\ d\beta \end{pmatrix} - \frac{1}{\epsilon} \left\langle \mathbf{P} \odot \frac{\partial L}{\partial \mathbf{P}}, d\mathbf{C} \right\rangle.$

Since $\begin{pmatrix} d\alpha \\ d\beta \end{pmatrix} = -J^{-1} \mathbf{h}$, we have $dL = -\mathbf{w}^\top J^{-1} \mathbf{h} - \frac{1}{\epsilon} \left\langle \mathbf{P} \odot \frac{\partial L}{\partial \mathbf{P}}, d\mathbf{C} \right\rangle$.

### Step 5: Extract Gradients

- **For $\mathbf{C}$**: $\mathbf{h} = \begin{pmatrix} \mathbf{P}^\top d\mathbf{C} \mathbf{1}_n \\ \mathbf{P} d\mathbf{C}^\top \mathbf{1}_m \end{pmatrix}$. Let $\begin{pmatrix} \lambda \\ \mu \end{pmatrix} = -J^{-1} \mathbf{w}$, then:

$\frac{\partial L}{\partial \mathbf{C}_{ij}} = -\frac{1}{\epsilon} \mathbf{P}_{ij} \frac{\partial L}{\partial \mathbf{P}_{ij}} - \lambda_i - \mu_j,$

where $\begin{pmatrix} \lambda \\ \mu \end{pmatrix}$ satisfies:

$J \begin{pmatrix} \lambda \\ \mu \end{pmatrix} = -\begin{pmatrix} \mathbf{w}_\alpha \\ \mathbf{w}_\beta \end{pmatrix}.$

- **For $\mathbf{a}$**: $\mathbf{h} = \begin{pmatrix} d\mathbf{a} \\ \mathbf{0} \end{pmatrix}$, so $\frac{\partial L}{\partial \mathbf{a}} = -\lambda$.

- **For $\mathbf{b}$**: $\mathbf{h} = \begin{pmatrix} \mathbf{0} \\ d\mathbf{b} \end{pmatrix}$, so $\frac{\partial L}{\partial \mathbf{b}} = -\mu$.

## Final Backpropagation Rule

Given $\frac{\partial L}{\partial \mathbf{P}}$:

1. Compute:
    $\mathbf{w}_\alpha = \frac{1}{\epsilon} \mathbf{P} \left( \frac{\partial L}{\partial \mathbf{P}} \right)^\top \mathbf{1}_n$, $\mathbf{w}_\beta = \frac{1}{\epsilon} \mathbf{P}^\top \frac{\partial L}{\partial \mathbf{P}} \mathbf{1}_m$.
2. Solve the linear system:
    $-\frac{1}{\epsilon} \begin{pmatrix} \text{diag}(\mathbf{a}) & \mathbf{P} \\ \mathbf{P}^\top & \text{diag}(\mathbf{b}) \end{pmatrix} \begin{pmatrix} \lambda \\ \mu \end{pmatrix} = -\begin{pmatrix} \mathbf{w}_\alpha \\ \mathbf{w}_\beta \end{pmatrix}$.
3. Compute gradients:
    - $\frac{\partial L}{\partial \mathbf{C}_{ij}} = -\frac{1}{\epsilon} \mathbf{P}_{ij} \frac{\partial L}{\partial \mathbf{P}_{ij}} - \lambda_i - \mu_j$,
    - $\frac{\partial L}{\partial \mathbf{a}} = -\lambda$,
    - $\frac{\partial L}{\partial \mathbf{b}} = -\mu$.
