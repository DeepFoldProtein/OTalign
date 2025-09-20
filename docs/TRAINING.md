# Training

## 🧬 CATH 기반 학습 데이터셋 구축 절차

### **1단계: 원본 데이터**

* CATH (v4.4.0) database입니다.
* CATH S40 데이터셋 (서열 유사도 40% 이하로 중복)과 전체 domain/superfamily list입니다.

### **2단계: 데이터 필터링 (선택 사항)**

* **길이 기준 필터링**: 계산 효율성을 위해 지나치게 긴 단백질 도메인을 제외할 수 있습니다.
* 도메인 길이가 300개 또는 1,000개 아미노산을 초과하는 경우 데이터셋에서 제외합니다.

### **3단계: 쌍 생성 및 균형 샘플링**

* **Positive 샘플 (상동 관계) 생성**:
  * **같은 CATH 상동성 슈퍼패밀리(Homologous Superfamily) 내**에서 도메인 쌍을 샘플링합니다. 이는 모델이 학습해야 할 '정답' 관계에 해당합니다.

* **Negative 샘플 (비상동 관계) 생성**:
  * **서로 다른 CATH 슈퍼패밀리**에 속한 도메인 쌍을 샘플링합니다.
  * 모든 조합은 수가 너무 많으므로, 전체 데이터셋의 균형을 맞추기 위해 **과소추출(undersampling)** 합니다.
  * 이 쌍들에 대해 TM-align을 실행하여 **TM-score가 0.2 미만**인 쌍들만 Negative 샘플로 사용합니다.

### **4단계: 정답(Ground Truth) 정렬 생성**

* **TM-align 실행**: 3단계에서 만든 모든 도메인 쌍에 대해 **TM-align과 같은 구조 기반 정렬 도구를 실행**합니다.
* **결과 저장**: 각 쌍에 대해 생성된 **구조 정렬 정보**와 **TM-score**를 저장합니다. 이것이 모델이 학습할 정답(label)이 됩니다.

### **5단계: 데이터셋 분할**

모델의 일반화 성능을 정확하게 평가하기 위해 데이터셋을 학습(train), 검증(validation), 테스트(test) 세트로 분할합니다.

* **Group-aware Splitting**: 데이터 유출(leakage)을 방지하기 위해 **같은 슈퍼패밀리나 폴드(fold)에 속한 도메인들은 반드시 같은 세트(train, val, 또는 test)에만 포함**되도록 분할합니다.
* 만약 같은 폴드의 일부 도메인이 학습 세트에 있고 나머지가 테스트 세트에 있다면, 모델은 구조(?)를 암기하여 쉽게 정답을 맞출 수 있어 성능이 과대평가될 수 있습니다.

## Traning objective

학습 데이터 쌍이 Positive인지 Negative인지에 따라 다른 Loss를 적용하는 조건부 함수입니다.

$$
L_{\text{total}} =
\begin{cases}
L_{\text{positive}} & \text{if positive pair} \\
L_{\text{negative}} & \text{if negative pair}
\end{cases}
$$여기서 각 Loss는 다음과 같이 정의됩니다.

* **Positive Pair의 경우**:

$$
L_{\text{positive}} = L_{\text{alignment}} + \lambda_1 \cdot L_{\text{sparsity}} \quad \lambda_1 := 0.1
$$

* **Negative Pair의 경우**:

$$
L_{\text{negative}} = \lambda_2 \cdot L_{\text{emptiness}} \quad \lambda_2 := 1.0
$$

### 1. 주된 정렬 Loss ($L_{\text{alignment}}$)

Positive pair에 대해서만 계산되며, 예측과 정답의 차이를 측정합니다.

$$
L_{\text{alignment (GenKL)}} = \sum_{i=1}^{L_A} \sum_{j=1}^{L_B} \left( Q_{ij} \log \frac{Q_{ij}}{P_{ij}} - Q_{ij} + P_{ij} \right)
$$

### 2. 정규화 항 (Regularization Terms)

#### **희소성 정규화 ($L_{\text{sparsity}}$ for Positive Pairs)**

Positive pair의 정렬 경로가 뚜렷하고 희소해지도록 유도합니다.
$$L_{\text{sparsity}} = \sum_{i=1}^{L_A} \sum_{j=1}^{L_B} |P_{ij}|$$

#### **'비움' 정규화 ($L_{\text{emptiness}}$ for Negative Pairs)**

Negative pair에 대해서는 정렬을 시도하지 않도록 예측 Plan의 총 질량을 0에 가깝게 만듭니다.
$$L_{\text{emptiness}} = \sum_{i=1}^{L_A} \sum_{j=1}^{L_B} |P_{ij}|$$

### 3. 기호 설명 (Notation)

* $A, B$: 각각 길이 $L_A, L_B$를 갖는 두 단백질 서열.
* $i, j$: 서열 $A$의 $i$번째 아미노산, 서열 $B$의 $j$번째 아미노산을 가리키는 인덱스.
* $P$: UOT Solver가 출력한 **Unnormalized** Transport Plan. $P_{ij}$는 $i$와 $j$ 간의 정렬에 할당된 '질량(mass)'입니다.
* $P'$: $P$를 전체 합으로 나누어 정규화한 **Normalized** Transport Plan.있습니다.
* $Q$: 정답(Ground Truth) 정렬 행렬. $Q_{ij}$는 $i$와 $j$가 실제로 정렬되면 1, 아니면 0의 값을 갖습니다.
* $\lambda_1, \lambda_2$: 각 정규화 항의 강도를 조절하는 하이퍼파라미터.
