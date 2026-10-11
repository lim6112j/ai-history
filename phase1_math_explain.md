# Phase 1 — Backpropagation & MLP: 수학적 근거와 주요 개념

`notebooks/phase1_mlp_backprop.ipynb`의 수학적 배경과 용어 설명.

---

## 1. 전방 계산 (MLP)

각 층은 affine 변환 + 활성화:

$$a^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}, \quad h^{(l)} = \phi(a^{(l)})$$

마지막 층 출력 $z = W^{(L)} h^{(L-1)} + b^{(L)}$ 에 softmax를 쓴 뒤 교차 엔트로피 손실:

$$\mathcal{L} = -\log p_y, \quad p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$$

## 2. 역전파 = 연쇄법칙 (Chain Rule)

손실의 각 파라미터에 대한 미분은 연쇄법칙의 곱 형태이고, 이걸 **출력에서 입력 방향으로 층별로 재사용**하는 게 역전파:

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} \, h^{(l-1)\top}, \qquad \delta^{(l)} = \left( W^{(l+1)\top} \delta^{(l+1)} \right) \odot \phi'(a^{(l)})$$

핵심 항 3개 (노트북의 `Dense`, `Sigmoid`, `SoftmaxCrossEntropy` 클래스에 대응):

- **Softmax + CE 결합**: $\dfrac{\partial \mathcal{L}}{\partial z} = p - \text{onehot}(y)$ — softmax의 야코비안을 전개하면 하나의 항으로 합쳐짐
- **Sigmoid**: $\phi'(a) = \phi(a)(1 - \phi(a))$
- **ReLU**: $\phi'(a) = \mathbf{1}[a > 0]$

## 3. 구현 검증 — 수치 미분

체인 룰 코드가 맞는지 **중앙차분(centered difference)** 으로 검증:

$$\frac{\partial \mathcal{L}}{\partial w_i} \approx \frac{\mathcal{L}(w_i + \epsilon) - \mathcal{L}(w_i - \epsilon)}{2\epsilon}$$

오차 $O(\epsilon^2)$ 이므로 $\epsilon = 10^{-5}$일 때 정확한 구현은 상대오차 < 1e-6. 이게 "구현이 수학적으로 옳다"는 근거.

## 4. Vanishing Gradient (실험 A)

$|\sigma'(a)| \le 0.25$ ($a=0$에서 최대)이므로, 역전파 한 층을 거칠 때마다 gradient 크기에 최대 0.25 배수가 곱해짐:

$$\|\delta^{(l)}\| \lesssim (0.25)^{L-l} \cdot \prod \|W\| \cdot \|\delta^{(L)}\|$$

10층이면 첫 층 gradient는 $\sim 0.25^9 \approx 4 \times 10^{-6}$ 수준으로 지수적 감쇠 → log scale에서 층별 $\|dW\|$ 가 깊이별로 격차가 남.

## 5. ReLU + He init이 해결하는 이유 (실험 B/C)

- ReLU 미분은 양수 영역에서 정확히 **1** → 감쇠 계수가 곱해지지 않음
- He init ($W \sim \mathcal{N}(0, 2/n_{in})$)은 순전파/역전파 분산을 층마다 유지하도록 설계됨: $\text{Var}[\delta^{(l)}] \approx \text{Var}[\delta^{(l+1)}]$ — ReLU가 절반을 0으로 만드는 것을 보상하기 위한 2배 인자

**요약**: 연쇄법칙을 층 단위 모듈로 분해해 손으로 구현하고, 수치 미분으로 검증한 뒤, gradient 곱셈 구조에서 나오는 지수적 감쇠(문제)와 ReLU/He init(해결)을 실험으로 확인.

---

## 주요 용어 개념

### 데이터 관련

| 용어 | 뜻 |
|---|---|
| **Flatten** | 28×28 이미지를 784(=28×28)차원의 1차원 벡터로 펼치는 것. CNN은 공간 구조를 쓰지만 MLP는 1차원 입력만 받음 |
| **표준화 (Standardization)** | `(x - mean) / std` 로 평균 0, 분산 1로 맞추는 변환. 학습 데이터 통계로 변환해야 하고, 테스트에도 **같은** 통계를 적용 |
| **Zero-centered** | 입력 평균이 0 근처. 활성화 함수(특히 ReLU)가 입력 평균이 0일 때 기울기 흐름이 안정적이라 깊은 네트워크에 필수 |

### 모델 관련

| 용어 | 뜻 |
|---|---|
| **MLP** (다층 퍼셉트론) | Dense 층을 여러 개 쌓은 가장 기본적인 신경망. 층마다 `W·a + b` → 활성화 함수 |
| **Dense (완전연결 층)** | 이전 층의 모든 뉴런이 다음 층의 모든 뉴런과 연결된 층. 파라미터 = 행렬 W + 벡터 b |
| **활성화 함수 (φ)** | 층 사이에 넣는 비선형 함수 (Sigmoid, ReLU, Tanh). 없으면 아무리 쌓아도 선형 모델과 같아짐 |
| **Softmax** | 출력 벡터를 **확률 분포**로 바꾸는 함수. 모든 원소가 0~1이고 합이 1이 됨 |
| **One-hot 인코딩** | 정답 클래스를 `[0,0,1,0,...]`처럼 해당 자리만 1인 벡터로 표현 |
| **He init** | 가중치 초기화 방법. `std = sqrt(2 / n_in)` 로 설정해 ReLU 네트워크에서 층을 거쳐도 신호 분산이 유지되도록 설계 |

### 학습 관련

| 용어 | 뜻 |
|---|---|
| **Loss (손실 함수)** | "모델이 얼마나 틀렸는지"를 숫자로 나타낸 것. 학습 = 이 값을 최소화하는 파라미터를 찾는 것 |
| **Cross-Entropy** | 분류 문제의 표준 손실. `-log(정답 클래스의 예측 확률)` — 정답 확률이 높을수록 loss가 0에 가까움 |
| **Gradient (기울기)** | 손실을 각 파라미터로 편미분한 벡터. "파라미터를 어느 방향으로 움직이면 loss가 커지는가"를 가리킴 |
| **역전파 (Backpropagation)** | gradient를 출력층에서 입력층 방향으로 연쇄법칙으로 효율적으로 계산하는 알고리즘 |
| **Learning rate (학습률, LR)** | gradient 반대 방향으로 파라미터를 얼마나 크게 업데이트할지 정하는 상수. 노트북에서는 0.1 |
| **Epoch** | 전체 학습 데이터를 한 번 다 보는 단위. 노트북에서는 10 epochs |
| **Batch (미니배치)** | 한 번의 gradient 계산에 사용하는 데이터 묶음 (128개). 전체 데이터를 다 쓰면 너무 비싸므로 나눠서 계산 |
| **SGD (확률적 경사 하강)** | `W ← W - lr × gradient` 로 미니배치마다 파라미터를 갱신하는 방식 |

### 평가/검증 관련

| 용어 | 뜻 |
|---|---|
| **Accuracy (정확도)** | 예측이 정답과 일치한 비율. 가장 직관적인 분류 지표 |
| **Train / Val (학습/검증)** | 학습에 쓰는 데이터와, 학습에 안 쓰고 성능을 확인하는 데이터. val acc가 train acc보다 낮으면 과적합 의심 |
| **수치 미분 (Numerical differentiation)** | `(L(w+ε) - L(w-ε)) / 2ε` 처럼 정의 그대로 계산하는 미분. 느리지만 확실하므로 역전파 구현 검증용으로 씀 |
| **상대오차 (rel err)** | `|수치미분 - 역전파| / max(|수치미분|, |역전파|)`. 두 방법이 일치하는 정도를 측정 |

### 실험 관련

| 용어 | 뜻 |
|---|---|
| **Vanishing gradient (기울기 소실)** | 역전파는 층별 미분 값의 **곱**이므로, 미분 절댓값이 1보다 작으면(Sigmoid는 최대 0.25) 층을 거칠수록 기울기가 지수적으로 0에 수렴 → 앞쪽 층이 학습되지 않음 |
| **Gradient norm (‖dW‖₂)** | gradient 벡터의 크기(L2 노름). vanishing gradient를 관찰할 때 쓰는 지표 — "기울기가 얼마나 큰지" |
| **Log scale** | 값들이 수십~수만 배 차이가 나므로(지수적 감쇠) 로그 축으로 그려야 변화가 잘 보임 |
| **Depth (깊이)** | 층의 개수. 깊을수록 표현력은 커지지만 학습이 어려워짐 — 실험 A가 바로 이 트레이드오프를 보여줌 |