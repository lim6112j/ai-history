# AI History with Code — Vanilla to ResNet

고수준 API(`nn.Sequential` 등)에 의존하지 않고, 신경망 구조 발전의 핵심 아이디어를
직접 구현하고 학습 성능을 비교·분석하는 학습 프로젝트.

- 데이터셋: **Fashion-MNIST** (전 단계 동일 데이터로 학습 곡선 직접 비교)
- 공통 비교 지표: train/val loss & accuracy, 층별 gradient norm, 파라미터 수

## 로드맵

| Phase | 시대 | 모델 | 핵심 아이디어 | 상태 |
|-------|------|------|--------------|------|
| 1 | 1980s~ | NumPy MLP + 수동 역전파 | Chain rule, Vanishing Gradient | ✅ 구현 |
| 2 | 1998 | LeNet-5 | Convolution, Pooling, 공간 구조 보존 | ✅ 구현 |
| 3 | 2012 | AlexNet | ReLU, Dropout, GPU 학습 | ✅ 구현 |
| 4 | 2014 | VGG-11/16 | 3×3 커널 중첩, BatchNorm | ✅ 구현 |
| 5 | 2015 | ResNet-18/50 | Skip Connection, Degradation 해결 | ⬜ |

## 구조

```
src/dl_history/
├── data.py          # Fashion-MNIST 로딩 (NumPy / DataLoader 모드)
├── train.py         # 공용 PyTorch 학습 루프 (Phase 2+)
├── metrics.py       # epoch별 metric + 층별 gradient norm 기록기
├── viz.py           # loss/acc 곡선, gradient norm plot
└── models/
    ├── mlp_numpy.py # Phase 1: 순수 NumPy MLP + 수동 역전파
    ├── lenet5.py    # Phase 2: LeNet-5
    ├── alexnet.py   # Phase 3: AlexNet
    ├── vgg.py       # Phase 4: VGG-11/16
    └── resnet.py    # Phase 5 (스텁)
notebooks/           # 단계별 실험 노트북
scripts/run_phase1.py  # CLI 재현 엔트리
results/             # 곡선 PNG + metric JSON
```

## 실행

```bash
uv sync                                      # 의존성 설치
uv run python -m pytest                       # gradient check 유닛테스트
uv run python scripts/run_phase1.py          # Phase 1 실험 → results/*.png
uv run python scripts/run_phase2.py          # Phase 2 실험 → results/*.png
uv run python scripts/run_phase3.py          # Phase 3 실험 → results/*.png
uv run python scripts/run_phase4.py          # Phase 4 실험 → results/*.png
uv run jupyter lab notebooks/                # 노트북 열기
```

## 개념 설명 문서

| 문서 | 내용 |
|------|------|
| [phase1_math_explain.md](phase1_math_explain.md) | Phase 1의 수학적 근거 (연쇄법칙, 기울기 소실, 수치 미분 검증) |
| [phase3_easy_explain.md](phase3_easy_explain.md) | Phase 3를 수식·코드 없이 비유와 예시로 설명 (ReLU, MaxPool, Dropout, Augmentation) |

## Phase 1 실험 요약

- **Exp A**: 3/5/10층 Sigmoid MLP — 층이 깊어질수록 초기층(입력 쪽)의
  gradient norm이 0으로 수렴 (Vanishing Gradient)
- **Exp B**: 동일 10층에서 Sigmoid(Xavier) vs ReLU(He) — ReLU가 수렴 속도에서 압도
- **Exp C**: 2층 baseline — 약 88% 검증 정확도

## Phase 2 실험 요약

- **Exp A**: 동일 LeNet-5에서 Sigmoid / Tanh / ReLU — Sigmoid는 학습 초반
  plateau를 매우 늦게 탈출하고, Tanh·ReLU가 수렴 속도에서 우위
- **Exp B**: LeNet-5(~62k 파라미터) vs flat MLP 784-128-10(~102k 파라미터) —
  파라미터가 더 적은데도 Convolution의 공간 구조 보존으로 더 높은 정확도
- 데이터 표준화(전체 train 통계)는 Phase 1과 동일하게 `data.py`의 loader에 적용

## Phase 3 실험 요약

AlexNet(약 770만 파라미터, 28×28 입력에 맞게 축소) · SGD lr=0.01, momentum=0.9, 15 epoch

- **Exp A (Dropout on/off)**: dropout=0.5가 train−val gap을 0.0357 → 0.0242로 줄이고
  검증 정확도도 0.9178 → 0.9245로 높임. Dropout이 없으면 train이 더 빨리
  과적합 방향으로 가지만 검증 성능은 뒤처진다.
- **Exp B (Data Augmentation on/off)**: RandomCrop(pad=4) + Flip을 적용하니
  검증 정확도가 0.9245 → 0.9080으로 **오히려 하락**. 28×28 이미지에 pad=4 crop은
  변형이 커서, 15 epoch로는 증강된 분포에 충분히 수렴하지 못한 것으로 해석된다
  (증강은 학습 난이도를 높이므로 epoch를 늘리면 재검토 필요).
- Dropout은 PyTorch의 inverted dropout이라 평가 시 rescaling이 불필요함을
  `tests/test_alexnet.py`에서 직접 검증.

## Phase 4 실험 요약

VGG-11(입력 28→32 zero padding, classifier 512) · SGD lr=0.01, momentum=0.9, 8 epoch

> ⚠️ 실험 실행이 중단되어 `results/phase4_*.png`와 `phase4_summary.json`은 아직
> 생성되지 않았다. `uv run python scripts/run_phase4.py`로 재생성한다.
> 아래 Exp A는 중단 전까지 측정된 값이다.

- **Exp A (3×3 중첩 vs 5×5 하나)**: 같은 receptive field에서 3×3 두 개(VGG-11)가
  파라미터 9.76M에 검증 정확도 **0.9195**, 5×5 하나(VGG-11 wide)가 11.39M에 **0.9155** —
  더 적은 파라미터로 더 높은 정확도. VGG 논문의 "3×3 중첩" 주장을 재현.
- **Exp B (BatchNorm / 초기화)**: ⏳ 미완료 — 재실행 필요.

---

# 부록: 전체 구현 플랜

인공지능 발전의 핵심 전환점이었던 **Backpropagation**부터 **ResNet**까지의 핵심 신경망 구조를
직접 구현하고 비교·분석하는 단계별 프로젝트 플랜. 각 단계마다 구조적 한계와 이를 극복한
아이디어를 코드로 체감하는 것을 목표로 한다.

## 프로젝트 개요

- **목표:** 라이브러리의 고수준 API(`nn.Sequential` 등)에 의존하지 않고, 신경망 구조의
  발전 과정을 핵심 파라미터 및 순전파/역전파 흐름 위주로 직접 구현하고 학습 성능을 비교 분석
- **데이터셋:** Fashion-MNIST (전 단계 동일 데이터셋으로 학습 곡선 직접 비교)
- **공통 비교 지표:**
  1. Train / Validation Loss & Accuracy 곡선
  2. 기울기 소실(Vanishing Gradient) 발생 여부 (층별 Gradient norm 시각화)
  3. 에폭당 학습 시간 및 파라미터 수

## Phase 1. Backpropagation & MLP (1980s~1990s)

**개념:** 신경망 학습의 기본 원리인 역전파(Backpropagation) 메커니즘과 다층 퍼셉트론(MLP) 구현

**핵심 구현 과제:**
- AutoGrad/PyTorch 자동미분 기능 없이, **NumPy만으로** 2층/3층 MLP 및 역전파 체인 룰(Chain Rule) 직접 계산
- Sigmoid / Tanh 활성화 함수와 Cross Entropy Loss의 미분식 작성

**실험 및 비교 Point:**
- 신경망 깊이(3층 → 5층 → 10층)를 늘릴 때 Sigmoid 함수로 인해 초기 레이어의 Gradient가
  0으로 수렴하는 **기울기 소실(Vanishing Gradient) 현상** 확인

## Phase 2. LeNet-5 & Early CNN (1998)

**개념:** 공간 구조(Spatial Information)를 보존하는 합성곱(Convolution)과 풀링(Pooling) 레이어 도입

**핵심 구현 과제:**
- PyTorch 기반으로 `Conv2d`, `AvgPool2d`, `Linear`를 조합하여 **LeNet-5** 클래스 작성
- Conv Layer 출력 크기 계산 공식 `W_out = (W − F + 2P) / S + 1` 적용 확인

**실험 및 비교 Point:**
- Phase 1의 MLP 대비 파라미터 수가 크게 줄어듦에도 불구하고 이미지 특징 Extraction 성능이 뛰어남을 비교
- 깊은 레이어 학습 시 Sigmoid/Tanh 대신 **ReLU**를 도입했을 때 학습 속도 향상 확인

## Phase 3. AlexNet (2012)

**개념:** GPU 기반 대규모 딥러닝의 시작, ReLU 도입, Overfitting 방지를 위한 Dropout 적용

**핵심 구현 과제:**
- ReLU 활성화 함수 적용
- **Dropout** 레이어 추가 및 테스트 시 scaling 처리 확인
- Max Pooling과 다중 Conv 레이어 연결

**실험 및 비교 Point:**
- Dropout 적용 전/후의 과적합(Overfitting) 수치 비교
- Data Augmentation(Crop, Flip 등) 적용에 따른 검증 정확도 변화 측정

## Phase 4. VGGNet (2014)

**개념:** 3×3 소형 커널의 중첩을 통한 깊은 신경망(Deep Network) 구성과 표현력 증가

**핵심 구현 과제:**
- 5×5 커널 1개 대신 3×3 커널 2개를 쌓는 **VGG-11 / VGG-16** 블록 단위 함수화 (`make_layers` 기법)
- 깊은 네트워크에서의 **Batch Normalization** 유무에 따른 학습 안정성 제어

**실험 및 비교 Point:**
- 파라미터 수 대비 Receptive Field의 변화 이해
- Batch Normalization이 Gradient Vanishing/Exploding 문제 완화 및 학습 속도에 미치는 영향 관찰

## Phase 5. ResNet (2015)

**개념:** 잔차 학습(Residual Learning) 및 Skip Connection을 통한 초고층 신경망 학습 극복

**핵심 구현 과제:**
- **BasicBlock** 및 **Bottleneck Block** 직접 작성 (`y = F(x) + x`)
- Downsampling 시 Shortcut Connection의 차원 맞춤(1×1 Conv) 구현
- ResNet-18 및 ResNet-50 구축

**실험 및 비교 Point:**
- Plain Network(Skip Connection 없음) vs ResNet의 20층/56층 학습 성능 비교
- Plain Network는 층이 깊어질수록 오히려 Train Error가 커지는 **Degradation Problem** 발생 확인
- ResNet은 층이 매우 깊어져도 안정적으로 Loss가 감소함을 시각화

## 로드맵 요약

```
[Phase 1] NumPy MLP & Manual Backprop        ✅ 구현 완료 (2026-10-07)
   └── Sigmoid의 한계 및 Vanishing Gradient 체감

[Phase 2] LeNet-5 (1998)                          ✅ 구현 완료 (2026-10-10)
   └── Spatial Feature 추출과 ReLU/Pooling의 효과

[Phase 3] AlexNet (2012)                          ✅ 구현 완료 (2026-10-11)
   └── Dropout으로 Overfitting 완화 (Data Augmentation은 epoch 부족으로 재검토)

[Phase 4] VGGNet (2014)                      ✅ 구현 완료 (2026-10-11)
   └── 3x3 Convolution 중첩과 Batch Normalization의 영향
       (실험 결과는 재실행 필요)

[Phase 5] ResNet (2015)
   └── Skip Connection 구현 및 Deep Network Degradation 해결
```

각 단계별로 재사용 모듈(`src/dl_history/`) + 실험 노트북(`notebooks/`) + CLI 재현
스크립트(`scripts/`) 조합으로 구성하며, 구조 발전의 아이디어를 손으로 명확하게 정립한다.