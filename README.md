# AI History with Code — Vanilla to ResNet

고수준 API(`nn.Sequential` 등)에 의존하지 않고, 신경망 구조 발전의 핵심 아이디어를
직접 구현하고 학습 성능을 비교·분석하는 학습 프로젝트.

- 데이터셋: **Fashion-MNIST** (전 단계 동일 데이터로 학습 곡선 직접 비교)
- 공통 비교 지표: train/val loss & accuracy, 층별 gradient norm, 파라미터 수

## 로드맵

| Phase | 시대 | 모델 | 핵심 아이디어 | 상태 |
|-------|------|------|--------------|------|
| 1 | 1980s~ | NumPy MLP + 수동 역전파 | Chain rule, Vanishing Gradient | ✅ 구현 |
| 2 | 1998 | LeNet-5 | Convolution, Pooling, 공간 구조 보존 | ⬜ |
| 3 | 2012 | AlexNet | ReLU, Dropout, GPU 학습 | ⬜ |
| 4 | 2014 | VGG-11/16 | 3×3 커널 중첩, BatchNorm | ⬜ |
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
    ├── lenet5.py    # Phase 2 (스텁)
    ├── alexnet.py   # Phase 3 (스텁)
    ├── vgg.py       # Phase 4 (스텁)
    └── resnet.py    # Phase 5 (스텁)
notebooks/           # 단계별 실험 노트북
scripts/run_phase1.py  # CLI 재현 엔트리
results/             # 곡선 PNG + metric JSON
```

## 실행

```bash
uv sync                                      # 의존성 설치
uv run pytest                                # gradient check 유닛테스트
uv run python scripts/run_phase1.py          # Phase 1 실험 → results/*.png
uv run jupyter lab notebooks/                # 노트북 열기
```

## Phase 1 실험 요약

- **Exp A**: 3/5/10층 Sigmoid MLP — 층이 깊어질수록 초기층(입력 쪽)의
  gradient norm이 0으로 수렴 (Vanishing Gradient)
- **Exp B**: 동일 10층에서 Sigmoid(Xavier) vs ReLU(He) — ReLU가 수렴 속도에서 압도
- **Exp C**: 2층 baseline — 약 88% 검증 정확도