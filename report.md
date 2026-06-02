# 촉각 깊이 추정 모델 비교 및 TouchNet 추론 최적화 보고서

## 0. 요약 (TL;DR)

- **TouchNet(CNN)** 은 DPT(ViT) 대비 **추정 품질이 좋았지만 추론이 느렸다.**
- 원인을 분석해 **저해상도 추론 + GPU Poisson + torch.compile** 3가지 최적화를 적용한 결과,
  TouchNet 추론이 **BASE 대비 7.8배 빨라졌고(20fps → 155fps)**, 형상은 거의 그대로 보존되었다.
- 최적화된 TouchNet 은 **DPT 보다도 빠르면서**(155fps vs ~86fps) 품질 이점을 유지한다.

---

## 1. 배경 및 목표

DIGIT 비전 촉각 센서의 접촉 깊이(depth/heightmap)를 추정하는 두 모델을 같은 입력으로 비교했다.

| 모델 | 구조 | 출처 |
|------|------|------|
| **TouchNet** | Fully-Convolutional CNN (9-layer, 큰 커널, 풀해상도) | py3DCal 사전학습 (`digit_pretrained_weights.pth`) |
| **DPT** | Dense Prediction Transformer (ViT-small/16) | NeuralFeels tactile transformer (`dpt_real.p`, `dpt_sim.p`) |

상세 아키텍처(레이어 흐름 다이어그램)는 [`ARCHITECTURE.md`](ARCHITECTURE.md) 참고.

목표:
1. 두 모델의 **품질·속도** 비교
2. 느린 쪽(TouchNet)의 **추론 속도 최적화** 및 정량 검증

---

## 2. 실험 환경

- GPU: **NVIDIA RTX 4070 Laptop**, CUDA 12.8, FP16
- conda env `py3dcal` (Python 3.10, torch 2.11+cu128)
- 센서: DIGIT (예: D21424), 입력 240×320
- DPT 는 `timm` + 로컬 `examples/dpt_lite/`(neuralfeels 모델 코드 복사본)로 **경량 구동**(neuralfeels 전체 설치 불필요)
- 공정 비교: 두 계열 모두 **무접촉(blank) 기준 상대 접촉깊이**로 정렬
  (TouchNet=입력에서 blank 차감 / DPT=heightmap − blank_heightmap)

---

## 3. 결과 1 — TouchNet vs DPT

### 3.1 품질 (정성 관찰)

- **DPT(`dpt_real`)**: 본 DIGIT 입력에서 출력 heightmap 이 거의 **포화(예: [254,255])** 되어 접촉 반응이 약했다.
- **DPT(`dpt_sim`)**: real 보다 변별 구조가 살아 있었으나(예: [95,255]), 절대 스케일이 모호.
- **TouchNet**: blank 차감 + Poisson 적분으로 **접촉 형상이 또렷**하고, `px_per_mm` 로 **mm 단위**까지 직접 산출.
- → **품질은 TouchNet 우세**로 판단(엄밀한 GT 기반 정량 평가는 아님, 실시간 스트리밍 관찰 기준).

### 3.2 속도 (실측, FP16, 동일 프레임)

```
TouchNet(CNN)   :  49.37 ms/frame  ->   20.3 fps
dpt_real(ViT)   :  11.60 ms/frame  ->   86.2 fps
dpt_sim(ViT)    :  12.54 ms/frame  ->   79.7 fps
=> DPT 가 TouchNet 대비 약 4.25배 빠름
```

**핵심 모순**: 파라미터는 TouchNet(≈3.9M) < DPT(≈25.9M) 인데도 **DPT 가 더 빠르다.**

### 3.3 왜 DPT 가 더 빠른가 (분석)

1. **파라미터 수 ≠ 연산량.** TouchNet 은 적은 가중치를 **320×240 풀해상도**에서 매 레이어 반복 적용 → FLOPs 큼.
2. **해상도.** TouchNet 은 9겹 내내 다운샘플 없음. DPT 는 224×224 → patch16 → **196 토큰(≈8.6× 압축)**.
3. **연산 종류.** TouchNet 의 7×7·5×5 큰 conv 는 GPU 활용률 낮음. ViT 는 대부분 **행렬곱(GEMM)** → **FP16 텐서코어**에 최적.
4. **메모리 대역폭.** 256ch 풀해상도 활성값 vs 196×384 토큰.
5. **CPU 후처리.** TouchNet 은 매 프레임 **CPU Poisson 적분**(scipy DST/IDST) + GPU↔CPU 동기화가 붙음.

(상세 분석: [`README.md`](README.md) "왜 이런 결과가 나오는가 — 속도 분석" 절)

---

## 4. TouchNet 추론 최적화

분석에서 도출한 3가지 최적화를 적용했다.

| # | 최적화 | 내용 | 근거 |
|---|--------|------|------|
| 1 | **저해상도 추론** | 입력을 1/2(160×120)로 줄여 추론 후 깊이 업스케일. conv FLOPs ∝ H×W → 최대 효과. 좌표 임베딩은 원본 픽셀 범위(1..W,1..H)로 생성해 학습 분포 정합. | §3.3-(2) |
| 2 | **GPU Poisson** | scipy CPU DST/IDST 를 **torch matmul** 로 GPU 에서 수행(scipy 와 수치 동일, 최대오차 ~1e-4). GPU↔CPU 동기화 1회로 축소. | §3.3-(5) |
| 3 | **torch.compile** | conv+BN+ReLU 퓨전/커널 최적화(+ FP16 + channels_last). | §3.3-(3,4) |

### 4.1 결과 2 — 최적화 전/후 (BASE vs OPT)

- **BASE**: FP32 · 풀해상도 · scipy Poisson · compile 미사용 (= 아무 최적화 없음)
- **OPT** : FP16 · 1/2 해상도 · GPU Poisson · torch.compile

```
=== 속도 비교 요약 ===
BASE (fp32, full-res, scipy) :  50.31 ms ->   19.9 fps
OPT  (fp16, 1/2, GPU+compile):   6.45 ms ->  155.0 fps
=> OPT 가 BASE 대비 7.80배 빠름
[마지막 프레임] 결과 corr=0.872, 스칼라정합 k=0.345
```

| | 구성 | ms/frame | fps |
|---|------|---------:|----:|
| **BASE** | fp32 · full-res · scipy Poisson | 50.31 | 19.9 |
| **OPT** | fp16 · 1/2 res · GPU Poisson · compile | **6.45** | **155.0** |

- **속도: 7.80× 향상** (50.3ms → 6.5ms, 19.9 → 155.0 fps)
- **형상 보존: corr = 0.872** (BASE 대비 OPT 깊이맵 형상 상관)
- `k = 0.345`: 저해상도 Poisson 은 **절대 깊이 스케일을 바꾼다**(형상은 보존, mm 절대값은 재보정 필요). 그래서 비교는 스칼라 정합 후 형상으로 평가.

> 참고: `corr`/`k` 는 측정 시 접촉이 있을 때 의미가 있다. 무접촉(평탄)에서는 노이즈 기준값이 된다.

### 4.2 단계별 기여 (참고 측정)

| 구성 | ms/frame | fps |
|------|---------:|----:|
| A) baseline (scipy, full-res, fp32) | ~50 | ~20 |
| B) + GPU Poisson | ~49 | ~20 (forward 가 지배적이라 단독효과 작음) |
| C) + torch.compile (fp16, full-res) | ~20 | ~50 |
| **OPT) + 저해상도(1/2)** | **~6.5** | **~155** |

→ **가장 큰 레버는 저해상도(#1)**, compile(#3)이 보조. GPU Poisson(#2)은 단독 속도 이득은 작지만 CPU 동기화 제거로 파이프라인 지연을 줄인다.

---

## 5. 결론

- **품질은 TouchNet, 속도는 (원래) DPT** 라는 trade-off 가 있었다.
- TouchNet 이 느린 근본 원인은 **풀해상도 + 큰 커널 conv + CPU Poisson** 이었고,
  이를 겨냥한 3가지 최적화로 **추론을 7.8배 가속**(20→155fps)하면서 **형상(corr 0.87)은 유지**했다.
- 그 결과 **최적화 TouchNet(155fps) 은 DPT(~86fps) 보다도 빠르면서** 품질 이점을 살릴 수 있어,
  실시간 촉각 깊이 추정에서 trade-off 를 상당 부분 해소했다.

### 향후 과제
- 저해상도로 바뀐 **절대 깊이 스케일(k) 재보정** (px_per_mm/scale 보정 또는 fine-tune).
- 더 큰 도약은 **다운샘플형 인코더-디코더(U-Net) 재학습** 또는 TensorRT/ONNX 익스포트.
- GT 기반 **정량 품질 평가**(현재 품질 비교는 정성 관찰).

---

## 6. 확장 분석 — NeuralFeels 의 ViT(DPT)를 TouchNet 으로 대체 가능한가?

### 6.1 NeuralFeels 에서 촉각 깊이 모델의 역할/인터페이스

NeuralFeels 파이프라인에서 촉각 ViT 는 **교체 가능한 깊이 추정 모듈**이다. 코드상 계약은 단 두 메서드다
(`neuralfeels/contrib/tactile_transformer/tactile_depth.py`, `neuralfeels/modules/sensor.py`):

```
DIGIT 이미지 ──▶ image2heightmap(image) ──▶ heightmap
                 heightmap2mask(heightmap) ──▶ contact mask
        depth = heightmap * mask
        depth = DepthTransform(cam_dist)(depth)   # px→m, +cam_dist(0.022m), 비접촉=NaN
        ──▶ 포인트클라우드(backproject) ──▶ Neural SDF / pose 최적화(SLAM)
```

즉 다운스트림(SLAM/neural field)은 **per-pixel heightmap + contact mask** 만 요구한다. ViT 인지 CNN 인지는 무관하다.
실제로 NeuralFeels 의 ViT 는 그들 데이터(tacto_ycb 400K, feelsight in-hand 84K 등)로 **학습/ablation** 되는 모듈이다
(`docs/note/vit-ablation-report.html`).

### 6.2 호환성 평가

TouchNet 도 "DIGIT 이미지 → per-pixel 깊이(heightmap)" 를 출력하므로 **인터페이스 수준에서는 대체 가능**하다.
`image2heightmap`/`heightmap2mask` 를 TouchNet 으로 감싼 `TactileDepth` 어댑터를 만들면 된다. 단, 다음을 맞춰야 한다.

| 항목 | DPT(현행) | TouchNet | 필요 작업 |
|------|-----------|----------|-----------|
| 출력 | 절대 heightmap [0..255], bg_template 차감 | **blank 차감된 접촉깊이**(Poisson 적분) | TouchNet 출력을 NeuralFeels 의 "heightmap−bg" 자리에 매핑 |
| 단위/스케일 | DepthTransform 의 px→m `scale` + cam_dist 로 보정 | px(상대) → `px_per_mm`(근사) | **메트릭 스케일 재보정**(§4 의 k≈0.34 모호성과 동일 이슈) |
| 부호 규약 | gel_depth 음수(접촉이 카메라 쪽으로) | 양수 깊이 | 부호 정렬 |
| 기준 프레임 | bg_template(첫 프레임/bg_id) | **blank 이미지 필수**(좌표 임베딩+blank 차감) | 센서별 blank 공급(NeuralFeels 가 보유) |
| 전처리 | Resize224+Normalize | **5채널(RGB+좌표 임베딩)** | 어댑터에서 좌표 임베딩 생성 |
| 학습 도메인 | tacto/feelsight(in-hand, 40+물체) | **캘리브레이션 프로빙(구 압입, 고정 셋업)** | **도메인 갭** → 재학습/파인튜닝 권장 |
| 미분가능/학습 | ViT 백본 학습됨 | nn.Module(학습 가능) | 동일 학습 루프에 편입 가능 |
| 속도 | ~86 fps | 최적화 후 **~155 fps** | §4 로 해소 (더 이상 병목 아님) |

### 6.3 핵심 쟁점

- **가능 (인터페이스)**: heightmap+mask 계약만 만족하면 되므로 어댑터로 끼울 수 있다.
- **진짜 관건은 정확도/도메인**: NeuralFeels ViT 는 in-hand manipulation 접촉·다물체로 학습되어 일반화가 핵심(ablation 문서의 주제).
  py3DCal TouchNet 은 **고정 셋업의 구 압입 캘리브레이션**으로 학습되어 접촉 패턴 분포가 다르다. 그대로 꽂으면 pose tracking 정확도는 보장되지 않으며,
  **NeuralFeels 도메인 데이터로 TouchNet 재학습/파인튜닝**해야 공정 비교가 된다.
- **메트릭 스케일**: §4 에서 본 절대 깊이 스케일 모호성(k)을 NeuralFeels 의 `DepthTransform`(px→m, cam_dist) 규약에 맞춰 보정해야 포인트클라우드가 올바른 m 단위가 된다.
- **속도는 더 이상 장애물 아님**: 최적화 TouchNet(155fps)이 DPT(86fps)보다 빠르므로, "TouchNet 은 품질 좋지만 느려서 SLAM 루프에 부담"이라는 단점이 사라진다.

### 6.4 결론 & 권장 절차

**대체 가능하다(인터페이스 호환).** 다만 단순 가중치 교체가 아니라 다음이 필요하다:
1. `TactileDepth` 어댑터 작성 — TouchNet + 좌표임베딩 + (GPU)Poisson 을 `image2heightmap`/`heightmap2mask` 시그니처로 노출.
2. **메트릭 스케일·부호 보정** — NeuralFeels `DepthTransform`(cam_dist, scale) 규약에 정렬.
3. **도메인 재학습/파인튜닝** — NeuralFeels 학습 데이터(tacto/feelsight)로 TouchNet 재학습 후 pose tracking ablation 으로 정량 평가.
4. 속도는 §4 최적화로 충분(오히려 이점).

> 요약: **"넣을 수는 있다(계약 호환). 잘 되게 하려면 스케일 보정 + 도메인 재학습이 필요하다."** 속도 단점은 이미 해소했으므로, 남는 일은 정확도(일반화) 검증이다.

---

## 7. 재현 방법

```bash
conda activate py3dcal

# TouchNet vs DPT(real+sim) 실시간 비교
python examples/stream_compare_touchnet_dpt.py --serial D21424
bash examples/benchmark_touchnet_dpt.sh D21424 10            # 속도 벤치마크

# TouchNet 최적화 전/후 실시간 비교
python examples/stream_compare_touchnet_opt.py --serial D21424
bash examples/benchmark_touchnet_opt.sh D21424 2 10          # 속도 벤치마크(1/2 해상도, 10초)

# GPU Poisson 정확성 + 단계별 속도
python examples/benchmark_touchnet_opts.py --serial D21424
```

관련 문서: [`README.md`](README.md) · [`ARCHITECTURE.md`](ARCHITECTURE.md) · [`ANALYSIS_KO.md`](ANALYSIS_KO.md)
