# Installing the package
```python
pip3 install py3dcal
```

<br>

# For instructions on how to use this package, please visit https://rohankotanu.github.io/3DCal/

<br>

**Note:** Project was tested with Python 3.10.4

<br>

---

# 개발/실행 환경 셋업 (conda)

이 저장소를 직접 클론해서 개발·실행하기 위한 환경 구성 방법입니다. (PyPI 설치만 필요하면 위 `pip3 install py3dcal` 로 충분합니다.)

## 사전 준비
- [conda](https://docs.conda.io/) (miniforge/miniconda)
- (선택) NVIDIA GPU + 드라이버 — 실시간 추론/스트리밍에 권장. 없으면 CPU 로도 동작합니다.

## 방법 1) environment.yml 로 한 번에 생성 (권장)
저장소 루트에 포함된 `environment.yml` 을 사용합니다.

```bash
conda env create -f environment.yml
conda activate py3dcal
```

- `torch` / `torchvision` 은 PyTorch 의 **CUDA 12.8 휠**(`--extra-index-url`)로 설치됩니다.
- **GPU 가 없으면** `environment.yml` 의 extra-index-url 을
  `https://download.pytorch.org/whl/cpu` 로 바꾼 뒤 생성하세요.
- 마지막 줄의 `-e .` 가 이 저장소를 editable 모드로 설치합니다.

## 방법 2) 수동 설치
```bash
conda create -n py3dcal python=3.10 -y
conda activate py3dcal

# CUDA(GPU) 빌드 — torchvision>=0.23 요구사항 충족을 위해 cu128 사용
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# (CPU 전용이면 위 대신)
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 저장소를 editable 로 설치 (나머지 의존성도 함께 설치됨)
pip install -e .
```

## 설치 확인
```bash
python -c "import torch, py3DCal; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

# 연결된 3D 프린터 시리얼 포트 탐색 (콘솔 진입점)
list-com-ports
```

<br>

---

# 예제 스크립트 (DIGIT 센서)

`examples/` 디렉토리에 DIGIT 센서로 사전학습 모델을 확인/시각화하는 스크립트가 있습니다.
처음 실행 시 사전학습 가중치(`digit_pretrained_weights.pth`)를 Zenodo 에서 자동으로 받아 `--root`(기본 `./digit_weights`)에 캐시합니다.

## 1) 여러 DIGIT 센서 깊이맵 일괄 확인 — `examples/check_digit_sensors.py`
연결된 DIGIT 센서들을 자동 탐지해, 무접촉(blank)·접촉(contact) 프레임을 캡처하고
사전학습 TouchNet 으로 센서별 깊이맵을 계산해 한 그림으로 저장/표시합니다.

```bash
conda activate py3dcal

# 연결된 모든 DIGIT 자동 탐지 (대화형: Enter 로 단계 진행)
python examples/check_digit_sensors.py

# 시리얼 직접 지정
python examples/check_digit_sensors.py --serials D21422 D21429 D21424 D21418

# 입력 없이 자동 진행(blank 후 --delay 초 뒤 contact 캡처)
python examples/check_digit_sensors.py --auto --delay 5
```

주요 옵션: `--serials`, `--auto`, `--delay`, `--flush`(버퍼 드레인 프레임 수, 접촉이 안 잡히면 늘리기),
`--device`, `--root`, `--outdir`, `--no-show`(헤드리스 저장).

## 2) 실시간 깊이 스트리밍 시각화 — `examples/stream_digit_depth_3d.py`
DIGIT 한 대를 실시간으로 읽어 매 프레임 깊이를 계산하고, **DIGIT 카메라 이미지 + 2D 깊이 + 3D 깊이**를
같은 프레임으로 동기 스트리밍합니다.

```bash
conda activate py3dcal

# 기본: 카메라 + 2D + 3D 3패널 (첫 번째 센서 자동)
python examples/stream_digit_depth_3d.py --serial D21424

# 모드 선택 (카메라 이미지는 모든 모드에 항상 포함)
python examples/stream_digit_depth_3d.py --serial D21424 --mode 2d   # 카메라 + 2D 깊이
python examples/stream_digit_depth_3d.py --serial D21424 --mode 3d   # 카메라 + 3D 깊이

# 창 없이 깊이 추정값만 콘솔에 출력 (최대 fps 측정 / 실시간 변화 확인)
python examples/stream_digit_depth_3d.py --serial D21424 --no-render
```

- 조작: `q`/`ESC` 종료, `r` blank(기준) 재캡처
- **깊이 단위(mm)**: `--px-per-mm` 로 px→mm 변환 계수를 지정합니다. 기본 15.0 은 DIGIT 근사값이며,
  정확한 값은 캘리브레이션 `annotations/metadata.json` 의 `px_per_mm` 를 사용하세요. `--px-per-mm 0` 이면 상대단위로 표시합니다.
- 성능: 백그라운드 캡처 스레드(지연 최소화) + FP16 가속(`--no-fp16` 로 끄기)이 기본 적용됩니다.
- 기타 옵션: `--downsample`(3D 표면 해상도), `--zmax`(깊이축 상한 고정), `--target-fps`, `--headless N`(디스플레이 없이 N프레임 처리 후 스냅샷 저장).

> 위 두 스크립트는 실제 DIGIT 하드웨어가 연결되어 있어야 하며, 시각화는 디스플레이가 필요합니다(헤드리스는 `--no-show` / `--headless` 사용).

## 3) TouchNet(CNN) vs DPT(ViT) 비교 — `examples/stream_compare_touchnet_dpt.py`
py3DCal TouchNet 과 NeuralFeels 의 tactile transformer(DPT, `dpt_real.p` / `dpt_sim.p`)를 같은 DIGIT 입력으로 실시간 비교합니다.
DPT 는 `timm` + 로컬 `examples/dpt_lite/`(neuralfeels 에서 복사한 자체 완결 모델 코드)로 **가볍게** 구동되며, neuralfeels 전체 설치는 필요 없습니다.

```bash
conda activate py3dcal
pip install timm einops   # DPT 비교용 추가 의존성(최초 1회)

WDIR=/home/avery/Documents/neuralfeels/deploy/weights/tactile_transformer

# TouchNet vs dpt_real vs dpt_sim 동시 비교 (2D 히트맵). --dpt-weights 에 여러 .p 지정 가능.
python examples/stream_compare_touchnet_dpt.py --serial D21424 \
    --dpt-weights $WDIR/dpt_real.p $WDIR/dpt_sim.p

# 3D 표면 비교
python examples/stream_compare_touchnet_dpt.py --serial D21424 --mode 3d \
    --dpt-weights $WDIR/dpt_real.p $WDIR/dpt_sim.p
```
- 패널: `[카메라 | TouchNet(mm) | DPT#1(rel) | DPT#2(rel) ...]`, 모두 같은 프레임으로 동기. `q`/ESC 종료, `r` blank 재캡처.
- 공정 비교: 둘 다 blank 기준 상대 접촉깊이(TouchNet=입력에서 blank 차감 / DPT=heightmap−blank_heightmap).
- TouchNet 은 mm, DPT 는 상대값(0–255 Δheightmap)이라 각자 컬러스케일로 표시됩니다.

### 추론 속도 자동 벤치마크 — `examples/benchmark_touchnet_dpt.sh`
`--no-render` 로 약 N초간 모델별 추론시간/fps 를 측정해 비교 요약을 출력합니다.
```bash
bash examples/benchmark_touchnet_dpt.sh D21424 10
# real+sim 둘 다 측정(추가 인자는 python 으로 그대로 전달)
bash examples/benchmark_touchnet_dpt.sh D21424 10 --dpt-weights $WDIR/dpt_real.p $WDIR/dpt_sim.p
```
참고(RTX 4070 Laptop, FP16): DPT(ViT-small@224) 가 TouchNet(CNN, 320×240 전 해상도)보다 추론이 수 배 빠릅니다.

### 왜 이런 결과가 나오는가 — 속도 분석

실측 예 (RTX 4070 Laptop, FP16, `--no-render --duration 10`):
```
=== 추론 속도 비교 요약 ===
측정 프레임 수: 136  (경과 10.1s)
TouchNet(CNN)   :  49.37 ms/frame  ->   20.3 fps
dpt_real(ViT)   :  11.60 ms/frame  ->   86.2 fps
dpt_sim(ViT)    :  12.54 ms/frame  ->   79.7 fps
=> dpt_real(ViT) 가 가장 빠름 (가장 느린 TouchNet(CNN) 대비 4.25배)
```

파라미터 수는 TouchNet(약 3.9M) < DPT(약 25.9M) 인데도 **DPT 가 더 빠릅니다.** 직관과 반대인 이유는 다음과 같습니다.

1. **파라미터 수 ≠ 연산량(FLOPs)/지연시간.**
   TouchNet 은 파라미터는 적지만 그 가중치를 **320×240 전 해상도**에서 매 레이어 반복 적용하므로 FLOPs 가 큽니다.
   ViT 는 파라미터가 많아도 196개 토큰에 대한 행렬곱(GEMM)에 모여 있어 실제 연산량과 지연이 작습니다.

2. **공간 해상도 차이가 결정적.**
   TouchNet 은 9개 conv 레이어 내내 다운샘플 없이 320×240(=76,800 픽셀)을 유지합니다. conv FLOPs 는 H×W 에 비례하므로 매 레이어가 풀해상도 비용을 냅니다.
   반면 DPT 는 입력을 **224×224 로 줄이고 patch16 으로 14×14=196 토큰**(≈8.6배 다운샘플)으로 바로 압축합니다.

3. **큰 커널 conv vs 효율적 GEMM + 텐서코어.**
   TouchNet 은 7×7·5×5 같은 큰 커널(비용 ∝ k²)을 256채널 풀해상도 특징맵에 적용합니다. 큰 conv 는 노트북 GPU 에서 활용률이 낮습니다.
   ViT 는 거의 전부 큰 행렬곱이라 **FP16 텐서코어**에 매우 잘 매핑되어 GPU 피크 활용에 가깝습니다.

4. **메모리 대역폭.**
   TouchNet 의 256채널×320×240 활성값은 매우 커서 대역폭에 묶입니다. ViT 의 활성값(196×384)은 작습니다.

5. **CPU 후처리 꼬리.**
   TouchNet 파이프라인은 GPU 추론 뒤 **CPU 에서 Poisson 적분**(`scipy.fftpack` DST/IDST, 320×240)과 GPU→CPU 동기화가 매 프레임 들어갑니다. DPT 는 heightmap 을 바로 출력해 이 비용이 없습니다(대신 PIL 리사이즈 정도의 가벼운 후처리만).

**측정 해석상 주의:**
- 모델별 시간은 각 호출이 `.cpu()`/PIL 에서 GPU 동기화되는 지점까지의 wall-clock 이며, 세 모델을 한 루프에서 순차 실행하므로 GPU 스케줄링 간섭으로 절대값은 실행마다 변동합니다(TouchNet 40~67ms 등). **상대 순서는 안정적**입니다.
- `dpt_real` 과 `dpt_sim` 은 **아키텍처가 동일**(vit_small_patch16_224)하여 속도 차이는 측정 노이즈 수준입니다. 둘의 의미 있는 차이는 속도가 아니라 **추정 품질**(real=실측 학습, sim=시뮬레이션 학습)에 있습니다.

> 요약: "ViT 라서 무겁다"는 통념과 달리, 여기서는 **TouchNet 이 풀해상도·큰 커널 conv + CPU Poisson** 때문에 느리고, **DPT 는 저해상도 토큰 + GEMM/텐서코어** 덕에 빠릅니다. 속도는 DPT 우위, 품질·단위(mm) 직접성은 용도에 따라 trade-off 입니다.

<br>

---

# 전체 파이프라인 개요

py3DCal 은 3D 프린터로 촉각 센서를 자동 프로빙해 데이터를 모으고, 신경망을 학습시켜 센서 출력을 깊이맵/접촉 예측으로 변환합니다.

1. **데이터 수집** — `Calibrator(printer, sensor).probe(...)` (DIGIT/GelSight) 또는 `.probe_reskin(...)` (ReSkin)
2. **어노테이션** — `annotate(dataset_path, probe_radius_mm)` (비전 센서, 대화형 GUI)
3. **학습** — `train_model(model, dataset, ...)` → `weights.pth`, `losses.csv`
4. **추론** — `get_depthmap` / `save_2d_depthmap` / `show_2d_depthmap` (비전), `get_reskin_contact` (자기)

자세한 코드 구조 분석은 한글 문서 [`ANALYSIS_KO.md`](ANALYSIS_KO.md) 를 참고하세요.
