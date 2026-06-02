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

<br>

---

# 전체 파이프라인 개요

py3DCal 은 3D 프린터로 촉각 센서를 자동 프로빙해 데이터를 모으고, 신경망을 학습시켜 센서 출력을 깊이맵/접촉 예측으로 변환합니다.

1. **데이터 수집** — `Calibrator(printer, sensor).probe(...)` (DIGIT/GelSight) 또는 `.probe_reskin(...)` (ReSkin)
2. **어노테이션** — `annotate(dataset_path, probe_radius_mm)` (비전 센서, 대화형 GUI)
3. **학습** — `train_model(model, dataset, ...)` → `weights.pth`, `losses.csv`
4. **추론** — `get_depthmap` / `save_2d_depthmap` / `show_2d_depthmap` (비전), `get_reskin_contact` (자기)

자세한 코드 구조 분석은 한글 문서 [`ANALYSIS_KO.md`](ANALYSIS_KO.md) 를 참고하세요.
