# py3DCal 소스코드 분석 (한글)

> 이 문서는 `py3dcal` 패키지의 구조와 동작 원리를 한글로 정리한 것입니다.
> 영어 요약 및 Claude Code 작업 가이드는 `CLAUDE.md`를 참고하세요.

## 1. 개요

`py3dcal`은 **촉각 센서(tactile sensor) 캘리브레이션**을 위한 PyPI 패키지입니다. 핵심 아이디어는 다음과 같습니다.

1. **3D 프린터**를 이용해 촉각 센서의 젤 표면을 정해진 좌표/깊이로 자동으로 눌러(probe) 데이터를 수집한다.
2. 수집한 데이터로 **신경망**을 학습시켜, 센서의 원시 출력(이미지 또는 자기장 값)을 **깊이맵(depth map)** 또는 **접촉 예측(contact prediction)** 으로 변환한다.

지원하는 센서는 두 계열입니다.

| 계열 | 센서 | 입력 | 모델 | 출력 |
|------|------|------|------|------|
| 비전 기반 | DIGIT, GelSight Mini | RGB 카메라 이미지 | `TouchNet` (CNN) | 표면 기울기 → 깊이맵 |
| 자기 기반 | ReSkin | 자력계(magnetometer) 값 | `MagNet` (MLP) | 3D 접촉 위치 |

> 참고: 테스트 스위트, 린터 설정, 별도 빌드 단계는 없습니다. 사용자용 상세 문서는 https://rohankotanu.github.io/3DCal/ 에 있습니다. 테스트 환경은 Python 3.10.4 기준입니다.

## 2. 전체 파이프라인

`examples/full_pipeline.py`가 표준 흐름을 보여줍니다. 패키지는 4단계로 구성되며, 각 단계는 `py3DCal/__init__.py`에서 최상위 함수로 재노출(re-export)됩니다.

```python
import py3DCal as p3d
from py3DCal import datasets, models

# 1단계: 데이터 수집
digit = p3d.DIGIT("D20966")
ender3 = p3d.Ender3("/dev/ttyUSB0")
calibrator = p3d.Calibrator(printer=ender3, sensor=digit)
calibrator.probe(calibration_file_path="misc/probe_points.csv")

# 2단계: 데이터 어노테이션 (비전 센서 전용)
p3d.annotate(dataset_path="./sensor_calibration_data", probe_radius_mm=2.0)

# 3단계: 모델 학습
my_dataset = datasets.TactileSensorDataset(root='./sensor_calibration_data')
touchnet = models.TouchNet()
p3d.train_model(model=touchnet, dataset=my_dataset, num_epochs=60, batch_size=64)

# 4단계: 추론 (깊이맵 생성)
depthmap = p3d.get_depthmap(model=touchnet,
                            image_path="path/to/target/image",
                            blank_image_path="./sensor_calibration_data/blank_images/blank.png")
```

### 1단계 — 데이터 수집 (`data_collection/Calibrator.py`)

- `Calibrator(printer, sensor)`를 만들고 `.probe(...)`(비전) 또는 `.probe_reskin(...)`(자기)을 호출합니다.
- 동작: 프린터를 홈 위치로 보낸 뒤, 캘리브레이션 점 CSV의 각 좌표로 이동 → Z축으로 젤에 눌러 넣음 → 센서 출력 기록.
- 이동 시간은 X/Y 10 mm/s, Z 4 mm/s 가정으로 계산해 `time.sleep()`으로 대기합니다(동기/개루프 방식, 피드백 없음).
- 사람이 프로브를 탈/부착하도록 `input()`으로 대기하는 **대화형(interactive)** 절차입니다 → CI에서 무인 실행 불가.

### 2단계 — 어노테이션 (`model_training/lib/annotate_dataset.py`)

- `annotate(dataset_path, probe_radius_mm)`: Matplotlib 기반 대화형 GUI.
- 두 장의 프로브 이미지에 원을 맞춰 **px_per_mm(픽셀당 mm)** 비율을 계산합니다.
- 키 조작: `w/a/s/d` 이동, `r/f` 크기 조절, `q` 다음 단계, `1/2/3` 뷰 전환(원본 / 차영상 / bitwise-not).
- 결과물: `annotations/annotations.csv`, `annotations/metadata.json`.
- **비전 파이프라인 전용** (ReSkin은 어노테이션 단계 없음).

### 3단계 — 학습 (`model_training/lib/train_model.py`)

- `train_model(model, dataset, num_epochs=60, batch_size=64, lr=1e-4, ...)`.
- AdamW 옵티마이저, 기본 손실함수 `MSELoss`, 기본 디바이스 `cpu`.
- `_validate_model_and_dataset`가 모델↔데이터셋 짝을 강제: `TouchNet`↔`TactileSensorDataset`, `MagNet`↔`ReSkinDataset`.
- 출력물 `weights.pth`, `losses.csv`는 **현재 작업 디렉토리(CWD)** 에 저장됩니다(데이터셋 폴더가 아님).

### 4단계 — 추론 (`model_training/lib/depthmaps.py`, `reskin_prediction.py`)

- 비전: `get_depthmap` / `save_2d_depthmap` / `show_2d_depthmap`.
- 자기: `get_reskin_contact`.

## 3. 핵심 아키텍처

### 3.1 하드웨어 드라이버 — 추상 베이스 클래스(ABC)

`Printer`(`printers/Printer.py`)와 `Sensor`(`sensors/Sensor.py`)는 ABC입니다. 새 하드웨어를 추가하려면 상속해서 구현합니다.

- **Printer**: `connect`, `disconnect`, `send_gcode`, `get_response`, `initialize` 구현 필수. 베이스가 `go_to(x, y, z)` 제공.
  - 유일한 구현체 `Ender3`: `pyserial`로 115200 baud G-code 통신. `initialize()`는 홈 명령 후 `ok` 응답 4번을 기다려 완료 판단.
- **Sensor**: `connect`, `disconnect`, `capture_image` 구현 필수. 베이스가 `flush_frames` 제공.
  - 각 센서는 캘리브레이션 기하 정보를 인스턴스 속성으로 설정: `x_offset`, `y_offset`, `z_offset`(센서 표면 높이), `z_clearance`, `max_penetration`, `default_calibration_file`.
  - **`Calibrator`가 이 속성들을 읽어 프린터 이동을 계획**하므로, 값이 틀리면 프로브가 젤을 너무 세게 누르거나 빗나갑니다.

`capture_image()` 반환값은 센서 계열마다 다릅니다.
- 비전 센서: RGB `numpy` 이미지 (DIGIT은 `cv2.flip`으로 좌우 반전).
- ReSkin: 자력계 채널 값들의 평탄한 리스트 (`Bx/By/Bz/T` × 5 = 20개, 시리얼에서 탭 구분으로 읽음).

하드웨어 라이브러리가 없는 머신에서도 import가 되도록 `from digit_interface import Digit`를 `try/except`로 감싸는 **import 가드** 패턴을 사용합니다.

### 3.2 캘리브레이션 데이터 포맷

`probe()`가 생성하는 디렉토리 구조 (`<data_save_path>/sensor_calibration_data/`):

```
annotations/probe_data.csv     # img_name, x_mm, y_mm, penetration_depth_mm
blank_images/blank.png         # 접촉 없는 기준 이미지
probe_images/                  # 프로브 이미지, 파일명 <idx>_X<x>Y<y>Z<z>.png
```

캘리브레이션 점 CSV(각 센서의 `default.csv` 또는 사용자가 넘긴 `calibration_file_path`)의 컬럼: `x_mm, y_mm, penetration_depth_mm, num_images` (헤더 1줄은 건너뜀). 깊이가 센서의 `max_penetration`을 넘는 점은 프로빙 시 건너뜁니다.

`probe_reskin()`은 이미지 대신 단일 평탄 `probe_data.csv`(+ `no_contact_data.csv`)에 자력계 채널을 그대로 기록합니다 — 별도 어노테이션 단계 없음.

### 3.3 비전 모델 입력은 5채널 텐서 (`models/touchnet.py`)

`TouchNet`은 9층 완전 합성곱(fully-convolutional) CNN입니다.
- **입력 5채널**, **출력 2채널**(x/y 표면 기울기).
- 5채널 = RGB 3채널 + **좌표 임베딩 2채널**. `add_coordinate_embeddings`가 픽셀별 열(column)/행(row) 인덱스를 채널로 덧붙입니다.
- 이 좌표 임베딩은 학습(`TactileSensorDataset`)과 추론(`get_depthmap`) 양쪽에서 동일하게 적용됩니다. **TouchNet에 입력을 넣는 새 코드 경로는 반드시 이 임베딩을 재현해야** 채널 수(5)가 맞습니다.

`get_depthmap` 흐름:
1. 입력 이미지에서 blank 이미지를 뺀다(차영상).
2. 좌표 임베딩을 추가한다.
3. 모델로 기울기맵(gradient map)을 얻는다.
4. `fast_poisson`(Poisson 표면 재구성, `lib/fast_poisson.py`)으로 기울기를 적분해 깊이맵을 만든다.

**사전학습 가중치**: `TouchNet(load_pretrained=True, sensor_type=...)`로 켜면 Zenodo에서 가중치 파일을 다운로드합니다. `SensorType`(DIGIT / GELSIGHTMINI)이 파일을 선택하며, `root`에 캐시되어 있으면 재다운로드를 건너뜁니다.

### 3.4 자기 모델 (`models/magnet.py`)

`MagNet`은 MLP입니다.
- 입력 15차원(자력계 특성) → `fc1(15→200) → fc2 → fc3(→40) → fc4 → fc5 → fc6(→3)`.
- 출력 3차원(3D 접촉 위치).
- 마찬가지로 `load_pretrained=True` 시 Zenodo에서 `reskin_pretrained_weights.pth` 다운로드.

### 3.5 데이터셋과 분할 (`model_training/datasets/`)

- `TactileSensorDataset`(비전 공통, DIGIT/GelSightMini 데이터셋의 베이스)와 `ReSkinDataset`이 각각 존재.
- `TactileSensorDataset.__getitem__`는 `(좌표임베딩 포함 이미지, 기울기맵 타깃)`을 반환. 타깃 기울기맵은 어노테이션된 접촉 원으로부터 `precompute_gradients` / `get_gradient_map`이 생성.
- 기본값 `subtract_blank=True`, `add_coordinate_embeddings=True`.
- `split_dataset`(`split_dataset.py`)은 좌표 단위로 train/val을 나눕니다. **같은 (x_mm, y_mm)의 여러 이미지가 train/val에 섞이지 않도록** 고유 좌표를 먼저 분리한 뒤 다시 merge → 데이터 누수(leakage) 방지. `random_state=42`로 재현성 보장.

### 3.6 입력 검증 규약 (`model_training/lib/validate_parameters.py`)

- `validate_device`, `validate_root`, `validate_dataset`로 인자 검증을 중앙화.
- 각 공개 함수 시작부에서 호출 → 호출 스택 깊은 곳에서 실패하지 않고 앞단에서 설명 메시지와 함께 `ValueError`/`TypeError`를 던지는 패턴.

## 4. 주의할 점(Gotchas)

- 일부 모듈에 불필요한 `from pyexpat import model` import가 남아 있음(`depthmaps.py`, `reskin_prediction.py`). 사용되지 않으니 의존하거나 전파하지 말 것.
- `Calibrator`의 수집 흐름은 `input()`/stdout 출력에 의존하는 대화형 절차. 하드웨어가 연결된 터미널 환경을 전제하며 무인 실행 불가.
- 어노테이션/시각화 GUI는 대화형 Matplotlib 백엔드(디스플레이)가 필요.
- `default_calibration_file` 경로는 각 센서 모듈 디렉토리 기준으로 `os.path.dirname(os.path.abspath(__file__))`로 해석됨. CSV들은 `MANIFEST.in`의 `recursive-include py3DCal/data_collection *.csv`로 패키지에 포함됨.
- `Calibrator.disconnect_sensor()`에는 버그로 보이는 부분이 있음(`self.sensor.connect()` 호출 후 `self.printer.disconnect()`를 호출). 센서 연결 해제 로직이 프린터를 대상으로 함.

## 5. 콘솔 진입점

`setup.py`의 유일한 콘솔 스크립트: `list-com-ports` → `py3DCal.list_com_ports` (`utils/utils.py`). 연결된 프린터의 시리얼 포트를 탐색해 출력합니다(플랫폼별 포트 후보를 열어보고 성공한 것만 반환).
