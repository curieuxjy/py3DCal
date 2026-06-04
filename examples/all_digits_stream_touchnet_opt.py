"""
연결된 DIGIT 센서 여러 개(기본 4개)를 '동시에' 실시간 스트리밍하면서, 각 센서의
깊이맵을 TouchNet OPT 경로로 추정해 3D 표면으로 함께 그리는 스크립트.

OPT 경로(= stream_compare_touchnet_opt.py 의 OPT 와 동일):
  #1 저해상도 추론(--infer-scale) + #2 GPU Poisson(torch matmul) + #3 torch.compile
  + FP16/channels_last. 저해상도에서도 좌표 임베딩은 원본 픽셀 범위(1..W,1..H)로 생성해
  학습 분포와 정합. 모델 가중치는 모든 센서가 공유(동일 DIGIT 사전학습 모델).

레이아웃: 센서당 [원본 카메라 | 깊이] 2패널. 2개면 1x2, 3~4개면 2x2 블록으로 배치.
모든 센서를 같은 루프에서 동기 갱신하고, 합산 fps 를 로그로 출력.

지연 최적화(여러 센서 동시 입력):
  - 캡처는 센서별 백그라운드 스레드(FrameGrabber)로 분리 — 카메라 I/O 가 추론과 겹친다.

프레임 섞임 / 무접촉 노이즈 방지 (실측으로 확정한 권장 설정):
  - 한 프레임 안에 다른 센서 영상이 일부 섞이는 '부분 섞임'은 동일 USB 디스크립터의
    DIGIT 들을 한 USB 컨트롤러에서 동시 스트리밍할 때 uvcvideo 가 isochronous 패킷을
    잘못 라우팅해 생긴다(드라이버 레벨, read 시점과 무관). 핵심은 대역폭을 줄이는 것.
  - 실측 결과 4대 동시에는 'YUYV(무압축) + --cam-fps 15'(기본값)가 섞임/노이즈 모두 0.
    MJPEG 은 대역폭은 줄지만 압축 잡음이 phantom depth(무접촉인데 깊이가 뜸)를 만들어
    기본 OFF. (즉 노이즈 원인은 대개 MJPEG 압축이었음.)
  - 센서가 1~2대면 --cam-fps 30~60 으로 올려도 됨. 근본 해결은 센서를 '서로 다른 USB
    호스트 컨트롤러'에 분산하는 것(그러면 fps 안 낮춰도 됨).
  - serial->dev_name 매핑은 연결 시 출력하고 중복(오매핑)이면 즉시 중단.
  - cv2 버퍼=1 + 매 grab drain 으로 항상 최신 프레임만 사용(지연/잔상 제거).
  - 추가 노이즈 억제: --diff-thresh(frame-blank dead-band, fast_poisson 적분 증폭 원천 차단),
    --blank-frames(깨끗한 기준), --depth-floor(출력 바닥값), --depth-ema(시간축 평활).

추론/지연:
  - 추론은 모든 센서를 '한 배치(N장)'로 묶어 단일 forward + 단일 GPU->CPU 동기화로 처리.
    (센서별 순차 forward + 센서별 .cpu() 가 가장 큰 지연원이라 제거.)
  - CPU->GPU 전송은 pinned memory + non_blocking 으로 비동기화.

모드:
  (기본)       실시간 3D 렌더(라이브). q/ESC 종료, r 모든 센서 blank 재캡처.
  --mode 2d    3D 표면 대신 2D 깊이 이미지로.
  --no-render  창 없이 센서별 fps/최대깊이만 출력(Ctrl-C 또는 --duration 종료).
  --headless N 디스플레이 없이 N프레임 후 스냅샷 저장.

실행:
  conda activate py3dcal
  python examples/all_digits_stream_touchnet_opt.py
  python examples/all_digits_stream_touchnet_opt.py --serials D20753 D20765 D21424 D20966
  python examples/all_digits_stream_touchnet_opt.py --mode 2d
  python examples/all_digits_stream_touchnet_opt.py --no-render --duration 10
"""
import argparse
import math
import os
import sys
import threading
import time

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib

import py3DCal as p3d
from py3DCal import DIGIT, models, SensorType
from py3DCal.model_training.lib.add_coordinate_embeddings import add_coordinate_embeddings
from py3DCal.model_training.lib.fast_poisson import fast_poisson

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_touchnet_opts import GpuPoisson

try:
    from digit_interface import DigitHandler
except Exception:
    DigitHandler = None


def detect_serials():
    """연결된 DIGIT 센서의 시리얼 번호 목록을 중복 없이 반환."""
    if DigitHandler is None:
        return []
    serials = []
    for d in DigitHandler.list_digits():
        s = d.get("serial") if isinstance(d, dict) else None
        if s and s not in serials:
            serials.append(s)
    return serials


def _fourcc_str(cap):
    v = int(cap.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((v >> 8 * i) & 0xFF) for i in range(4))


def _configure_camera(sensor, fps, mjpeg, width=320, height=240):
    """카메라 대역폭(=USB isochronous 예약)을 줄여 다중 DIGIT 동시 스트리밍 시
    프레임 부분 섞임(uvcvideo 패킷 오라우팅)을 완화한다.

    - mjpeg=True : 픽셀 포맷을 MJPEG 로(YUYV 대비 대역폭 대폭 ↓). 가장 효과 큼.
    - fps        : 낮출수록 대역폭 ↓.
    - buffersize=1 : 항상 최신 프레임.

    반환: 적용된 {fourcc, fps} (디버그 출력용). cap 이 없으면 None.
    """
    cap = getattr(sensor, "_Digit__dev", None)  # digit_interface 내부 cv2.VideoCapture
    if cap is None:
        return None
    if mjpeg:
        # FOURCC 는 보통 해상도 설정 전에 지정해야 적용된다.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return {"fourcc": _fourcc_str(cap), "fps": cap.get(cv2.CAP_PROP_FPS)}


def _camera_ok(sensor, n=5):
    """포맷 변경 후 실제로 프레임이 정상적으로 들어오는지 확인."""
    try:
        for _ in range(n):
            f = sensor.capture_image()
            if f is None or f.size == 0:
                return False
        return True
    except Exception:
        return False


# DIGIT QVGA(320x240) 스트림이 디스크립터로 실제 지원하는 프레임 간격(fps).
# 15 같은 값은 '지원되지 않아' uvcvideo 가 임의로 30/60 으로 반올림한다 → 대역폭이
# 비결정적으로 60 까지 튀어 다중 동시 스트리밍에서 부분 섞임이 재발한다.
QVGA_SUPPORTED_FPS = (30, 60)


def _snap_cam_fps(requested):
    """요청 fps 를 QVGA 가 실제 지원하는 값으로 스냅. (지원 외 값은 드라이버가 무시함)"""
    if requested in QVGA_SUPPORTED_FPS:
        return requested, False
    # 요청값 이하의 가장 큰 지원값(대역폭을 줄이는 방향). 없으면 최소 지원값.
    below = [v for v in QVGA_SUPPORTED_FPS if v <= requested]
    snapped = max(below) if below else min(QVGA_SUPPORTED_FPS)
    return snapped, True


def _measure_fps(sensor, n=30):
    """포맷 적용 후 카메라가 실제로 내보내는 fps 를 측정(요청대로 반영됐는지 검증)."""
    t = time.time()
    k = 0
    for _ in range(n):
        f = sensor.capture_image()
        if f is not None and f.size:
            k += 1
    dt = time.time() - t
    return (k / dt) if dt > 0 else 0.0


def _force_qvga_fps(sensor_digit, dev_name, fps, width=320, height=240):
    """프레임 간격을 '실제로' fps 로 고정한다(다중 동시 스트리밍 대역폭 축소의 핵심).

    이 DIGIT uvcvideo 드라이버의 두 가지 함정:
      (1) OpenCV 의 CAP_PROP_FPS 설정을 '무시'한다(=cap.set 으로는 못 바꿈).
      (2) S_FMT(=set_resolution)가 프레임 간격을 포맷 기본값(60)으로 '리셋'한다.
    따라서 간격을 유지하는 유일한 방법은: 장치를 '닫은 상태'에서 v4l2-ctl 로 포맷+간격을
    함께 박고, 그 뒤 S_FMT 를 일으키지 않는 plain open 으로 다시 여는 것(프로브로 확인).

    digit 이 연 내부 cv2.VideoCapture(_Digit__dev)를 닫고 새 핸들로 교체한다.
    성공 여부(bool)와 get-parm readback 문자열을 반환.
    """
    import shutil
    import subprocess
    cap = getattr(sensor_digit, "_Digit__dev", None)
    if cap is None or not dev_name or shutil.which("v4l2-ctl") is None:
        return False, "재오픈 불가(cap/dev/v4l2-ctl 없음)"
    try:
        cap.release()
    except Exception:
        pass
    time.sleep(0.3)  # 닫힘이 커널에 반영될 시간(너무 빠른 재오픈은 select timeout 유발)
    try:
        subprocess.run(["v4l2-ctl", "-d", dev_name,
                        f"--set-fmt-video=width={width},height={height},pixelformat=YUYV",
                        f"--set-parm={fps}"],
                       capture_output=True, text=True, timeout=5)
        out = subprocess.run(["v4l2-ctl", "-d", dev_name, "--get-parm"],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception as e:
        return False, str(e)
    time.sleep(0.1)
    newcap = cv2.VideoCapture(dev_name, cv2.CAP_V4L2)
    if not newcap.isOpened():
        return False, "재오픈 실패"
    # plain open 은 VGA(640x480)로 열린다 → QVGA 를 강제한다. 장치 간격이 위에서 30 으로
    # 박혀 있으므로, 이 S_FMT 는 (60 으로 리셋하지 않고) 30 을 유지한다(프로브 [B] 확인).
    newcap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    newcap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    try:
        newcap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    sensor_digit._Digit__dev = newcap
    msg = "?"
    for line in out.splitlines():
        if "Frames per second" in line:
            msg = line.split(":", 1)[1].strip()
    return True, msg


class FrameGrabber(threading.Thread):
    """센서별 백그라운드 캡처 스레드(최신 프레임만 유지).

    각 센서는 자기 cv2.VideoCapture(독립 /dev/videoN)를 가진다. 매 grab 시 drain 으로
    그 카메라의 최신 프레임만 보관한다(지연/잔상 제거). 동시 스트리밍 시 발생하는
    프레임 '부분 섞임'(uvcvideo isochronous 패킷 오라우팅)은 여기서가 아니라 연결 단계의
    대역폭 축소(MJPEG/fps, _configure_camera)로 완화한다.
    """
    def __init__(self, sensor, serial, drain=2):
        super().__init__(daemon=True)
        self.sensor = sensor
        self.serial = serial
        self._drain = drain
        self._lock = threading.Lock()
        self._frame = None
        self._running = True
        self._err = None

    def run(self):
        while self._running:
            try:
                # 자기 카메라의 잔여 버퍼를 비우고 가장 최신 프레임만 보관한다.
                f = self.sensor.capture_image()
                for _ in range(self._drain):
                    f = self.sensor.capture_image()
            except Exception as e:
                self._err = e
                break
            with self._lock:
                self._frame = f

    def latest(self):
        with self._lock:
            return None if self._frame is None else self._frame

    def stop(self):
        self._running = False


def capture_blank(grabber, n_avg=10, settle=0.6):
    while grabber.latest() is None:
        time.sleep(0.01)
    acc = None
    for _ in range(n_avg):
        f = grabber.latest().astype(np.float32)
        acc = f if acc is None else acc + f
        time.sleep(settle / n_avg)
    return (acc / n_avg).astype(np.uint8)


def coord_embed_scaled(img_low, full_H, full_W):
    """저해상도 이미지에 좌표 임베딩 추가 — 값 범위는 원본(1..W,1..H) 유지(학습 분포 정합)."""
    _, h, w = img_low.shape
    x = torch.linspace(1, full_W, w).unsqueeze(0).repeat(h, 1).unsqueeze(0)
    y = torch.linspace(1, full_H, h).unsqueeze(1).repeat(1, w).unsqueeze(0)
    return torch.cat([img_low, x, y], dim=0)


def downsample(arr, factor):
    return arr if factor <= 1 else arr[::factor, ::factor]


def grid_shape(n):
    """센서 개수에 맞는 (행, 열) 격자. 1->1x1, 2->1x2, 3~4->2x2, 그 이상도 근사 정사각."""
    if n <= 1:
        return 1, 1
    if n == 2:
        return 1, 2
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def main():
    ap = argparse.ArgumentParser(description="여러 DIGIT 센서 동시 깊이 3D 스트리밍 (TouchNet OPT)")
    ap.add_argument("--serials", nargs="*", default=None, help="DIGIT 시리얼들. 생략 시 자동 탐지.")
    ap.add_argument("--root", default="./digit_weights", help="사전학습 가중치 저장/로드 디렉토리.")
    ap.add_argument("--infer-scale", type=int, default=2, help="저해상도 배수(2=절반).")
    ap.add_argument("--px-per-mm", type=float, default=15.0, help="px->mm 변환. 0이면 상대단위.")
    ap.add_argument("--warmup", type=int, default=50, help="센서 연결 후 버릴 프레임 수.")
    # 실측 결과: 4대 동시에는 YUYV(무압축) + 저fps 가 섞임/노이즈 모두 가장 깨끗했다.
    # MJPEG 은 대역폭은 줄지만 압축 잡음이 phantom depth 를 만들어 기본 OFF 로 둔다.
    ap.add_argument("--mjpeg", dest="mjpeg", action="store_true", default=False,
                    help="카메라 포맷을 MJPEG 로(기본 OFF). 대역폭은 줄지만 압축 잡음으로 무접촉 "
                         "노이즈가 늘 수 있음. 컨트롤러 대역폭이 부족하고 fps 도 못 낮출 때만.")
    ap.add_argument("--no-mjpeg", dest="mjpeg", action="store_false",
                    help="MJPEG 비활성(기본). 무압축 YUYV 사용 → 압축 잡음 없음.")
    ap.add_argument("--cam-fps", type=int, default=30,
                    help="카메라 캡처 FPS. DIGIT QVGA 는 30/60 만 지원하므로 그 외 값은 자동으로 "
                         "스냅된다(15 등 비지원 값은 드라이버가 임의로 30/60 으로 반올림 → 대역폭이 "
                         "비결정적으로 튀어 섞임 재발). 다중 동시면 30 권장, 1~2대면 60 가능.")
    ap.add_argument("--diff-thresh", type=float, default=0.04,
                    help="입력 dead-band(0~1). |frame-blank| 가 이보다 작으면 0 처리 → 무접촉 노이즈 "
                         "가 Poisson 적분으로 깊이가 되는 것을 차단. 0이면 비활성. (≈0.04=10/255)")
    ap.add_argument("--depth-floor", type=float, default=0.0,
                    help="출력 깊이 바닥값(단위=px-per-mm 적용 후). d=clip(d-floor,0). 잔여 노이즈 제거용.")
    ap.add_argument("--blank-frames", type=int, default=20,
                    help="blank 기준 프레임 평균 장수(많을수록 기준이 깨끗 → 노이즈↓).")
    ap.add_argument("--depth-ema", type=float, default=0.0,
                    help="깊이 시간축 평활(0~1, 0=off). 예 0.5: 프레임 간 깜빡이는 노이즈 완화(약간의 잔상).")
    ap.add_argument("--min-scale", type=float, default=1.0,
                    help="깊이 컬러맵/z축 최소 표시 범위(단위=mm 또는 rel). 무접촉 시 자동스케일이 "
                         "노이즈 수준까지 쪼그라들어 미세 노이즈가 크게 보이는 걸 방지. 작은 접촉을 보려면 낮추세요.")
    ap.add_argument("--mode", choices=["2d", "3d"], default="3d")
    ap.add_argument("--downsample", type=int, default=4, help="[3d] 표면 다운샘플 배수.")
    ap.add_argument("--rcount", type=int, default=40)
    ap.add_argument("--ccount", type=int, default=40)
    ap.add_argument("--target-fps", type=float, default=0.0)
    ap.add_argument("--no-render", action="store_true", help="창 없이 fps/수치만 출력.")
    ap.add_argument("--duration", type=float, default=0.0, help="[--no-render] N초 측정 후 자동 종료.")
    ap.add_argument("--headless", type=int, default=0, metavar="N", help="N프레임 후 스냅샷 저장.")
    ap.add_argument("--outdir", default="./digit_check_results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    px = args.px_per_mm if args.px_per_mm > 0 else None
    unit = "mm" if px else "rel"
    s = args.infer_scale
    print(f"디바이스: {device} | infer-scale: 1/{s} | 단위: {unit}")

    no_render = args.no_render
    headless = args.headless > 0
    if not no_render:
        if headless:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if args.mode == "3d":
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # QVGA 가 실제 지원하는 fps 로 스냅(비지원 값은 드라이버가 무시 → 대역폭 비결정적).
    if not args.mjpeg:
        snapped, changed = _snap_cam_fps(args.cam_fps)
        if changed:
            print(f"[주의] --cam-fps {args.cam_fps} 는 DIGIT QVGA 비지원값 → {snapped} 으로 스냅 "
                  f"(지원: {QVGA_SUPPORTED_FPS}). 비지원 값은 드라이버가 임의 반올림해 섞임이 재발합니다.")
            args.cam_fps = snapped

    serials = args.serials if args.serials else detect_serials()
    if not serials:
        print("[오류] DIGIT 센서를 찾을 수 없습니다. --serials 로 지정하세요."); sys.exit(1)
    print(f"대상 DIGIT 센서: {serials}")
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.root, exist_ok=True)

    # ----- 모델(모든 센서 공유): OPT(fp16 + channels_last + compile) -----
    print("TouchNet(DIGIT) 사전학습 로드 중...")
    opt_model = models.TouchNet(load_pretrained=True, sensor_type=SensorType.DIGIT, root=args.root)
    opt_model.to(device).eval().to(memory_format=torch.channels_last)
    opt_model_c = torch.compile(opt_model)

    # ----- 센서 연결 + 그래버 -----
    print("\n센서 연결 중...")
    sensors = {}
    grabbers = {}
    dev_names = {}  # serial -> /dev/videoN  (섞임 진단용)
    for serial in serials:
        try:
            sensor = DIGIT(serial); sensor.connect()
            dev = getattr(sensor.sensor, "dev_name", "?")
            parm_ok, parm_msg = (False, "")
            if args.mjpeg:
                # 대역폭 축소(MJPEG) + 버퍼=1.
                _configure_camera(sensor.sensor, args.cam_fps, mjpeg=True)
                if not _camera_ok(sensor):
                    print(f"  [{serial}] MJPEG 미지원으로 보임 → YUYV 폴백")
                    try: sensor.disconnect()
                    except Exception: pass
                    sensor = DIGIT(serial); sensor.connect()
                    dev = getattr(sensor.sensor, "dev_name", "?")
                    parm_ok, parm_msg = _force_qvga_fps(sensor.sensor, dev, args.cam_fps)
            else:
                # 핵심: 닫힌 상태 set-parm + S_FMT 없는 재오픈으로 프레임 간격을 실제로 고정한다.
                # (이 드라이버는 OpenCV CAP_PROP_FPS 를 무시하고, S_FMT 는 간격을 60 으로 리셋함)
                parm_ok, parm_msg = _force_qvga_fps(sensor.sensor, dev, args.cam_fps)
            sensor.flush_frames(args.warmup)
            # 스트리밍 시작 후 '실제' fps 측정 — 대역폭 축소가 정말 반영됐는지 검증.
            act_fps = _measure_fps(sensor, n=30)
            dev_names[serial] = dev
            g = FrameGrabber(sensor, serial); g.start()
            sensors[serial] = sensor
            grabbers[serial] = g
            drv = parm_msg if parm_msg else "?"
            fmt = f"{'MJPG' if args.mjpeg else 'YUYV'}@{args.cam_fps}req"
            warn = ""
            if not args.mjpeg and act_fps > args.cam_fps * 1.4:
                warn = (f"  <- [경고] 실측 {act_fps:.0f}fps: 대역폭 축소 미반영 → 섞임 위험"
                        + ("" if parm_ok else f" (v4l2 설정 실패: {parm_msg})"))
            print(f"  [연결됨] {serial} -> {dev}  [{fmt}, drv={drv}, 실측 ~{act_fps:.0f}fps]{warn}")
        except Exception as e:
            print(f"  [연결 실패] {serial}: {e}")
    if not sensors:
        print("[오류] 연결된 센서가 없습니다."); sys.exit(1)
    serials = list(sensors.keys())

    # 섞임의 가장 흔한 원인: 서로 다른 시리얼이 같은 /dev/video 노드로 매핑됨.
    # (DIGIT 카메라는 USB 디스크립터가 모두 동일해 udev 매핑이 어긋나면 이렇게 된다.)
    dup = {}
    for sr, dev in dev_names.items():
        dup.setdefault(dev, []).append(sr)
    collided = {dev: srs for dev, srs in dup.items() if len(srs) > 1}
    if collided:
        print("\n[오류] 여러 센서가 같은 카메라 장치로 매핑되었습니다 (= 프레임 섞임의 원인):")
        for dev, srs in collided.items():
            print(f"   {dev}  <-  {', '.join(srs)}")
        print("  udev 매핑이 어긋난 상태입니다. 위 시리얼들의 USB 재연결 또는 "
              "다른 USB 포트/허브로 분리 후 다시 시도하세요.")
        for g in grabbers.values():
            g.stop()
        time.sleep(0.05)
        for sn in sensors.values():
            try: sn.disconnect()
            except Exception: pass
        sys.exit(1)
    print(f"  장치 매핑 OK (중복 없음): {dev_names}")

    # ----- 센서별 blank/Poisson/스케일 준비 -----
    print("\n무접촉 상태로 두세요. 각 센서 blank 캡처...")
    time.sleep(1.0)
    state = {}  # serial -> dict(blank_low_t, Hl, Wl, H, W, gpu_poisson)
    gpu_poisson_cache = {}  # (Hl,Wl) -> GpuPoisson 공유
    for serial in serials:
        blank = capture_blank(grabbers[serial], n_avg=args.blank_frames)
        H, W = blank.shape[:2]
        Hl, Wl = H // s, W // s
        key = (Hl, Wl)
        if key not in gpu_poisson_cache:
            gpu_poisson_cache[key] = GpuPoisson(Hl, Wl, device)
        blank_low = cv2.resize(blank, (Wl, Hl), interpolation=cv2.INTER_AREA)
        blank_low_t = torch.from_numpy(np.ascontiguousarray(blank_low)).permute(2, 0, 1).float() / 255.0
        state[serial] = dict(blank_low_t=blank_low_t, Hl=Hl, Wl=Wl, H=H, W=W,
                             gpu_poisson=gpu_poisson_cache[key])
        print(f"  [{serial}] blank 완료 ({W}x{H} -> {Wl}x{Hl})")
    print("blank 캡처 완료. 센서를 눌러보세요.\n")

    # 모든 DIGIT 센서는 같은 해상도/모델을 공유하므로 한 번의 forward 로 '배치 추론'한다.
    # (센서별 순차 forward + 센서별 .cpu() 동기화가 가장 큰 지연원이라 이를 제거.)
    use_cuda = device == "cuda"
    pin = use_cuda  # CPU->GPU 비동기 전송용 pinned memory
    H0, W0 = state[serials[0]]["H"], state[serials[0]]["W"]
    Hl0, Wl0 = state[serials[0]]["Hl"], state[serials[0]]["Wl"]
    uniform = all(state[sr]["H"] == H0 and state[sr]["W"] == W0 for sr in serials)
    poisson0 = state[serials[0]]["gpu_poisson"]

    # ----- 노이즈 억제 파라미터 -----
    diff_thresh = args.diff_thresh   # 입력 dead-band (무접촉 노이즈가 Poisson 으로 증폭되는 것 차단)
    depth_floor = args.depth_floor   # 출력 바닥값
    depth_ema = args.depth_ema       # 시간축 평활
    _ema_state = {}                  # serial -> 직전 깊이(EMA)

    def deadband(diff):
        """|diff|<thresh 를 0으로(무접촉 픽셀이 정확히 0이 되어 Poisson 적분 시 평탄)."""
        if diff_thresh <= 0:
            return diff
        return diff.masked_fill(diff.abs() < diff_thresh, 0.0)

    def postprocess(sr, d):
        """출력 깊이에 바닥값/시간축 평활 적용. d: ndarray(H,W)."""
        if depth_floor > 0:
            d = np.clip(d - depth_floor, 0, None)
        if depth_ema > 0:
            prev = _ema_state.get(sr)
            d = d if prev is None else (depth_ema * prev + (1 - depth_ema) * d)
            _ema_state[sr] = d
        return d

    def depth_opt_batch(frames):
        """frames: {serial: ndarray} -> {serial: depth ndarray}. 단일 forward + 단일 .cpu()."""
        order = [sr for sr in serials if sr in frames]
        if not order:
            return {}
        if uniform:
            # 전처리(저해상도+좌표임베딩)를 한 배치로 쌓는다.
            augs = []
            for sr in order:
                st = state[sr]
                fl = cv2.resize(frames[sr], (Wl0, Hl0), interpolation=cv2.INTER_AREA)
                img = torch.from_numpy(np.ascontiguousarray(fl)).permute(2, 0, 1).float() / 255.0
                augs.append(coord_embed_scaled(deadband(img - st["blank_low_t"]), H0, W0))
            batch = torch.stack(augs, 0)
            if pin:
                batch = batch.pin_memory()
            batch = batch.to(device, non_blocking=pin).to(memory_format=torch.channels_last)
            with torch.no_grad():
                if use_cuda:
                    with torch.autocast("cuda", dtype=torch.float16):
                        out = opt_model_c(batch)          # (N,2,Hl,Wl)
                else:
                    out = opt_model_c(batch)
            out = out.float()
            ds = [torch.clamp(-poisson0(out[i, 0], out[i, 1]), min=0) for i in range(len(order))]
            d = torch.stack(ds, 0)                          # (N,Hl,Wl) GPU
            d = F.interpolate(d[None], size=(H0, W0), mode="bilinear", align_corners=False)[0]
            d = d.cpu().numpy()                             # 동기화 1회
            if px:
                d = d / px
            return {sr: postprocess(sr, d[i]) for i, sr in enumerate(order)}
        # 해상도가 섞이면(드묾) 센서별 처리로 폴백
        return {sr: _depth_opt_single(sr, frames[sr]) for sr in order}

    def _depth_opt_single(serial, frame):
        st = state[serial]
        Hl, Wl, H, W = st["Hl"], st["Wl"], st["H"], st["W"]
        fl = cv2.resize(frame, (Wl, Hl), interpolation=cv2.INTER_AREA)
        img = torch.from_numpy(np.ascontiguousarray(fl)).permute(2, 0, 1).float() / 255.0
        aug = coord_embed_scaled(deadband(img - st["blank_low_t"]), H, W).unsqueeze(0).to(device)
        aug = aug.to(memory_format=torch.channels_last)
        with torch.no_grad():
            if use_cuda:
                with torch.autocast("cuda", dtype=torch.float16):
                    out = opt_model_c(aug)
            else:
                out = opt_model_c(aug)
        o = out.squeeze(0).float()
        d = torch.clamp(-st["gpu_poisson"](o[0], o[1]), min=0)
        d = F.interpolate(d[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        d = d.cpu().numpy()
        d = d / px if px else d
        return postprocess(serial, d)

    def reblank(serial):
        st = state[serial]
        blank = capture_blank(grabbers[serial], n_avg=args.blank_frames)
        bl = cv2.resize(blank, (st["Wl"], st["Hl"]), interpolation=cv2.INTER_AREA)
        st["blank_low_t"] = torch.from_numpy(np.ascontiguousarray(bl)).permute(2, 0, 1).float() / 255.0

    def wait_all_frames():
        for serial in serials:
            while grabbers[serial].latest() is None:
                if grabbers[serial]._err:
                    raise grabbers[serial]._err
                time.sleep(0.005)

    # 워밍업: torch.compile + cudnn autotune (배치 크기 N 으로 컴파일되도록 동일 형태로)
    print("워밍업(compile + cudnn autotune)...")
    wait_all_frames()
    for _ in range(5):
        depth_opt_batch({sr: grabbers[sr].latest() for sr in serials})

    def disconnect_all():
        for serial in serials:
            grabbers[serial].stop()
        time.sleep(0.05)
        for serial in serials:
            try:
                sensors[serial].disconnect()
            except Exception:
                pass
        print("연결 해제.")

    # ---------- no-render ----------
    if no_render:
        print("no-render: 무접촉으로 두고 센서별 노이즈 측정 (Ctrl-C 종료)\n")
        fc = 0; ct = 0.0; t0 = time.time()
        last_d = {}
        nf = 0.3 if px else 0.3  # 노이즈 판정 임계(단위=mm 또는 rel)
        acc = {sr: {"max": 0.0, "p99": 0.0, "frac": 0.0, "n": 0} for sr in serials}
        try:
            while True:
                t_a = time.time()
                frames = {sr: grabbers[sr].latest() for sr in serials}
                frames = {sr: f for sr, f in frames.items() if f is not None}
                out = depth_opt_batch(frames)
                last_d.update(out)
                ct += time.time() - t_a
                fc += 1
                for sr, d in out.items():
                    a = acc[sr]
                    a["max"] += float(d.max())
                    a["p99"] += float(np.percentile(d, 99))
                    a["frac"] += float((d > nf).mean())
                    a["n"] += 1
                if fc % 5 == 0:
                    maxes = " | ".join(f"{sr}={last_d[sr].max():.3f}" for sr in serials if sr in last_d)
                    print(f"frame {fc} | {len(serials)} sensors {fc*len(serials)/ct:6.1f} infer-fps "
                          f"| loop {fc/(time.time()-t0):5.1f} fps | max[{unit}] {maxes}")
                if args.duration and (time.time() - t0) >= args.duration:
                    break
        except KeyboardInterrupt:
            print("\n중단됨.")
        finally:
            disconnect_all()
        if fc > 0:
            print("\n=== 속도 ===")
            print(f"센서 {len(serials)}개 동시 추론: 프레임당 {1000*ct/fc:6.2f} ms "
                  f"(센서당 {1000*ct/fc/len(serials):6.2f} ms) -> {fc*len(serials)/ct:6.1f} infer-fps")
            print(f"\n=== 무접촉 노이즈 (작을수록 좋음, 임계 {nf:.2f}{unit}) ===")
            print(f"  {'serial':>10}  {'평균 max':>9}  {'평균 p99':>9}  {'노이즈 픽셀%':>10}")
            for sr in serials:
                a = acc[sr]
                if a["n"] == 0:
                    continue
                m, p, fr = a["max"]/a["n"], a["p99"]/a["n"], 100*a["frac"]/a["n"]
                flag = "  <- 노이즈 큼" if p > nf else ""
                print(f"  {sr:>10}  {m:9.3f}  {p:9.3f}  {fr:9.2f}%{flag}")
            print(f"\n  포맷={'MJPEG' if args.mjpeg else 'YUYV'} cam-fps={args.cam_fps} "
                  f"diff-thresh={args.diff_thresh} | 1대 vs 4대 같은 옵션으로 이 수치를 비교하세요.")
        return

    # ---------- 렌더 ----------
    # 센서당 [카메라 | 깊이] 2패널. 센서들을 grid_shape 블록으로 배치하고
    # 각 블록을 가로 2칸(카메라+깊이)으로 펼쳐 총 (rows, 2*cols) 격자를 만든다.
    is3d = args.mode == "3d"
    n = len(serials)
    rows, cols = grid_shape(n)
    ncols_total = 2 * cols
    fig = plt.figure(figsize=(4.2 * ncols_total, 4.0 * rows))
    fig.suptitle(f"DIGIT x{n} 카메라+깊이 동시 (TouchNet OPT: 1/{s} + GPU-Poisson + compile)", fontsize=13)

    # 센서별 초기 깊이/렌더 핸들
    depths0 = depth_opt_batch({sr: grabbers[sr].latest() for sr in serials})
    emas = {sr: max(depths0[sr].max(), 1e-3) for sr in serials}
    # vmax_of: 표시 범위는 EMA 최댓값을 따라가되 min_scale 아래로는 안 내려간다.
    # (무접촉이면 d.max()≈0 이라 자동스케일이 노이즈 수준으로 붕괴해 노이즈가 크게 보임)
    min_scale = max(args.min_scale, 1e-6)
    def vmax_of(sr):
        return max(emas[sr], min_scale)
    axes = {}        # serial -> 깊이 axis
    ims = {}         # serial -> 깊이 image(2d)
    cam_ims = {}     # serial -> 카메라 image
    surfs = {sr: None for sr in serials}
    grids = {}
    for i, sr in enumerate(serials):
        gr, gc = i // cols, i % cols
        cam_idx = gr * ncols_total + gc * 2 + 1
        depth_idx = cam_idx + 1

        # 카메라 패널
        ax_cam = fig.add_subplot(rows, ncols_total, cam_idx)
        ax_cam.set_title(f"{sr} camera"); ax_cam.axis("off")
        cam_ims[sr] = ax_cam.imshow(np.ascontiguousarray(grabbers[sr].latest()[:, :, ::-1]))

        # 깊이 패널
        if is3d:
            ax = fig.add_subplot(rows, ncols_total, depth_idx, projection="3d")
            ax.set_title(f"{sr} depth ({unit})")
            Hd, Wd = downsample(depths0[sr], args.downsample).shape
            Yg, Xg = np.mgrid[0:Hd, 0:Wd]
            grids[sr] = (Xg, Yg)
        else:
            ax = fig.add_subplot(rows, ncols_total, depth_idx)
            ax.set_title(f"{sr} depth ({unit})"); ax.axis("off")
            im = ax.imshow(depths0[sr], cmap="viridis", vmin=0, vmax=vmax_of(sr))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ims[sr] = im
        axes[sr] = ax

    state_flags = {"quit": False, "reblank": False}

    def on_key(ev):
        if ev.key in ("q", "escape"): state_flags["quit"] = True
        elif ev.key == "r": state_flags["reblank"] = True

    if not headless:
        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.ion(); plt.show(block=False)

    min_dt = (1.0 / args.target_fps) if args.target_fps > 0 else 0.0
    fc = 0; ct = 0.0; t0 = time.time(); last = t0
    try:
        while True:
            t_a = time.time()
            frames = {}
            for sr in serials:
                frame = grabbers[sr].latest()
                if frame is None:
                    if grabbers[sr]._err: raise grabbers[sr]._err
                    continue
                frames[sr] = frame
            if len(frames) < n:
                time.sleep(0.005); continue
            depths = depth_opt_batch(frames)
            ct += time.time() - t_a
            fc += 1

            for sr in serials:
                d = depths[sr]
                cam_ims[sr].set_data(np.ascontiguousarray(frames[sr][:, :, ::-1]))
                emas[sr] = 0.9 * emas[sr] + 0.1 * max(d.max(), 1e-3)
                vmax = vmax_of(sr)
                if is3d:
                    ax = axes[sr]; Xg, Yg = grids[sr]
                    if surfs[sr] is not None: surfs[sr].remove()
                    surfs[sr] = ax.plot_surface(Xg, Yg, downsample(d, args.downsample), cmap="viridis",
                                                linewidth=0, antialiased=False, vmin=0, vmax=vmax,
                                                rcount=args.rcount, ccount=args.ccount)
                    ax.set_zlim(0, vmax)
                else:
                    ims[sr].set_data(d); ims[sr].set_clim(0, vmax)

            if fc % 15 == 0:
                maxes = " ".join(f"{sr}={depths[sr].max():.2f}" for sr in serials)
                print(f"  infer {fc*n/ct:5.1f}fps | loop {fc/(time.time()-t0):5.1f}fps | max[{unit}] {maxes}")

            if headless:
                if fc >= args.headless: break
            else:
                fig.canvas.draw_idle(); fig.canvas.flush_events()
                if state_flags["quit"] or not plt.fignum_exists(fig.number): break
                if state_flags["reblank"]:
                    state_flags["reblank"] = False
                    print("모든 센서 blank 재캡처...")
                    for sr in serials:
                        reblank(sr)
                    print("blank 갱신 완료.")
            if min_dt:
                dd = time.time() - last
                if dd < min_dt: time.sleep(min_dt - dd)
            last = time.time()
    except KeyboardInterrupt:
        print("\n중단됨.")
    finally:
        disconnect_all()

    if headless:
        snap = os.path.join(args.outdir, f"all_digits_touchnet_opt_{args.mode}.png")
        fig.savefig(snap, dpi=130, bbox_inches="tight")
        print(f"스냅샷 저장: {snap}")
    print("=== 완료 ===")


if __name__ == "__main__":
    main()
