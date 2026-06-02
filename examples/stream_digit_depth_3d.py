"""
DIGIT 센서 한 대를 실시간 스트리밍하면서 py3DCal 사전학습 TouchNet(DIGIT) 모델로
깊이맵을 매 프레임 계산해 실시간 시각화한다. (3D 표면 또는 2D 히트맵)

실시간성(=sync) 개선 포인트:
  1) 백그라운드 캡처 스레드(FrameGrabber)가 카메라 프레임을 계속 읽어 '최신 프레임'만
     유지한다. 메인 루프는 항상 가장 최근 프레임으로 추론/렌더하므로 V4L2 버퍼에
     프레임이 쌓여 화면이 뒤늦게 따라오는 지연(latency)이 사라진다.
  2) 추론 전처리에서 PIL 왕복을 제거하고 텐서로 직접 변환(get_depthmap 과 동일한
     수치 결과를 유지하면서 더 빠르다).
  3) 3D 렌더는 plot_surface 의 폴리곤 수를 rcount/ccount 로 제한하고, plt.pause 대신
     draw_idle()+flush_events() 로 가볍게 갱신한다.
  4) --mode 2d 는 imshow 한 장만 set_data 로 갱신 → 3D 보다 훨씬 빠르다(수십 fps).

조작: 'q'/ESC 종료, 'r' blank 재캡처.

DIGIT 카메라 이미지는 모든 모드에서 항상 함께 표시되며, 같은 프레임으로 동기 갱신된다.
  --mode all (기본): 카메라 + 2D 깊이 + 3D 깊이 (3패널)
  --mode 2d        : 카메라 + 2D 깊이 (2패널)
  --mode 3d        : 카메라 + 3D 깊이 (2패널)

실행 (디스플레이 필요):
  conda activate py3dcal
  python examples/stream_digit_depth_3d.py                      # 기본: 3패널
  python examples/stream_digit_depth_3d.py --serial D21424 --mode 2d
  python examples/stream_digit_depth_3d.py --serial D21424 --downsample 6

점검용(디스플레이 없이 N프레임 처리 후 스냅샷 저장):
  conda run -n py3dcal python examples/stream_digit_depth_3d.py --headless 60
"""
import argparse
import os
import sys
import threading
import time

import numpy as np
import torch
import matplotlib

import py3DCal as p3d  # noqa: F401
from py3DCal import DIGIT, models, SensorType
from py3DCal.model_training.lib.add_coordinate_embeddings import add_coordinate_embeddings
from py3DCal.model_training.lib.fast_poisson import fast_poisson

try:
    from digit_interface import DigitHandler
except Exception as e:  # pragma: no cover
    DigitHandler = None
    _IMPORT_ERR = e


def detect_first_serial():
    if DigitHandler is None:
        return None
    serials = []
    for d in DigitHandler.list_digits():
        s = d.get("serial") if isinstance(d, dict) else None
        if s and s not in serials:
            serials.append(s)
    return serials[0] if serials else None


class FrameGrabber(threading.Thread):
    """카메라를 계속 읽어 '최신 프레임'만 보관하는 데몬 스레드.

    DIGIT.capture_image() -> get_frame() -> VideoCapture.read() 는 다음 프레임이
    올 때까지 블로킹하므로, 이 스레드는 자연스럽게 카메라 fps 로 돌며 self.frame 을
    항상 최신으로 덮어쓴다. 메인 루프는 latest() 로 가장 최근 프레임만 가져간다.
    """
    def __init__(self, sensor):
        super().__init__(daemon=True)
        self.sensor = sensor
        self._lock = threading.Lock()
        self._frame = None
        self._running = True
        self._err = None

    def run(self):
        while self._running:
            try:
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


def frame_to_tensor(frame):
    """uint8 HxWx3 프레임 -> (3,H,W) float 텐서 [0,1] (CPU). PIL 왕복 없이.

    py3DCal 파이프라인은 Image.fromarray(frame).convert('RGB') 후 ToTensor 인데,
    frame 이 이미 3채널이므로 채널 변환 없이 동일 결과가 된다.
    add_coordinate_embeddings 가 CPU 텐서를 만들므로 임베딩까지는 CPU 에서 처리하고,
    모델 입력 직전에 GPU 로 올린다(get_depthmap 과 동일한 순서/수치).
    """
    t = torch.from_numpy(np.ascontiguousarray(frame))
    return t.permute(2, 0, 1).float().div_(255.0)


def compute_depth(model, frame, blank_t, device, use_fp16=False, px_per_mm=None):
    """최신 프레임 -> 깊이맵(ndarray). get_depthmap 과 동일한 수치 파이프라인.

    TouchNet 은 표면 기울기(dz/dx, dz/dy)를 출력하고 fast_poisson 이 이를 적분한다.
    학습 타깃 기울기가 '픽셀' 단위(구 반지름을 px 로 둠)이므로 적분된 깊이도 '픽셀'
    단위다. px_per_mm 가 주어지면 depth_mm = depth_px / px_per_mm 로 mm 로 변환한다.

    use_fp16=True 면 모델 forward 를 FP16 autocast + channels_last 로 실행해
    추론을 ~2배 빠르게 한다(실시간 시각화에는 정밀도 영향 무시 가능).
    """
    img = frame_to_tensor(frame)
    aug = img - blank_t                                  # CPU
    aug = add_coordinate_embeddings(aug).unsqueeze(0).to(device)  # 임베딩(CPU) 후 GPU 로
    if use_fp16:
        aug = aug.to(memory_format=torch.channels_last)
    with torch.no_grad():
        if use_fp16:
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(aug)
        else:
            out = model(aug)
    out = out.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    depth = fast_poisson(out[:, :, 0], out[:, :, 1])
    depth = np.clip(-depth, a_min=0, a_max=None)         # 픽셀 단위 상대 깊이
    if px_per_mm:
        depth = depth / px_per_mm                        # -> mm
    return depth


def capture_blank(grabber, n_avg=10, settle=0.6):
    """그래버가 보관 중인 최신 프레임들을 모아 평균낸 무접촉 기준 프레임."""
    # 그래버가 첫 프레임을 채울 때까지 대기
    while grabber.latest() is None:
        time.sleep(0.01)
    acc, n = None, 0
    for _ in range(n_avg):
        f = grabber.latest().astype(np.float32)
        acc = f if acc is None else acc + f
        n += 1
        time.sleep(settle / n_avg)
    return (acc / n).astype(np.uint8)


def downsample(arr, factor):
    return arr if factor <= 1 else arr[::factor, ::factor]


def main():
    parser = argparse.ArgumentParser(description="DIGIT 실시간 깊이 스트리밍 시각화")
    parser.add_argument("--serial", default=None, help="DIGIT 시리얼. 생략 시 첫 번째 자동 탐지.")
    parser.add_argument("--root", default="./digit_weights", help="사전학습 가중치 디렉토리.")
    parser.add_argument("--device", default=None, help="추론 디바이스. 기본: cuda 가능시 cuda.")
    parser.add_argument("--mode", choices=["all", "2d", "3d"], default="all",
                        help="all: DIGIT 이미지+2D 깊이+3D 깊이 3개 동시(기본) / 2d: 깊이 히트맵만 / 3d: 깊이 표면만.")
    parser.add_argument("--downsample", type=int, default=4,
                        help="[3d] 표면 다운샘플 배수. 클수록 가볍다. 기본 4.")
    parser.add_argument("--rcount", type=int, default=40,
                        help="[3d] plot_surface 행 폴리곤 상한. 작을수록 빠르다. 기본 40.")
    parser.add_argument("--ccount", type=int, default=40,
                        help="[3d] plot_surface 열 폴리곤 상한. 기본 40.")
    parser.add_argument("--warmup", type=int, default=50, help="연결 후 버릴 프레임 수.")
    parser.add_argument("--zmax", type=float, default=None,
                        help="z(깊이) 상한 고정값. 미지정 시 EMA 자동 스케일.")
    parser.add_argument("--target-fps", type=float, default=0.0,
                        help="렌더 상한 fps(0=무제한). CPU 점유 줄이고 싶을 때.")
    parser.add_argument("--no-fp16", action="store_true",
                        help="FP16 가속을 끄고 FP32 로 추론(기본은 cuda 에서 FP16 사용).")
    parser.add_argument("--px-per-mm", type=float, default=15.0,
                        help="깊이(px)->mm 변환 계수. 캘리브레이션 metadata.json 의 px_per_mm 값을 쓰면 정확. "
                             "기본 15.0 은 DIGIT 근사값(센싱면 ~16mm/240px). 0 으로 주면 변환 없이 상대단위 표시.")
    parser.add_argument("--headless", type=int, default=0, metavar="N",
                        help="디스플레이 없이 N프레임 처리 후 스냅샷 저장(점검용).")
    parser.add_argument("--outdir", default="./digit_check_results", help="스냅샷 저장 위치.")
    parser.add_argument("--no-render", action="store_true",
                        help="창을 전혀 띄우지 않고 깊이 추정값만 콘솔에 계속 출력(최대 fps 측정용).")
    parser.add_argument("--print-n", type=int, default=8,
                        help="--no-render 에서 매 프레임 출력할 깊이 표본 개수. 기본 8.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = (device == "cuda") and (not args.no_fp16)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True  # 고정 입력크기 conv 알고리즘 autotune
    px_per_mm = args.px_per_mm if args.px_per_mm and args.px_per_mm > 0 else None
    unit = "mm" if px_per_mm else "rel"  # 축 단위 표기
    print(f"디바이스: {device} (cuda available: {torch.cuda.is_available()}) | fp16: {use_fp16}")
    if px_per_mm:
        print(f"깊이 단위: mm (px_per_mm={px_per_mm}; 정확한 값은 캘리브레이션 metadata.json 참고)")
    else:
        print("깊이 단위: 상대값(rel). --px-per-mm 를 주면 mm 로 표시됩니다.")

    no_render = args.no_render
    headless = args.headless > 0
    if not no_render:
        if headless:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if args.mode == "3d":
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    serial = args.serial or detect_first_serial()
    if not serial:
        print("[오류] DIGIT 센서를 찾을 수 없습니다. 연결을 확인하거나 --serial 로 지정하세요.")
        sys.exit(1)
    print(f"대상 DIGIT: {serial} | 모드: {args.mode}")

    os.makedirs(args.root, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)

    print("사전학습 DIGIT 가중치 로드 중...")
    model = models.TouchNet(load_pretrained=True, sensor_type=SensorType.DIGIT, root=args.root)
    model.to(device)
    model.eval()
    if use_fp16:
        model = model.to(memory_format=torch.channels_last)

    sensor = DIGIT(serial)
    sensor.connect()
    sensor.flush_frames(n=args.warmup)
    print("센서 연결 완료.")

    grabber = FrameGrabber(sensor)
    grabber.start()

    print("\n무접촉 상태로 두세요. 기준(blank) 프레임을 캡처합니다...")
    time.sleep(1.0)
    blank = capture_blank(grabber)
    blank_t = frame_to_tensor(blank)
    print("blank 캡처 완료. 이제 센서를 눌러보세요.\n")

    # 첫 프레임/깊이맵으로 격자/아티스트 초기화
    frame0 = grabber.latest()
    depth0 = compute_depth(model, frame0, blank_t, device, use_fp16, px_per_mm)

    # --no-render: 창 없이 깊이 추정값만 콘솔에 계속 출력(최대 fps 측정/실시간 변화 확인용)
    if no_render:
        n = max(1, args.print_n)
        print(f"no-render 모드: 창 없이 깊이값만 출력합니다. (Ctrl-C 종료)")
        print(f"표본은 깊이맵 중앙 가로선에서 {n}점을 뽑습니다. max/mean 과 함께 실시간으로 바뀝니다.\n")
        min_dt = (1.0 / args.target_fps) if args.target_fps > 0 else 0.0
        frame_count = 0
        t0 = time.time()
        last = t0
        try:
            while True:
                frame = grabber.latest()
                if frame is None:
                    if grabber._err:
                        raise grabber._err
                    time.sleep(0.005)
                    continue
                depth = compute_depth(model, frame, blank_t, device, use_fp16, px_per_mm)
                frame_count += 1
                fps = frame_count / (time.time() - t0)
                h, w = depth.shape
                idxs = np.linspace(0, w - 1, n).astype(int)
                row = depth[h // 2, idxs]
                sample = " ".join(f"{v:6.3f}" for v in row)
                print(f"[{fps:6.1f} fps] max={depth.max():6.3f} mean={depth.mean():6.4f} {unit} | center[{sample}]")
                if min_dt:
                    dt = time.time() - last
                    if dt < min_dt:
                        time.sleep(min_dt - dt)
                last = time.time()
        except KeyboardInterrupt:
            print("\n중단됨(Ctrl-C).")
        finally:
            grabber.stop()
            time.sleep(0.05)
            try:
                sensor.disconnect()
            except Exception:
                pass
            print("센서 연결 해제 완료.")
        return

    state = {"quit": False, "reblank": False}

    def on_key(event):
        if event.key in ("q", "escape"):
            state["quit"] = True
        elif event.key == "r":
            state["reblank"] = True

    # z 자동 스케일용 EMA
    z_ema = max(depth0.max(), 1e-3)

    im = None        # 2D 깊이 히트맵 아티스트
    im_raw = None    # DIGIT 원본 카메라 이미지 아티스트
    surf = None      # 3D 표면 아티스트
    ax = None        # 3D 축
    need_3d = args.mode in ("all", "3d")

    def rgb_for_display(fr):
        """py3DCal DIGIT 프레임(BGR 배열)을 화면 표시용 RGB 로 변환."""
        return np.ascontiguousarray(fr[:, :, ::-1])

    # 3D 표면용 좌표 격자(다운샘플 기준)
    if need_3d:
        ds0 = downsample(depth0, args.downsample)
        H, W = ds0.shape
        Y, X = np.mgrid[0:H, 0:W]

    # DIGIT 카메라 이미지는 모든 모드에서 항상 표시한다.
    # 패널 구성: [카메라] (+ [2D 깊이]) (+ [3D 깊이])  — 모두 같은 프레임으로 동기 갱신.
    ncols = 1 + (args.mode in ("2d", "all")) + need_3d
    fig = plt.figure(figsize=(6 * ncols, 6.5))
    col = 1

    ax_img = fig.add_subplot(1, ncols, col)
    ax_img.set_title(f"DIGIT {serial} camera")
    ax_img.axis("off")
    im_raw = ax_img.imshow(rgb_for_display(frame0))
    col += 1

    if args.mode in ("2d", "all"):
        ax2d = fig.add_subplot(1, ncols, col)
        ax2d.set_title(f"depth (2D) [{unit}]")
        ax2d.axis("off")
        im = ax2d.imshow(depth0, cmap="viridis", vmin=0, vmax=z_ema)
        cbar = fig.colorbar(im, ax=ax2d, fraction=0.046, pad=0.04)
        cbar.set_label(f"depth ({unit})")
        col += 1

    if need_3d:
        ax = fig.add_subplot(1, ncols, col, projection="3d")
        ax.set_title("depth (3D)")
        ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)"); ax.set_zlabel(f"depth ({unit})")
        col += 1

    if not headless:
        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.ion()
        plt.show(block=False)

    min_dt = (1.0 / args.target_fps) if args.target_fps > 0 else 0.0
    frame_count = 0
    t0 = time.time()
    last = t0
    try:
        while True:
            frame = grabber.latest()
            if frame is None:
                if grabber._err:
                    raise grabber._err
                time.sleep(0.005)
                continue

            depth = compute_depth(model, frame, blank_t, device, use_fp16, px_per_mm)

            # z 스케일: 고정값 없으면 EMA 로 부드럽게 추종
            if args.zmax is not None:
                zmax = args.zmax
            else:
                z_ema = 0.9 * z_ema + 0.1 * max(depth.max(), 1e-3)
                zmax = z_ema

            if im_raw is not None:      # DIGIT 원본 이미지
                im_raw.set_data(rgb_for_display(frame))
            if im is not None:          # 2D 깊이 히트맵
                im.set_data(depth)
                im.set_clim(0, zmax)
            if need_3d:                 # 3D 깊이 표면
                ds = downsample(depth, args.downsample)
                if surf is not None:
                    surf.remove()
                surf = ax.plot_surface(
                    X, Y, ds, cmap="viridis", linewidth=0, antialiased=False,
                    vmin=0, vmax=zmax, rcount=args.rcount, ccount=args.ccount,
                )
                ax.set_zlim(0, zmax)

            frame_count += 1
            if frame_count % 20 == 0:
                fps = frame_count / (time.time() - t0)
                print(f"  frame {frame_count} | depth max={depth.max():.3f} {unit} | render ~{fps:.1f} fps")

            if headless:
                if frame_count >= args.headless:
                    break
            else:
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                if state["quit"] or not plt.fignum_exists(fig.number):
                    break
                if state["reblank"]:
                    state["reblank"] = False
                    print("blank 재캡처 중... 무접촉 상태로 두세요.")
                    blank = capture_blank(grabber)
                    blank_t = frame_to_tensor(blank)
                    print("blank 갱신 완료.")

            if min_dt:
                dt = time.time() - last
                if dt < min_dt:
                    time.sleep(min_dt - dt)
            last = time.time()
    except KeyboardInterrupt:
        print("\n중단됨(Ctrl-C).")
    finally:
        grabber.stop()
        time.sleep(0.05)
        try:
            sensor.disconnect()
        except Exception:
            pass
        print("센서 연결 해제 완료.")

    if headless:
        snap = os.path.join(args.outdir, f"{serial}_live_{args.mode}_snapshot.png")
        fig.savefig(snap, dpi=130, bbox_inches="tight")
        print(f"스냅샷 저장: {snap}")

    print("=== 완료 ===")


if __name__ == "__main__":
    main()
