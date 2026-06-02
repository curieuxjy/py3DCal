"""
DIGIT 실시간 스트리밍으로 깊이 추정 모델들을 나란히 비교한다.

  (1) TouchNet           : py3DCal 사전학습 CNN (gradient -> Poisson 적분, blank 차감)
  (2) DPT (ViT) x N      : NeuralFeels tactile transformer (dpt_real.p / dpt_sim.p, vit_small_patch16_224)

패널: [DIGIT 카메라 | TouchNet 깊이 | DPT#1 | DPT#2 ...]  — 모두 같은 프레임으로 동기.
--dpt-weights 에 여러 .p 를 주면 DPT 패널이 그만큼 늘어난다(예: dpt_real + dpt_sim 동시 비교).

공정 비교:
  - 모두 '무접촉(blank) 기준 상대 접촉깊이'로 맞춘다.
      TouchNet : 입력에서 blank 이미지를 빼고 추론.
      DPT      : heightmap(현재) - heightmap(blank) (NeuralFeels heightmap2mask 와 동일).
  - TouchNet 은 mm(px_per_mm 변환), DPT 는 상대값(0~255 Δheightmap). 단위가 달라 각자 컬러스케일로 표시.

환경: py3dcal (digit_interface 라이브 캡처). DPT 는 timm + 로컬 dpt_lite 로 가볍게 구동.

실행:
  conda activate py3dcal
  # TouchNet vs DPT(real) vs DPT(sim) 3-way 비교(2D)
  python examples/stream_compare_touchnet_dpt.py --serial D21424 \
      --dpt-weights /home/avery/Documents/neuralfeels/deploy/weights/tactile_transformer/dpt_real.p \
                    /home/avery/Documents/neuralfeels/deploy/weights/tactile_transformer/dpt_sim.p
  # 3D 표면 비교
  python examples/stream_compare_touchnet_dpt.py --serial D21424 --mode 3d
  # 창 없이 모델별 fps 측정(약 10초)
  python examples/stream_compare_touchnet_dpt.py --serial D21424 --no-render --duration 10
"""
import argparse
import os
import sys
import threading
import time

import numpy as np
import torch
import matplotlib

import py3DCal as p3d
from py3DCal import DIGIT, models, SensorType
from py3DCal.model_training.lib.add_coordinate_embeddings import add_coordinate_embeddings
from py3DCal.model_training.lib.fast_poisson import fast_poisson

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dpt_lite.loader import build_dpt, dpt_heightmap

try:
    from digit_interface import DigitHandler
except Exception:
    DigitHandler = None

_WDIR = "/home/avery/Documents/neuralfeels/deploy/weights/tactile_transformer"
# 기본: dpt_real + dpt_sim 둘 다 -> TouchNet 포함 3개 동시 비교
_DEFAULT_DPT = [os.path.join(_WDIR, "dpt_real.p"), os.path.join(_WDIR, "dpt_sim.p")]


def detect_first_serial():
    if DigitHandler is None:
        return None
    for d in DigitHandler.list_digits():
        s = d.get("serial") if isinstance(d, dict) else None
        if s:
            return s
    return None


class FrameGrabber(threading.Thread):
    """카메라를 계속 읽어 '최신 프레임'만 보관(지연 최소화)."""
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


# ---------- TouchNet ----------
def touchnet_tensor(frame):
    t = torch.from_numpy(np.ascontiguousarray(frame))
    return t.permute(2, 0, 1).float().div_(255.0)


def touchnet_depth(model, frame, blank_t, device, use_fp16, px_per_mm):
    aug = touchnet_tensor(frame) - blank_t
    aug = add_coordinate_embeddings(aug).unsqueeze(0).to(device)
    if use_fp16:
        aug = aug.to(memory_format=torch.channels_last)
    with torch.no_grad():
        if use_fp16:
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(aug)
        else:
            out = model(aug)
    out = out.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    depth = np.clip(-fast_poisson(out[:, :, 0], out[:, :, 1]), 0, None)
    return depth / px_per_mm if px_per_mm else depth


def capture_blank(grabber, n_avg=10, settle=0.6):
    while grabber.latest() is None:
        time.sleep(0.01)
    acc = None
    for _ in range(n_avg):
        f = grabber.latest().astype(np.float32)
        acc = f if acc is None else acc + f
        time.sleep(settle / n_avg)
    return (acc / n_avg).astype(np.uint8)


def downsample(arr, factor):
    return arr if factor <= 1 else arr[::factor, ::factor]


def main():
    ap = argparse.ArgumentParser(description="DIGIT 실시간 TouchNet vs DPT(들) 비교")
    ap.add_argument("--serial", default=None, help="DIGIT 시리얼. 생략 시 첫 번째 자동 탐지.")
    ap.add_argument("--touchnet-root", default="./digit_weights", help="TouchNet 가중치 디렉토리.")
    ap.add_argument("--dpt-weights", nargs="+", default=_DEFAULT_DPT,
                    help="DPT(.p) 경로들. 기본은 dpt_real + dpt_sim 둘 다(=TouchNet 포함 3개 비교). 하나만 주면 2개 비교.")
    ap.add_argument("--device", default=None, help="기본: cuda 가능시 cuda.")
    ap.add_argument("--no-fp16", action="store_true", help="FP16 가속 끄기.")
    ap.add_argument("--px-per-mm", type=float, default=15.0,
                    help="TouchNet 깊이(px)->mm 변환 계수. 0 이면 상대단위.")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--mode", choices=["2d", "3d"], default="2d",
                    help="2d: 깊이 히트맵 비교(기본) / 3d: 깊이 표면 비교. 카메라 이미지는 항상 함께 표시.")
    ap.add_argument("--downsample", type=int, default=4, help="[3d] 표면 다운샘플 배수. 기본 4.")
    ap.add_argument("--rcount", type=int, default=40, help="[3d] plot_surface 행 폴리곤 상한.")
    ap.add_argument("--ccount", type=int, default=40, help="[3d] plot_surface 열 폴리곤 상한.")
    ap.add_argument("--target-fps", type=float, default=0.0, help="렌더 상한 fps(0=무제한).")
    ap.add_argument("--no-render", action="store_true", help="창 없이 모델별 fps/수치만 출력.")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="[--no-render] blank 캡처 후 이 시간(초)만 측정하고 자동 종료 + 최종 요약. 0=무한.")
    ap.add_argument("--headless", type=int, default=0, metavar="N",
                    help="디스플레이 없이 N프레임 처리 후 비교 스냅샷 저장.")
    ap.add_argument("--outdir", default="./digit_check_results")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = (device == "cuda") and not args.no_fp16
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    px_per_mm = args.px_per_mm if args.px_per_mm and args.px_per_mm > 0 else None
    tn_unit = "mm" if px_per_mm else "rel"
    print(f"디바이스: {device} | fp16: {use_fp16} | TouchNet 단위: {tn_unit}")

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
        print("[오류] DIGIT 센서를 찾을 수 없습니다. --serial 로 지정하세요.")
        sys.exit(1)
    os.makedirs(args.outdir, exist_ok=True)

    # ----- 모델 로드 -----
    print("TouchNet 로드 중...")
    tnet = models.TouchNet(load_pretrained=True, sensor_type=SensorType.DIGIT, root=args.touchnet_root)
    tnet.to(device).eval()
    if use_fp16:
        tnet = tnet.to(memory_format=torch.channels_last)

    dpt_models = []  # [(label, model), ...]
    for p in args.dpt_weights:
        label = os.path.splitext(os.path.basename(p))[0]
        print(f"DPT 로드 중: {label} ({p})")
        dpt_models.append((label, build_dpt(p, device=device)))

    # ----- 센서 + 그래버 -----
    sensor = DIGIT(serial)
    sensor.connect()
    sensor.flush_frames(n=args.warmup)
    grabber = FrameGrabber(sensor)
    grabber.start()
    print(f"센서 {serial} 연결 완료.\n무접촉 상태로 두세요. 기준(blank) 캡처...")
    time.sleep(1.0)
    blank = capture_blank(grabber)
    blank_t = touchnet_tensor(blank)
    blank_rgb = np.ascontiguousarray(blank[:, :, ::-1])
    dpt_bgs = [dpt_heightmap(m, blank_rgb, device=device, use_fp16=use_fp16) for _, m in dpt_models]
    print("blank 캡처 완료. 센서를 눌러보세요.\n")

    def compute(frame):
        """returns tn, tn_dt, [dp_i], [dp_dt_i]"""
        ta = time.time()
        tn = touchnet_depth(tnet, frame, blank_t, device, use_fp16, px_per_mm)
        tb = time.time()
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        dps, dts = [], []
        for (label, m), bg in zip(dpt_models, dpt_bgs):
            tc = time.time()
            hm = dpt_heightmap(m, rgb, device=device, use_fp16=use_fp16)
            dps.append(np.clip(hm - bg, 0, None))
            dts.append(time.time() - tc)
        return tn, (tb - ta), dps, dts

    labels = [lbl for lbl, _ in dpt_models]

    # ---------- no-render: fps/수치만 ----------
    if no_render:
        print("no-render: 모델별 fps/추정값 출력 (Ctrl-C 종료)\n")
        min_dt = (1.0 / args.target_fps) if args.target_fps > 0 else 0.0
        fc = 0; tn_time = 0.0; dp_times = [0.0] * len(dpt_models)
        t0 = time.time(); last = t0
        try:
            while True:
                frame = grabber.latest()
                if frame is None:
                    time.sleep(0.005); continue
                tn, tn_dt, dps, dts = compute(frame)
                fc += 1; tn_time += tn_dt
                for i, d in enumerate(dts):
                    dp_times[i] += d
                if fc % 5 == 0:
                    parts = [f"TouchNet {fc / tn_time:5.1f}fps(max={tn.max():.3f}{tn_unit})"]
                    for lbl, dpt_t, dp in zip(labels, dp_times, dps):
                        parts.append(f"{lbl} {fc / dpt_t:5.1f}fps(max={dp.max():.0f})")
                    parts.append(f"loop {fc / (time.time() - t0):5.1f}fps")
                    print(" | ".join(parts))
                if args.duration and (time.time() - t0) >= args.duration:
                    break
                if min_dt:
                    d = time.time() - last
                    if d < min_dt:
                        time.sleep(min_dt - d)
                last = time.time()
        except KeyboardInterrupt:
            print("\n중단됨.")
        finally:
            grabber.stop(); time.sleep(0.05)
            try: sensor.disconnect()
            except Exception: pass
            print("연결 해제.")
        if fc > 0:
            print("\n=== 추론 속도 비교 요약 ===")
            print(f"측정 프레임 수: {fc}  (경과 {time.time() - t0:.1f}s)")
            rows = [("TouchNet(CNN)", 1000 * tn_time / fc, fc / tn_time)]
            for lbl, dpt_t in zip(labels, dp_times):
                rows.append((f"{lbl}(ViT)", 1000 * dpt_t / fc, fc / dpt_t))
            for name, ms, fps in rows:
                print(f"{name:16s}: {ms:6.2f} ms/frame  -> {fps:6.1f} fps")
            fastest = min(rows, key=lambda r: r[1])
            slowest = max(rows, key=lambda r: r[1])
            print(f"=> {fastest[0]} 가 가장 빠름 (가장 느린 {slowest[0]} 대비 {slowest[1] / fastest[1]:.2f}배)")
        return

    # ---------- 렌더: [카메라 | TouchNet | DPT...] ----------
    tn0, _, dps0, _ = compute(grabber.latest())
    tn_ema = max(tn0.max(), 1e-3)
    dp_emas = [max(d.max(), 1.0) for d in dps0]
    state = {"quit": False, "reblank": False}

    def on_key(ev):
        if ev.key in ("q", "escape"):
            state["quit"] = True
        elif ev.key == "r":
            state["reblank"] = True

    is3d = args.mode == "3d"
    n_dpt = len(dpt_models)
    ncols = 2 + n_dpt
    fig = plt.figure(figsize=(5 * ncols, 6))
    ax0 = fig.add_subplot(1, ncols, 1); ax0.set_title(f"DIGIT {serial} camera"); ax0.axis("off")
    im_raw = ax0.imshow(np.ascontiguousarray(grabber.latest()[:, :, ::-1]))

    im_tn = None; surf_tn = None; im_dps = [None] * n_dpt; surf_dps = [None] * n_dpt
    if is3d:
        H, W = downsample(tn0, args.downsample).shape
        Yg, Xg = np.mgrid[0:H, 0:W]
        ax_tn = fig.add_subplot(1, ncols, 2, projection="3d")
        ax_tn.set_title(f"TouchNet ({tn_unit})"); ax_tn.set_zlabel(tn_unit)
        ax_dps = []
        for i, lbl in enumerate(labels):
            a = fig.add_subplot(1, ncols, 3 + i, projection="3d")
            a.set_title(f"DPT:{lbl} (rel)"); a.set_zlabel("Δhm")
            ax_dps.append(a)
    else:
        ax_tn = fig.add_subplot(1, ncols, 2); ax_tn.set_title(f"TouchNet ({tn_unit})"); ax_tn.axis("off")
        im_tn = ax_tn.imshow(tn0, cmap="viridis", vmin=0, vmax=tn_ema)
        fig.colorbar(im_tn, ax=ax_tn, fraction=0.046, pad=0.04)
        ax_dps = []
        for i, lbl in enumerate(labels):
            a = fig.add_subplot(1, ncols, 3 + i); a.set_title(f"DPT:{lbl} (rel)"); a.axis("off")
            im_dps[i] = a.imshow(dps0[i], cmap="magma", vmin=0, vmax=dp_emas[i])
            fig.colorbar(im_dps[i], ax=a, fraction=0.046, pad=0.04)
            ax_dps.append(a)
    fig.suptitle("TouchNet (CNN) vs DPT (ViT) — live depth", fontsize=14)

    if not headless:
        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.ion(); plt.show(block=False)

    min_dt = (1.0 / args.target_fps) if args.target_fps > 0 else 0.0
    fc = 0; tn_time = 0.0; dp_times = [0.0] * n_dpt; t0 = time.time(); last = t0
    try:
        while True:
            frame = grabber.latest()
            if frame is None:
                if grabber._err:
                    raise grabber._err
                time.sleep(0.005); continue
            tn, tn_dt, dps, dts = compute(frame)
            fc += 1; tn_time += tn_dt
            for i, d in enumerate(dts):
                dp_times[i] += d
            tn_ema = 0.9 * tn_ema + 0.1 * max(tn.max(), 1e-3)
            for i in range(n_dpt):
                dp_emas[i] = 0.9 * dp_emas[i] + 0.1 * max(dps[i].max(), 1.0)

            im_raw.set_data(np.ascontiguousarray(frame[:, :, ::-1]))
            if is3d:
                if surf_tn is not None:
                    surf_tn.remove()
                surf_tn = ax_tn.plot_surface(Xg, Yg, downsample(tn, args.downsample), cmap="viridis",
                                             linewidth=0, antialiased=False, vmin=0, vmax=tn_ema,
                                             rcount=args.rcount, ccount=args.ccount)
                ax_tn.set_zlim(0, tn_ema)
                for i in range(n_dpt):
                    if surf_dps[i] is not None:
                        surf_dps[i].remove()
                    surf_dps[i] = ax_dps[i].plot_surface(Xg, Yg, downsample(dps[i], args.downsample),
                                                         cmap="magma", linewidth=0, antialiased=False,
                                                         vmin=0, vmax=dp_emas[i],
                                                         rcount=args.rcount, ccount=args.ccount)
                    ax_dps[i].set_zlim(0, dp_emas[i])
            else:
                im_tn.set_data(tn); im_tn.set_clim(0, tn_ema)
                for i in range(n_dpt):
                    im_dps[i].set_data(dps[i]); im_dps[i].set_clim(0, dp_emas[i])

            if fc % 20 == 0:
                parts = [f"TouchNet {fc / tn_time:5.1f}fps"]
                for lbl, dpt_t in zip(labels, dp_times):
                    parts.append(f"{lbl} {fc / dpt_t:5.1f}fps")
                parts.append(f"loop {fc / (time.time() - t0):5.1f}fps")
                print("  " + " | ".join(parts))

            if headless:
                if fc >= args.headless:
                    break
            else:
                fig.canvas.draw_idle(); fig.canvas.flush_events()
                if state["quit"] or not plt.fignum_exists(fig.number):
                    break
                if state["reblank"]:
                    state["reblank"] = False
                    print("blank 재캡처...")
                    blank = capture_blank(grabber)
                    blank_t = touchnet_tensor(blank)
                    blank_rgb = np.ascontiguousarray(blank[:, :, ::-1])
                    dpt_bgs = [dpt_heightmap(m, blank_rgb, device=device, use_fp16=use_fp16)
                               for _, m in dpt_models]
                    print("blank 갱신 완료.")
            if min_dt:
                d = time.time() - last
                if d < min_dt:
                    time.sleep(min_dt - d)
            last = time.time()
    except KeyboardInterrupt:
        print("\n중단됨.")
    finally:
        grabber.stop(); time.sleep(0.05)
        try: sensor.disconnect()
        except Exception: pass
        print("연결 해제.")

    if headless:
        snap = os.path.join(args.outdir, f"{serial}_compare_{args.mode}.png")
        fig.savefig(snap, dpi=130, bbox_inches="tight")
        print(f"비교 스냅샷 저장: {snap}")
    print("=== 완료 ===")


if __name__ == "__main__":
    main()
