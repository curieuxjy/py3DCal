"""
TouchNet "아무 최적화 없음(BASE)" vs "전부 적용(OPT)" 실시간 스트리밍 비교.

  BASE : FP32, 풀해상도(320x240), scipy.fftpack Poisson(CPU), compile 미사용
  OPT  : #1 저해상도(--infer-scale) + #2 GPU Poisson(torch matmul) + #3 torch.compile + FP16/channels_last
         * 저해상도에서도 좌표 임베딩은 원본 픽셀 범위(1..W,1..H)로 생성해 학습 분포와 맞춤

패널: [DIGIT 카메라 | BASE 깊이 | OPT 깊이]  — 같은 프레임으로 동기 갱신. 모델별 fps 를 로그에 출력.

모드:
  (기본)       실시간 렌더(라이브). q/ESC 종료, r blank 재캡처.
  --no-render  창 없이 모델별 fps/수치 + (--duration 후) 속도·결과 요약.
  --headless N 디스플레이 없이 N프레임 후 비교 스냅샷 저장.
  --mode 3d    깊이를 3D 표면으로.

실행:
  conda activate py3dcal
  python examples/stream_compare_touchnet_opt.py --serial D21424
  python examples/stream_compare_touchnet_opt.py --serial D21424 --no-render --duration 10
  # 자동화: bash examples/benchmark_touchnet_opt.sh D21424
"""
import argparse
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


def detect_first_serial():
    if DigitHandler is None:
        return None
    for d in DigitHandler.list_digits():
        s = d.get("serial") if isinstance(d, dict) else None
        if s:
            return s
    return None


class FrameGrabber(threading.Thread):
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


def main():
    ap = argparse.ArgumentParser(description="TouchNet BASE vs OPT 실시간 비교")
    ap.add_argument("--serial", default=None)
    ap.add_argument("--root", default="./digit_weights")
    ap.add_argument("--infer-scale", type=int, default=2, help="OPT 저해상도 배수(2=절반).")
    ap.add_argument("--px-per-mm", type=float, default=15.0, help="px->mm 변환. 0이면 상대단위.")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--mode", choices=["2d", "3d"], default="2d")
    ap.add_argument("--downsample", type=int, default=4, help="[3d] 표면 다운샘플 배수.")
    ap.add_argument("--rcount", type=int, default=40)
    ap.add_argument("--ccount", type=int, default=40)
    ap.add_argument("--target-fps", type=float, default=0.0)
    ap.add_argument("--no-render", action="store_true", help="창 없이 fps/수치만 출력.")
    ap.add_argument("--duration", type=float, default=0.0, help="[--no-render] N초 측정 후 자동 종료+요약.")
    ap.add_argument("--headless", type=int, default=0, metavar="N", help="N프레임 후 스냅샷 저장.")
    ap.add_argument("--outdir", default="./digit_check_results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    px = args.px_per_mm if args.px_per_mm > 0 else None
    unit = "mm" if px else "rel"
    print(f"디바이스: {device} | infer-scale: 1/{args.infer_scale} | 단위: {unit}")

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
        print("[오류] DIGIT 센서를 찾을 수 없습니다. --serial 지정 필요."); sys.exit(1)
    os.makedirs(args.outdir, exist_ok=True)

    # ----- 모델: BASE(fp32) / OPT(fp16+channels_last+compile) -----
    print("TouchNet 로드 중...")
    base_model = models.TouchNet(load_pretrained=True, sensor_type=SensorType.DIGIT, root=args.root)
    base_model.to(device).eval()
    opt_model = models.TouchNet(load_pretrained=True, sensor_type=SensorType.DIGIT, root=args.root)
    opt_model.to(device).eval().to(memory_format=torch.channels_last)
    opt_model_c = torch.compile(opt_model)

    # ----- 센서 + 그래버 -----
    sensor = DIGIT(serial); sensor.connect(); sensor.flush_frames(args.warmup)
    grabber = FrameGrabber(sensor); grabber.start()
    print(f"센서 {serial} 연결 완료.\n무접촉 상태로 두세요. blank 캡처...")
    time.sleep(1.0)
    blank = capture_blank(grabber)
    H, W = blank.shape[:2]
    s = args.infer_scale
    Hl, Wl = H // s, W // s
    gpu_poisson = GpuPoisson(Hl, Wl, device)

    blank_full = torch.from_numpy(np.ascontiguousarray(blank)).permute(2, 0, 1).float() / 255.0
    blank_low = cv2.resize(blank, (Wl, Hl), interpolation=cv2.INTER_AREA)
    blank_low_t = torch.from_numpy(np.ascontiguousarray(blank_low)).permute(2, 0, 1).float() / 255.0
    print("blank 캡처 완료. 센서를 눌러보세요.\n")

    def depth_base(frame):
        img = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float() / 255.0
        aug = add_coordinate_embeddings(img - blank_full).unsqueeze(0).to(device)
        with torch.no_grad():
            out = base_model(aug)
        o = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        d = np.clip(-fast_poisson(o[:, :, 0], o[:, :, 1]), 0, None)
        return d / px if px else d

    def depth_opt(frame):
        fl = cv2.resize(frame, (Wl, Hl), interpolation=cv2.INTER_AREA)
        img = torch.from_numpy(np.ascontiguousarray(fl)).permute(2, 0, 1).float() / 255.0
        aug = coord_embed_scaled(img - blank_low_t, H, W).unsqueeze(0).to(device)
        aug = aug.to(memory_format=torch.channels_last)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float16):
                out = opt_model_c(aug)
        o = out.squeeze(0).float()
        d = torch.clamp(-gpu_poisson(o[0], o[1]), min=0)
        d = F.interpolate(d[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        d = d.cpu().numpy()
        return d / px if px else d

    def compute(frame):
        ta = time.time(); db = depth_base(frame); tb = time.time()
        do = depth_opt(frame); tc = time.time()
        return db, do, (tb - ta), (tc - tb)

    # 워밍업: torch.compile + cudnn autotune(BASE/OPT 모두) — 측정 안정화
    print("워밍업(compile + cudnn autotune)...")
    for _ in range(5):
        compute(grabber.latest())

    # ---------- no-render ----------
    if no_render:
        print("no-render: BASE/OPT fps·수치 출력 (Ctrl-C 종료)\n")
        fc = 0; bt = 0.0; ot = 0.0; t0 = time.time(); last = t0
        db = do = None
        try:
            while True:
                frame = grabber.latest()
                if frame is None:
                    time.sleep(0.005); continue
                db, do, bdt, odt = compute(frame)
                fc += 1; bt += bdt; ot += odt
                if fc % 5 == 0:
                    print(f"BASE {fc/bt:6.1f} fps (max={db.max():.3f}{unit}) | "
                          f"OPT {fc/ot:6.1f} fps (max={do.max():.3f}{unit}) | "
                          f"loop {fc/(time.time()-t0):5.1f} fps")
                if args.duration and (time.time() - t0) >= args.duration:
                    break
                last = time.time()
        except KeyboardInterrupt:
            print("\n중단됨.")
        finally:
            grabber.stop(); time.sleep(0.05)
            try: sensor.disconnect()
            except Exception: pass
            print("연결 해제.")
        if fc > 0:
            print("\n=== 속도 비교 요약 ===")
            print(f"BASE (fp32, full-res, scipy) : {1000*bt/fc:6.2f} ms -> {fc/bt:6.1f} fps")
            print(f"OPT  (fp16, 1/{s}, GPU+compile): {1000*ot/fc:6.2f} ms -> {fc/ot:6.1f} fps")
            print(f"=> OPT 가 BASE 대비 {bt/ot:.2f}배 빠름")
            if db is not None:
                a, b = db.flatten(), do.flatten()
                if a.std() > 1e-9 and b.std() > 1e-9:
                    corr = float(np.corrcoef(a, b)[0, 1])
                    k = float((a @ b) / (b @ b + 1e-12))
                    print(f"[마지막 프레임] 결과 corr={corr:.3f}, 스칼라정합 k={k:.3f}")
                else:
                    print("[결과] 무접촉(평탄)이라 형상 비교 생략. 눌러서 다시 측정하세요.")
        return

    # ---------- 렌더 ----------
    db0, do0, _, _ = compute(grabber.latest())
    b_ema = max(db0.max(), 1e-3); o_ema = max(do0.max(), 1e-3)
    state = {"quit": False, "reblank": False}

    def on_key(ev):
        if ev.key in ("q", "escape"): state["quit"] = True
        elif ev.key == "r": state["reblank"] = True

    is3d = args.mode == "3d"
    fig = plt.figure(figsize=(16, 6))
    ax0 = fig.add_subplot(1, 3, 1); ax0.set_title(f"DIGIT {serial} camera"); ax0.axis("off")
    im_raw = ax0.imshow(np.ascontiguousarray(grabber.latest()[:, :, ::-1]))
    im_b = im_o = surf_b = surf_o = None
    if is3d:
        Hd, Wd = downsample(db0, args.downsample).shape
        Yg, Xg = np.mgrid[0:Hd, 0:Wd]
        ax_b = fig.add_subplot(1, 3, 2, projection="3d"); ax_b.set_title(f"BASE depth ({unit})")
        ax_o = fig.add_subplot(1, 3, 3, projection="3d"); ax_o.set_title(f"OPT depth ({unit})")
    else:
        ax_b = fig.add_subplot(1, 3, 2); ax_b.set_title(f"BASE depth ({unit})"); ax_b.axis("off")
        im_b = ax_b.imshow(db0, cmap="viridis", vmin=0, vmax=b_ema); fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
        ax_o = fig.add_subplot(1, 3, 3); ax_o.set_title(f"OPT depth ({unit})"); ax_o.axis("off")
        im_o = ax_o.imshow(do0, cmap="viridis", vmin=0, vmax=o_ema); fig.colorbar(im_o, ax=ax_o, fraction=0.046, pad=0.04)
    fig.suptitle(f"TouchNet  BASE (full-res)  vs  OPT (1/{s} + GPU-Poisson + compile)", fontsize=13)

    if not headless:
        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.ion(); plt.show(block=False)

    min_dt = (1.0 / args.target_fps) if args.target_fps > 0 else 0.0
    fc = 0; bt = 0.0; ot = 0.0; t0 = time.time(); last = t0
    try:
        while True:
            frame = grabber.latest()
            if frame is None:
                if grabber._err: raise grabber._err
                time.sleep(0.005); continue
            db, do, bdt, odt = compute(frame)
            fc += 1; bt += bdt; ot += odt
            b_ema = 0.9 * b_ema + 0.1 * max(db.max(), 1e-3)
            o_ema = 0.9 * o_ema + 0.1 * max(do.max(), 1e-3)
            im_raw.set_data(np.ascontiguousarray(frame[:, :, ::-1]))
            if is3d:
                if surf_b is not None: surf_b.remove()
                surf_b = ax_b.plot_surface(Xg, Yg, downsample(db, args.downsample), cmap="viridis",
                                           linewidth=0, antialiased=False, vmin=0, vmax=b_ema,
                                           rcount=args.rcount, ccount=args.ccount)
                ax_b.set_zlim(0, b_ema)
                if surf_o is not None: surf_o.remove()
                surf_o = ax_o.plot_surface(Xg, Yg, downsample(do, args.downsample), cmap="viridis",
                                           linewidth=0, antialiased=False, vmin=0, vmax=o_ema,
                                           rcount=args.rcount, ccount=args.ccount)
                ax_o.set_zlim(0, o_ema)
            else:
                im_b.set_data(db); im_b.set_clim(0, b_ema)
                im_o.set_data(do); im_o.set_clim(0, o_ema)
            if fc % 15 == 0:
                print(f"  BASE {fc/bt:5.1f}fps | OPT {fc/ot:5.1f}fps | loop {fc/(time.time()-t0):5.1f}fps "
                      f"| max B={db.max():.3f} O={do.max():.3f} {unit}")
            if headless:
                if fc >= args.headless: break
            else:
                fig.canvas.draw_idle(); fig.canvas.flush_events()
                if state["quit"] or not plt.fignum_exists(fig.number): break
                if state["reblank"]:
                    state["reblank"] = False
                    print("blank 재캡처...")
                    blank = capture_blank(grabber)
                    blank_full = torch.from_numpy(np.ascontiguousarray(blank)).permute(2, 0, 1).float() / 255.0
                    bl = cv2.resize(blank, (Wl, Hl), interpolation=cv2.INTER_AREA)
                    blank_low_t = torch.from_numpy(np.ascontiguousarray(bl)).permute(2, 0, 1).float() / 255.0
                    print("blank 갱신 완료.")
            if min_dt:
                d = time.time() - last
                if d < min_dt: time.sleep(min_dt - d)
            last = time.time()
    except KeyboardInterrupt:
        print("\n중단됨.")
    finally:
        grabber.stop(); time.sleep(0.05)
        try: sensor.disconnect()
        except Exception: pass
        print("연결 해제.")

    if headless:
        snap = os.path.join(args.outdir, f"{serial}_touchnet_opt_{args.mode}.png")
        fig.savefig(snap, dpi=130, bbox_inches="tight")
        print(f"스냅샷 저장: {snap}")
    print("=== 완료 ===")


if __name__ == "__main__":
    main()
