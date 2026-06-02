"""
TouchNet 추론 속도 최적화 비교: baseline vs (#2 GPU Poisson) vs (#2 + #3 torch.compile)

  A) baseline      : 모델 forward(FP16) -> CPU 로 내려 scipy.fftpack Poisson 적분 (현재 방식)
  B) +GPU Poisson  : forward 출력을 GPU 에 둔 채 torch matmul 로 Poisson 적분(.cpu() 1회만)
  C) B + compile   : torch.compile(model) 까지 적용

GPU Poisson 은 scipy 의 DST/IDST 를 '항등행렬에 적용'해 만든 변환행렬로 matmul 하므로
scipy fast_poisson 과 수치적으로 동일하다(스크립트가 최대 오차를 출력해 검증).

실행:
  conda activate py3dcal
  python examples/benchmark_touchnet_opts.py --serial D21424 --iters 60
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.fftpack import dst, idst

import py3DCal as p3d
from py3DCal import DIGIT, models, SensorType
from py3DCal.model_training.lib.add_coordinate_embeddings import add_coordinate_embeddings
from py3DCal.model_training.lib.fast_poisson import fast_poisson


class GpuPoisson:
    """scipy fast_poisson 와 동일한 결과를 GPU matmul 로 계산.

    scipy.fftpack.dst/idst(norm='ortho') 를 항등행렬에 적용해 변환행렬을 만들어 두면
    dst(g, axis) == (행렬 @ g) 가 되어 GPU 에서 그대로 재현된다.
    """
    def __init__(self, H, W, device):
        Nh, Nw = H - 2, W - 2
        Mw = dst(np.eye(Nw), axis=0, norm="ortho")     # dst along width
        Mh = dst(np.eye(Nh), axis=0, norm="ortho")     # dst along height
        IMw = idst(np.eye(Nw), axis=0, norm="ortho")
        IMh = idst(np.eye(Nh), axis=0, norm="ortho")
        f32 = torch.float32
        self.Mw = torch.tensor(Mw, dtype=f32, device=device)
        self.Mh = torch.tensor(Mh, dtype=f32, device=device)
        self.IMw = torch.tensor(IMw, dtype=f32, device=device)
        self.IMh = torch.tensor(IMh, dtype=f32, device=device)
        x = torch.arange(1, W - 1, device=device, dtype=f32)   # 1..W-2  (Nw,)
        y = torch.arange(1, H - 1, device=device, dtype=f32)   # 1..H-2  (Nh,)
        denomx = 2 * torch.cos(np.pi * x / (W - 1)) - 2
        denomy = 2 * torch.cos(np.pi * y / (H - 1)) - 2
        self.denom = denomy[:, None] + denomx[None, :]         # (Nh, Nw)

    def __call__(self, Gx, Gy):
        # Gx, Gy: (H, W) float32 GPU 텐서
        Gxx = Gx[1:-1, 1:-1] - Gx[1:-1, :-2]
        Gyy = Gy[1:-1, 1:-1] - Gy[:-2, 1:-1]
        g = Gxx + Gyy                                  # (Nh, Nw)
        g1 = g @ self.Mw.t()                           # dst along width
        g2 = self.Mh @ g1                              # dst along height
        out = g2 / self.denom
        gx = out @ self.IMw.t()                        # idst along width
        gxy = self.IMh @ gx                            # idst along height
        return F.pad(gxy, (1, 1, 1, 1))                # (H, W)


def main():
    ap = argparse.ArgumentParser(description="TouchNet GPU-Poisson + torch.compile 벤치마크")
    ap.add_argument("--serial", default=None)
    ap.add_argument("--root", default="./digit_weights")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--no-fp16", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda" and not args.no_fp16
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"디바이스: {device} | fp16: {use_fp16}")

    # 센서에서 입력 한 장 확보 (순수 compute 비교를 위해 동일 프레임 재사용)
    serial = args.serial
    if serial is None:
        from digit_interface import DigitHandler
        ds = DigitHandler.list_digits()
        serial = ds[0]["serial"] if ds else None
    if not serial:
        raise SystemExit("DIGIT 센서를 찾을 수 없습니다. --serial 지정 필요.")
    sensor = DIGIT(serial); sensor.connect(); sensor.flush_frames(50)
    frame = sensor.capture_image()
    sensor.disconnect()
    H, W = frame.shape[:2]
    print(f"센서 {serial} | 입력 {W}x{H}")

    # 모델
    tnet = models.TouchNet(load_pretrained=True, sensor_type=SensorType.DIGIT, root=args.root)
    tnet.to(device).eval()
    if use_fp16:
        tnet = tnet.to(memory_format=torch.channels_last)

    blank_t = torch.zeros(3, H, W)  # 비교 목적이라 blank=0 (속도엔 영향 없음)
    img = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float() / 255.0
    aug = add_coordinate_embeddings(img - blank_t).unsqueeze(0).to(device)
    if use_fp16:
        aug = aug.to(memory_format=torch.channels_last)

    gpu_poisson = GpuPoisson(H, W, device)

    def forward(model):
        with torch.no_grad():
            if use_fp16:
                with torch.autocast("cuda", dtype=torch.float16):
                    return model(aug)
            return model(aug)

    def depth_scipy(model):
        o = forward(model).squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        return np.clip(-fast_poisson(o[:, :, 0], o[:, :, 1]), 0, None)

    def depth_gpu(model):
        o = forward(model).squeeze(0).float()          # (2,H,W) GPU
        d = gpu_poisson(o[0], o[1])
        return np.clip(-d.cpu().numpy(), 0, None)

    # ---- 정확성 검증: GPU Poisson vs scipy ----
    da = depth_scipy(tnet)
    db = depth_gpu(tnet)
    max_abs = np.abs(da - db).max()
    rel = max_abs / (np.abs(da).max() + 1e-9)
    print(f"\n[검증] GPU Poisson vs scipy 최대 절대오차={max_abs:.3e} (상대 {rel:.2e}) "
          f"-> {'OK 일치' if rel < 1e-3 else '주의: 불일치'}")

    # ---- torch.compile 모델 ----
    print("\ntorch.compile 준비 중(첫 호출 컴파일)...")
    tnet_c = torch.compile(tnet)
    for _ in range(3):
        depth_gpu(tnet_c)   # 컴파일 워밍업

    def bench(fn, model, label):
        for _ in range(args.warmup):
            fn(model)
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        for _ in range(args.iters):
            fn(model)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t) / args.iters
        print(f"{label:34s}: {1000*dt:6.2f} ms/frame -> {1/dt:6.1f} fps")
        return dt

    print(f"\n=== TouchNet 추론 속도 비교 ({args.iters} iters, 동일 프레임) ===")
    a = bench(depth_scipy, tnet,   "A) baseline (scipy Poisson, CPU)")
    b = bench(depth_gpu,   tnet,   "B) +GPU Poisson (torch matmul)")
    c = bench(depth_gpu,   tnet_c, "C) B + torch.compile")
    print("-" * 60)
    print(f"B 가 A 대비 {a/b:.2f}배, C 가 A 대비 {a/c:.2f}배 빠름")


if __name__ == "__main__":
    main()
