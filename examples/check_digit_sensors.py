"""
보유한 DIGIT 센서 여러 개(기본 4개)에 대해 py3DCal 사전학습 TouchNet(DIGIT) 모델을
실행해 깊이맵(depth map)이 정상적으로 생성되는지 한 번에 확인하는 스크립트.

동작:
  1) DigitHandler.list_digits() 로 연결된 DIGIT 센서를 자동 탐지 (또는 --serials 로 지정).
  2) 각 센서를 연결하고 워밍업 후, 접촉이 없는 상태에서 'blank(기준)' 프레임을 캡처.
  3) 사용자가 센서에 물체를 누른 뒤, 'contact(접촉)' 프레임을 캡처.
  4) Zenodo 에서 받은 DIGIT 사전학습 가중치로 각 센서의 깊이맵을 계산.
  5) 센서별 [접촉 이미지 / 깊이맵] 을 한 그림에 모아 저장(및 가능하면 화면 표시).

중요:
  - 캡처/전처리는 py3DCal 의 DIGIT 래퍼와 get_depthmap 을 그대로 사용한다.
    (DIGIT.capture_image() 가 학습 때와 동일한 cv2.flip 을 적용하므로 색/방향 정합이 보장됨)
  - 실제 DIGIT 하드웨어가 연결되어 있어야 한다.

실행 예:
  conda run -n py3dcal python examples/check_digit_sensors.py
  conda run -n py3dcal python examples/check_digit_sensors.py --serials D20753 D20765 D21424 D20966
"""
import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
from PIL import Image
import matplotlib

import py3DCal as p3d
from py3DCal import DIGIT, models, SensorType

try:
    from digit_interface import DigitHandler
except Exception as e:  # pragma: no cover - 하드웨어/라이브러리 미설치 환경
    DigitHandler = None
    _DIGIT_IMPORT_ERR = e


def detect_serials():
    """연결된 DIGIT 센서의 시리얼 번호 목록을 중복 없이 반환."""
    if DigitHandler is None:
        print(f"[오류] digit_interface 를 불러올 수 없습니다: {_DIGIT_IMPORT_ERR}")
        return []
    digits = DigitHandler.list_digits()
    serials = []
    for d in digits:
        s = d.get("serial") if isinstance(d, dict) else None
        if s and s not in serials:
            serials.append(s)
    return serials


def connect_sensors(serials, fps_warmup=50):
    """시리얼 목록으로 py3DCal DIGIT 센서들을 연결하고 워밍업."""
    sensors = {}
    for s in serials:
        try:
            sensor = DIGIT(s)
            sensor.connect()
            # 카메라 자동노출이 안정화되도록 초기 프레임을 버린다.
            sensor.flush_frames(n=fps_warmup)
            sensors[s] = sensor
            print(f"  [연결됨] {s}")
        except Exception as e:
            print(f"  [연결 실패] {s}: {e}")
    return sensors


def capture_all(sensors, n_flush=30):
    """모든 센서에서 한 프레임씩 캡처해 {serial: ndarray} 로 반환.

    V4L2/OpenCV 는 내부 버퍼에 오래된 프레임을 쌓아두므로, 캡처 직전
    충분히 많은 프레임을 버려야(=버퍼 드레인) 현재 상태(접촉 등)가 반영된
    프레임을 얻을 수 있다. 기본 30프레임(≈ 30fps 기준 1초).
    """
    frames = {}
    for s, sensor in sensors.items():
        sensor.flush_frames(n=n_flush)  # 카메라 버퍼 드레인
        frames[s] = sensor.capture_image()
    return frames


def save_frame(frame, path):
    """py3DCal 데이터 수집과 동일하게 PIL 로 저장(전처리 정합 유지)."""
    Image.fromarray(frame).save(path)


def main():
    parser = argparse.ArgumentParser(description="DIGIT 센서들 사전학습 깊이맵 확인")
    parser.add_argument("--serials", nargs="*", default=None,
                        help="확인할 DIGIT 시리얼 번호들. 생략 시 자동 탐지.")
    parser.add_argument("--expected", type=int, default=4,
                        help="기대하는 센서 개수(경고용). 기본 4.")
    parser.add_argument("--root", default="./digit_weights",
                        help="사전학습 가중치(.pth) 저장/로드 디렉토리.")
    parser.add_argument("--outdir", default="./digit_check_results",
                        help="캡처 이미지와 결과 그림 저장 디렉토리.")
    parser.add_argument("--device", default=None,
                        help="추론 디바이스. 기본: cuda 가능하면 cuda, 아니면 cpu.")
    parser.add_argument("--warmup", type=int, default=50,
                        help="센서 연결 후 버릴 프레임 수.")
    parser.add_argument("--no-show", action="store_true",
                        help="화면 표시 없이 파일로만 저장(헤드리스 환경).")
    parser.add_argument("--auto", action="store_true",
                        help="대화형 입력 없이 자동 진행(blank 캡처 후 --delay 초 뒤 contact 캡처).")
    parser.add_argument("--delay", type=float, default=3.0,
                        help="--auto 모드에서 blank 와 contact 캡처 사이 대기 시간(초).")
    parser.add_argument("--flush", type=int, default=30,
                        help="캡처 직전 버릴 프레임 수(카메라 버퍼 드레인). 접촉이 안 잡히면 늘리세요.")
    args = parser.parse_args()

    def prompt(msg):
        """대화형이면 입력을 기다리고, --auto 면 안내만 출력하고 진행."""
        if args.auto:
            print(msg + "  (--auto: 자동 진행)")
            return
        try:
            input(msg)
        except EOFError:
            print(
                "\n[오류] 표준 입력(stdin)이 연결되어 있지 않아 Enter 입력을 받을 수 없습니다.\n"
                "  대화형으로 실행하려면 아래 중 하나를 사용하세요:\n"
                "    1) conda activate py3dcal && python examples/check_digit_sensors.py\n"
                "    2) conda run --no-capture-output -n py3dcal python examples/check_digit_sensors.py\n"
                "  또는 입력 없이 자동 진행하려면 --auto 옵션을 추가하세요:\n"
                "    conda run -n py3dcal python examples/check_digit_sensors.py --auto --delay 5\n"
            )
            sys.exit(2)

    # 디바이스 결정
    import torch
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"디바이스: {device} (cuda available: {torch.cuda.is_available()})")

    # 헤드리스면 비대화형 백엔드
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 센서 시리얼 결정
    serials = args.serials if args.serials else detect_serials()
    print(f"\n대상 DIGIT 센서: {serials}")
    if not serials:
        print("[오류] 확인할 DIGIT 센서가 없습니다. 연결 상태를 확인하거나 --serials 로 지정하세요.")
        sys.exit(1)
    if len(serials) != args.expected:
        print(f"[경고] {args.expected}개를 기대했지만 {len(serials)}개가 지정/탐지되었습니다. 계속 진행합니다.")

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.root, exist_ok=True)

    # 사전학습 모델 로드 (없으면 Zenodo 에서 자동 다운로드)
    print("\n사전학습 DIGIT 가중치 로드 중...")
    model = models.TouchNet(load_pretrained=True, sensor_type=SensorType.DIGIT, root=args.root)
    model.to(device)
    model.eval()

    # 센서 연결
    print("\n센서 연결 중...")
    sensors = connect_sensors(serials, fps_warmup=args.warmup)
    if not sensors:
        print("[오류] 연결된 센서가 없습니다.")
        sys.exit(1)

    try:
        # 1) blank(무접촉) 기준 프레임
        print("\n[단계 1/2] 모든 센서에서 손/물체를 떼고 무접촉 상태로 두세요.")
        prompt("준비되면 Enter 를 눌러 blank(기준) 프레임을 캡처합니다...")
        blanks = capture_all(sensors, n_flush=args.flush)
        blank_paths = {}
        for s, f in blanks.items():
            p = os.path.join(args.outdir, f"{s}_blank.png")
            save_frame(f, p)
            blank_paths[s] = p
        print("  blank 캡처 완료.")

        # 2) contact(접촉) 프레임
        print("\n[단계 2/2] 각 센서에 물체를 눌러 접촉시키세요.")
        if args.auto:
            print(f"  (--auto: {args.delay}초 후 자동 캡처)")
            time.sleep(args.delay)
        else:
            input("준비되면 Enter 를 눌러 contact(접촉) 프레임을 캡처합니다...")
        contacts = capture_all(sensors, n_flush=args.flush)
        contact_paths = {}
        for s, f in contacts.items():
            p = os.path.join(args.outdir, f"{s}_contact.png")
            save_frame(f, p)
            contact_paths[s] = p
        print("  contact 캡처 완료.")

        # 접촉 강도 진단: contact 와 blank 의 차이 크기(센서가 눌림을 봤는지 확인)
        print("\n접촉 강도(contact-blank 평균 절대차, 클수록 강한 접촉):")
        for s in sensors:
            diff = np.abs(contacts[s].astype(np.float32) - blanks[s].astype(np.float32)).mean()
            flag = "  <- 접촉 약함/없음?" if diff < 2.0 else ""
            print(f"  [{s}] Δ={diff:.2f}{flag}")

        # 3) 센서별 깊이맵 계산
        print("\n깊이맵 계산 중...")
        depthmaps = {}
        for s in sensors:
            dm = p3d.get_depthmap(
                model=model,
                image_path=contact_paths[s],
                blank_image_path=blank_paths[s],
                device=device,
            )
            depthmaps[s] = dm
            print(f"  [{s}] depthmap shape={dm.shape}, min={dm.min():.3f}, max={dm.max():.3f}")

    finally:
        for s, sensor in sensors.items():
            try:
                sensor.disconnect()
            except Exception:
                pass
        print("\n센서 연결 해제 완료.")

    # 4) 결과 그림: 열=센서, 행=[접촉 이미지, 깊이맵]
    n = len(sensors)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8), squeeze=False)
    for i, s in enumerate(sensors):
        contact_rgb = np.asarray(Image.open(contact_paths[s]).convert("RGB"))
        axes[0][i].imshow(contact_rgb)
        axes[0][i].set_title(f"{s}\ncontact")
        axes[0][i].axis("off")

        im = axes[1][i].imshow(depthmaps[s], cmap="viridis")
        axes[1][i].set_title("depthmap")
        axes[1][i].axis("off")
        fig.colorbar(im, ax=axes[1][i], fraction=0.046, pad=0.04)

    fig.suptitle("DIGIT pretrained TouchNet depthmap check", fontsize=14)
    fig.tight_layout()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_png = os.path.join(args.outdir, f"digit_depthmap_check_{ts}.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\n결과 그림 저장: {out_png}")

    if not args.no_show:
        try:
            plt.show()
        except Exception as e:
            print(f"[정보] 화면 표시 불가(헤드리스로 추정): {e}. 파일만 저장됨.")

    print("\n=== 완료 ===")


if __name__ == "__main__":
    main()
