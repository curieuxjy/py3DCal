# 다중 DIGIT 동시 스트리밍 트러블슈팅 기록

`examples/all_digits_stream_touchnet_opt.py` 로 DIGIT 센서 4대를 **동시에** TouchNet OPT 깊이
추정 + 3D 시각화하면서 마주친 문제들과 해결 과정을 정리한다.

- 환경: DIGIT ×4 (D21422 / D21429 / D21424 / D21418), QVGA, CUDA, py3dcal conda env
- USB 토폴로지: 4대 모두 **하나의 USB 2.0 허브(한 호스트 컨트롤러)** 에 연결 (`lsusb -t` 로 확인)

---

## 요약 (TL;DR)

| # | 증상 | 진짜 원인 | 해결 |
|---|------|-----------|------|
| 1 | 다중 입력 시 지연 | 센서별 순차 forward + 센서별 `.cpu()` 동기화 | **배치 추론**(1회 forward + 1회 동기화), 캡처 스레드 분리 |
| 2 | 센서 패널에 다른 센서 영상이 **통째로** 섞임 | (의심) 동시 read 경쟁 → 사실은 아님 | 락은 효과 없었음 → 아래 #3 으로 수렴 |
| 3 | 한 센서 프레임 안에서 패치가 어긋남(torn frame) | 한 USB2 컨트롤러에 동일 UVC 4대 → isochronous 대역폭 과포화 | 카메라당 대역폭↓: **QVGA@30 강제** + 최종 **`uvcvideo quirks=128`** (아래 #3 상세) |
| 4 | `--cam-fps 30` 을 줘도 fps 가 안 내려감(=섞임 지속) | **드라이버가 OpenCV `CAP_PROP_FPS` 를 무시** + `S_FMT`(해상도 설정)가 간격을 60 으로 리셋 + plain open 이 VGA 강제 | **닫힌 상태 `v4l2-ctl --set-fmt+--set-parm` → 재오픈 → `cap.set(QVGA)`** (`_force_qvga_fps`) |
| 5 | 무접촉인데 depth 가 >1mm (노이즈) | 찢어진 프레임/MJPEG 압축잡음이 `fast_poisson` 적분으로 증폭 | 섞임 해결(=#3,#4) + MJPEG OFF(YUYV) + 입력 dead-band |
| 6 | 1대일 때와 비교해 화면 노이즈가 심함 | (측정값은 깨끗) **시각화 자동 스케일 붕괴** | **표시 스케일 하한 `--min-scale`** |

**최종 권장 실행** (먼저 커널 quirk 1회, 그다음 기본값 그대로):

```bash
# (1) 부팅/리부트 후 한 번: uvcvideo 가 예약하는 대역폭을 줄여 다중 동시 스트리밍 허용
sudo modprobe -r uvcvideo && sudo modprobe uvcvideo quirks=128

# (2) 실행 (기본값: YUYV, --cam-fps 30, --diff-thresh 0.04, --min-scale 1.0)
conda activate py3dcal
python examples/all_digits_stream_touchnet_opt.py
```

> `quirks=128` 적용 후 4대 동시 무접촉 측정이 `max≈0.001mm`(`--no-render`)로 깨끗해지고
> 섞임/노이즈가 거의 사라졌다. 완전 결정적 해법은 센서를 **서로 다른 USB 호스트 컨트롤러**로 분산.

---

## 1. 지연(latency) — 배치 추론

**증상:** 여러 센서를 받을수록 루프가 느려짐.

**원인:** 메인 루프가 센서마다 따로 모델 forward 를 돌리고, 매번 `.cpu()` 로 GPU→CPU
동기화를 했다. 4대면 forward 4회 + 동기화 4회.

**해결:**
- 캡처는 센서별 백그라운드 스레드(`FrameGrabber`)로 분리 → 카메라 I/O 가 추론과 겹침.
- 모든 DIGIT 이 같은 해상도/모델을 공유하므로 **한 배치 `(N,5,H,W)` 로 묶어 단일 forward**,
  Poisson 적분만 per-sample, 마지막에 **`.cpu()` 1회**.
- CPU→GPU 전송은 pinned memory + `non_blocking`.

---

## 2~3. 프레임 섞임 — USB 대역폭

**증상:** 처음엔 "센서 패널에 다른 센서 영상이 섞인다", 자세히 보니
**한 프레임 안에서 ~10% 는 다른 센서, ~90% 는 본인 센서** (torn/stitched frame).

**헛다리:** 동시 `read()` 경쟁이라 보고 공유 락으로 직렬화 → **효과 없음**.
(read 시점과 무관한 드라이버 레벨 문제였기 때문.)

**진단:**
- `serial → /dev/videoN` 매핑은 깨끗하고 중복 없음(연결 시 자동 출력 + 중복 검사).
  - 각 DIGIT 은 video 노드 2개 노출(짝수=capture, 홀수=metadata), `find_digit` 가 capture 선택.
- `lsusb -t`: 4대 모두 한 USB 2.0 허브 → 한 컨트롤러 대역폭 공유.

**원인:** 동일 USB 디스크립터(`ID_MODEL=DIGIT`)의 카메라들을 한 컨트롤러에서 raw YUYV 로
동시 스트리밍하면 isochronous 대역폭이 과포화되고, `uvcvideo` 가 한 프레임 버퍼에
옆 카메라 패킷을 섞어 채운다. **read/스레드/락으로는 못 고침.**

**방향:** 카메라당 대역폭을 줄인다(QVGA@30) + 드라이버 예약 대역폭을 줄인다(quirk). 단,
"30fps 로 내리는 것" 자체가 이 드라이버에선 함정투성이였다 — 아래 #4 가 그 전말이다.

> 이 DIGIT 카메라는 디스크립터에 **YUYV 포맷 하나뿐(MJPG 없음)** 이라(`v4l2-ctl --list-formats-ext`
> 로 확인) MJPEG 로 대역폭을 줄이는 길은 처음부터 불가능했다. 남은 건 fps 와 컨트롤러뿐.

---

## 4. "fps 가 안 내려간다" — uvcvideo/OpenCV fps 제어 함정 (이번 세션 핵심)

`--cam-fps 30` 을 줘도 연결 로그의 **실측이 ~61fps** 로 찍히며 섞임이 그대로였다. 원인을
하나씩 도구로 검증하며 좁혔다.

**증상 정밀화:** 섞임은 **한 센서 자기 프레임 안에서 패치들이 어긋나/뒤섞여** 보이는
torn-frame 이다. 다른 센서 내용이 아니라, isochronous 패킷이 잘못된 오프셋/순서로 같은
프레임 버퍼에 채워진 것. 그리고 한 번 desync 되면 그 상태로 **고착**된다(복구 안 됨).

**진단 도구:**
- `v4l2-ctl -d /dev/videoN --list-formats-ext` → 하드웨어 지원 포맷/간격. QVGA 320x240 은
  **30/60fps**, VGA 640x480 은 30/15fps 지원. **MJPG 없음.**
- `v4l2-ctl -d /dev/videoN --get-parm` → 현재 실제 간격. 스크립트 실행 중 60fps 확인.
- `lsusb -t` → DIGIT 4대 모두 **Bus 01 한 USB2 허브(480M)** 에 매달림. (다른 xhci 컨트롤러
  Bus 02 는 비어 있음 → 분산 가능.)
- `examples/_probe_fps.py` → 한 센서를 여러 방식으로 열어 **실측 fps + 해상도** 를 비교한
  일회성 프로브(원인 확정용; 본 스크립트와 별개).

**원인(3중):**
1. **드라이버가 OpenCV `cap.set(CAP_PROP_FPS, ...)` 를 무시한다.** 그래서 `digit_interface`
   의 `set_fps()` 도, 우리 설정도 간격을 못 바꿨다. 간격을 실제로 바꾸는 건
   `VIDIOC_S_PARM`(= `v4l2-ctl --set-parm`)뿐이고, **장치가 열리기 전(닫힌 상태)** 에 걸어야
   스트림 협상에 반영된다(열린 fd 가 있는 채로 걸면 streamon 에서 60 으로 되돌아감).
2. **`S_FMT`(= 해상도 설정/`set_resolution`)가 프레임 간격을 그 포맷의 기본 최대값으로
   리셋한다**(QVGA→60). `DIGIT.connect()` 가 내부에서 `set_resolution(QVGA)` 를 부르므로,
   connect 전에 set-parm 30 을 걸어도 connect 가 다시 60 으로 덮었다.
3. **OpenCV plain open 은 해상도를 VGA(640x480)로 강제한다.** 그래서 "닫힌 상태 set-parm 후
   그냥 재오픈" 하면 간격은 30 이 돼도 해상도가 VGA 로 올라가 **오히려 대역폭이 더 커졌다**
   (VGA@30 ≈ QVGA@60). 실제로 그 상태에서 D21424=12fps, D21418=9fps 로 굶었다.

**프로브로 확정한 해법 시퀀스 (`_force_qvga_fps`):**
1. `DIGIT.connect()` 가 연 내부 `cv2.VideoCapture`(`_Digit__dev`)를 **release(닫기)**.
2. 닫힌 상태에서 한 번에:
   `v4l2-ctl -d dev --set-fmt-video=width=320,height=240,pixelformat=YUYV --set-parm=30`
   → 포맷(QVGA)과 간격(30)을 함께 박는다(S_FMT 가 간격을 리셋하므로 둘을 같이).
3. **`cv2.VideoCapture(dev, CAP_V4L2)` 로 재오픈** 후, plain open 이 VGA 로 여니까
   `cap.set(FRAME_WIDTH/HEIGHT=320/240)` 로 **QVGA 강제**. 이때 장치 간격이 이미 30 으로
   박혀 있어 이 S_FMT 는 60 으로 리셋하지 않고 **30 을 유지**한다(프로브 [B] 로 확인).
4. 새 핸들을 `_Digit__dev` 에 꽂는다(`get_frame()` 은 거기서 read 하므로 그대로 동작).

추가 보강:
- `_measure_fps`(스트리밍 후 실측) + 연결 로그에 `drv=<v4l2 get-parm>, 실측 ~Nfps` 표기 →
  "지금 진짜 30 으로, QVGA 로 도는지"를 눈으로 검증. 실측>요청×1.4 면 `[경고]`.
- `_snap_cam_fps`: 비지원 fps(예: 15)는 30/60 으로 자동 스냅(드라이버 임의 반올림 방지).

**그래도 남은 마지막 1할 — `uvcvideo quirks=128`:**
QVGA@30 ×4 = 18.4 MB/s 까지 줄여도 단일 USB2 컨트롤러엔 여전히 빠듯해서, 한동안 깨끗하다가
**중간에 한 번 desync 되면 고착**됐다. 결정타는 커널 quirk:

```bash
sudo modprobe -r uvcvideo && sudo modprobe uvcvideo quirks=128   # = UVC_QUIRK_FIX_BANDWIDTH
```

드라이버가 예약하는 isochronous 대역폭을 줄여 한 컨트롤러에 더 많은 카메라를 태운다(다중
UVC 카메라 동시 사용의 표준 처방). 적용 후 4대 동시 무접촉이 `max≈0.001mm` 로 깨끗.
**완전 결정적 해법은 센서를 서로 다른 USB 호스트 컨트롤러로 물리 분산**하는 것(Bus 02 가 빔).

---

## 5. 무접촉 노이즈 — 찢어진 프레임 / MJPEG 압축 + Poisson 적분

**증상:** 아무것도 안 닿았는데 depth 가 1mm 이상으로 뜸. 1대일 때보다 4대일 때 심함.

**메커니즘:** `get_depthmap` 은 모델이 낸 **gradient 를 `fast_poisson` 로 적분**한다.
적분은 미세한 저주파 입력 노이즈를 **넓은 깊이 bias 로 증폭**한다. 입력에 frame-blank 잔차가
생기는 두 경로 모두 이 노이즈로 증폭됐다:
- **찢어진 프레임**(#3,#4): torn frame 은 blank 와 달라 `frame-blank ≠ 0` → phantom depth.
  → 섞임을 잡으면(QVGA@30 + quirk=128) 이 노이즈도 같이 사라진다(가장 큰 원인이었음).
- **MJPEG 손실 압축 잡음**: 프레임마다 달라져 `frame-blank ≠ 0`. → 이 카메라엔 MJPG 자체가
  없어 무관했지만, 일반적으로 적분 기반 파이프라인에선 MJPEG 를 피해야 한다.

**해결:**
- **MJPEG 기본 OFF** → 무압축 YUYV (압축 잡음 자체 제거).
- 입력 **dead-band** `--diff-thresh`(기본 0.04 ≈ 10/255): `|frame-blank|` 가 작으면 0 처리해
  무접촉 픽셀이 정확히 0 으로 모델에 들어가게 함 → 적분 증폭 원천 차단.
- 보조: `--blank-frames`(깨끗한 기준 평균), `--depth-floor`, `--depth-ema`(시간축 평활).

**검증 (per-sensor 격리 테스트, `--no-render` 노이즈 리포트):**

| 실행 | 평균 p99 | 노이즈 픽셀% | 판정 |
|------|---------|-------------|------|
| D21422 단독 (YUYV) | 0.000 | 0.00% | 깨끗 |
| D21429 단독 (YUYV) | 0.117 | 0.22% | 일시적(4대 땐 사라짐) |
| D21424 단독 (YUYV) | 0.000 | 0.00% | 깨끗 |
| D21418 단독 (YUYV) | 0.000 | 0.00% | 깨끗 |
| **4대 동시 (YUYV@30 + quirks=128)** | **0.001** | **0.00%** | **전부 깨끗** |

→ **센서 개체 불량 아님.** 4대 동시 YUYV@30 + quirk 에서 측정 노이즈는 사실상 0
(`--no-render` 로 `max≈0.001mm` 지속 확인).

---

## 6. "여전히 노이즈 심함" — 시각화 자동 스케일

**증상:** 측정값(`--no-render`)은 깨끗(p99≈0)한데 **3D/2D 화면은 노이즈투성이**.

**원인:** 렌더가 컬러맵 `vmax` / 3D `zlim` 을 최근 깊이 최댓값의 EMA(`emas`)에 자동으로
맞춘다. 무접촉이면 깊이 최댓값이 ~0.001mm 라 **표시 범위가 0.001mm 까지 붕괴**하고,
그 ±0.0005mm 짜리 미세 수치 노이즈가 색·높이 전체로 늘어나 폭발한 것처럼 보였다.
**실제 깊이는 평평했고, 보이는 것만 과장됨.**

**해결:** 표시 스케일 하한 **`--min-scale`(기본 1.0mm)** 추가.
- `vmax_of(sr) = max(emas[sr], min_scale)` 로 컬러맵/`zlim`/`clim` 모두 통일.
- 무접촉이면 0~1mm 스케일에서 평평한 ~0 으로 보이고, 실제 접촉(>1mm)이면 그때 스케일이 커짐.
- 약한 접촉(~0.5mm)까지 보려면 `--min-scale 0.5`, 노이즈가 더 보이면 `--min-scale 2.0`.

> 이 한 줄이 사용자의 "노이즈 심함" 불만을 실제로 해결한 결정타.

---

## 7. 속도 — 무엇이 병목인가 (렌더 vs 카메라 vs 추론)

세 가지 서로 다른 한계가 있고, 자주 혼동된다.

| 모드 | loop fps | infer fps(4대 합) | 병목 |
|------|---------|------------------|------|
| 3D 렌더(기본) | **~2.3** | ~150 | **matplotlib `plot_surface`** (압도적) |
| `--no-render` | **~40** | ~185 | 캡처+numpy+Poisson 오버헤드 |

- **렌더 한계:** 3D 표면 렌더가 루프를 2.3fps 로 묶는다. `--mode 2d`(imshow), `--downsample`↑,
  `--rcount/--ccount`↓ 로 크게 개선. 캡처/추론과 렌더 레이트를 분리하면(예: N프레임에 1번만
  렌더) 라이브 뷰를 부드럽게 유지하면서 처리율을 살릴 수 있다.
- **카메라 입력 한계(하드 캡):** 현재 대역폭 때문에 센서당 **30fps** 로 고정. 루프가 더 빨라도
  새 프레임은 30/s 만 들어온다. 60fps 입력을 원하면 `--cam-fps 60`(QVGA 실지원값) — 단
  quirk=128 이 켜진 상태에서만 4대가 USB2 에 들어갈 수 있고, 다시 찢어지면 컨트롤러 분산 필요.
- **추론 한계:** 로그의 `infer fps` 는 **4대 합산**. 센서당으로는 그 ÷4(≈40~46fps) 가 실제
  연산 여력. 즉 compute 는 30~46fps 는 여유, 60 은 빠듯.

---

## 관련 옵션 모음

| 옵션 | 기본 | 목적 |
|------|------|------|
| `--cam-fps` | 30 | USB 대역폭↓ → 다중 동시 섞임 방지. **QVGA 는 30/60 만 유효**(비지원 값은 자동 스냅+실측 경고). 1~2대면 60 가능 |
| `--mjpeg` / `--no-mjpeg` | OFF | MJPEG 는 대역폭↓지만 압축 노이즈 부작용 → 기본 YUYV |
| `--diff-thresh` | 0.04 | 입력 dead-band, Poisson 적분 노이즈 증폭 차단 |
| `--min-scale` | 1.0 | **표시 스케일 하한** (무접촉 노이즈 시각적 증폭 방지) |
| `--blank-frames` | 20 | 깨끗한 기준 blank 평균 장수 |
| `--depth-floor` | 0.0 | 출력 깊이 바닥값 |
| `--depth-ema` | 0.0 | 깊이 시간축 평활 |

## 교훈

1. 다중 동일 UVC 카메라 동시 스트리밍의 프레임 섞임은 **드라이버/대역폭 문제** — 코드 락으로 안 됨.
   결정타는 **`uvcvideo quirks=128`(FIX_BANDWIDTH)** + 카메라당 대역폭 축소(QVGA@30), 궁극은
   **USB 호스트 컨트롤러 분산**.
2. **OpenCV `CAP_PROP_FPS` 를 믿지 마라.** uvcvideo 장치에 따라 무시된다. 프레임 간격을 실제로
   바꾸는 건 `v4l2-ctl --set-parm`(=`VIDIOC_S_PARM`)이고 **장치를 열기 전(닫힌 상태)** 에 걸어야
   한다. `S_FMT`(해상도 설정)는 간격을 포맷 기본값으로 **리셋**하고, OpenCV plain open 은 해상도를
   **VGA 로 강제**한다 — 그래서 "닫힌 상태 set-fmt+set-parm → 재오픈 → cap.set(QVGA)" 순서가 필요.
3. **추측하지 말고 측정하라.** `v4l2-ctl --list-formats-ext/--get-parm` 로 하드웨어 실제 능력을,
   일회성 프로브(`_probe_fps.py`)로 "어떤 호출 순서가 실제로 먹히는지"를 먼저 확정했다. 연결 시
   `실측 fps + drv` 를 항상 출력해 가정이 아닌 실측으로 검증.
4. **대역폭은 fps 로 줄이는 게 우선**(무압축 유지). MJPEG 압축은 적분 기반 파이프라인에서 노이즈를
   만든다(게다가 이 카메라엔 MJPG 포맷 자체가 없었다).
5. "노이즈"를 의심하면 먼저 **측정값(수치)과 시각화(자동 스케일)를 분리**해서 본다. 측정이 깨끗한데
   화면이 시끄러우면 스케일/표시 문제(`--min-scale`)이고, 측정 자체가 시끄러우면 입력(찢어짐/압축)
   문제다.
6. **병목을 분리해서 보라:** 렌더(matplotlib 3D) / 카메라 입력 fps(하드 캡) / 추론(4대 합산)은
   서로 다른 한계다(§7).
