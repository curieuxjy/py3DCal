#!/usr/bin/env bash
# TouchNet 최적화 전/후 자동 비교.
#
# "아무 최적화 없음(BASE: fp32·풀해상도·scipy Poisson)" vs
# "전부 적용(OPT: 저해상도 + GPU Poisson + torch.compile + FP16)" 의
# 추론 속도와 결과(깊이맵 유사도)를 한 번에 측정/저장한다.
# 내부적으로 stream_compare_touchnet_opt.py 를 호출한다.
#
# 사용법:
#   bash examples/benchmark_touchnet_opt.sh [SERIAL] [INFER_SCALE] [DURATION] [추가인자...]
# 예:
#   bash examples/benchmark_touchnet_opt.sh                    # 첫 DIGIT 자동, 1/2 해상도, 10초
#   bash examples/benchmark_touchnet_opt.sh D21424 2 10        # 시리얼/스케일/측정시간 지정
#   bash examples/benchmark_touchnet_opt.sh D21424 4 10        # 1/4 해상도
#
# --no-render 로 약 DURATION 초간 BASE/OPT 모델별 추론 fps 를 측정하고 속도 요약을 출력한다.
# (마지막 프레임에서 결과 형상 corr 도 출력 — 의미 있는 값을 보려면 측정 중 센서를 눌러 주세요.)
# 결과 깊이맵 그림까지 보려면 렌더 모드로 직접 실행:
#   python examples/stream_compare_touchnet_opt.py --serial <S> --headless 60
#
# 필요: py3dcal conda 환경, DIGIT 센서 연결.
set -euo pipefail

ENV_NAME="py3dcal"
SERIAL="${1:-}"
INFER_SCALE="${2:-2}"
DURATION="${3:-10}"
shift "$(( $# < 3 ? $# : 3 ))" || true
EXTRA_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

SERIAL_ARG=()
if [[ -n "$SERIAL" ]]; then
  SERIAL_ARG=(--serial "$SERIAL")
fi

echo "==================================================="
echo " TouchNet 최적화 전/후 속도 비교 (BASE vs OPT)"
echo "  - 환경      : conda env '$ENV_NAME'"
echo "  - 센서      : ${SERIAL:-(자동 탐지)}"
echo "  - 저해상도  : 1/${INFER_SCALE}"
echo "  - 측정시간  : ${DURATION}s (blank 캡처 후. 측정 중 센서를 눌러 주세요)"
echo "==================================================="

conda run --no-capture-output -n "$ENV_NAME" python -u examples/stream_compare_touchnet_opt.py \
  "${SERIAL_ARG[@]}" \
  --infer-scale "$INFER_SCALE" \
  --no-render \
  --duration "$DURATION" \
  "${EXTRA_ARGS[@]}"
