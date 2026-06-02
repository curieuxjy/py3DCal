#!/usr/bin/env bash
# TouchNet(CNN) vs DPT(ViT) 추론 속도 자동 벤치마크.
#
# --no-render 모드로 약 10초간(기본) 두 모델의 추론 시간을 측정해 모델별 fps/ms 를 비교한다.
# 디스플레이/렌더 없이 순수 추론만 측정하므로 GPU 처리량을 정확히 비교할 수 있다.
#
# 사용법:
#   bash examples/benchmark_touchnet_dpt.sh [SERIAL] [DURATION_SEC] [추가인자...]
# 예:
#   bash examples/benchmark_touchnet_dpt.sh                       # 첫 DIGIT 자동, 10초, dpt_real
#   bash examples/benchmark_touchnet_dpt.sh D21424 10             # 시리얼/시간 지정
#   # dpt_real + dpt_sim 둘 다 측정(추가 인자는 그대로 python 으로 전달):
#   WDIR=/home/avery/Documents/neuralfeels/deploy/weights/tactile_transformer
#   bash examples/benchmark_touchnet_dpt.sh D21424 10 --dpt-weights $WDIR/dpt_real.p $WDIR/dpt_sim.p
#
# 필요: py3dcal conda 환경, DIGIT 센서 연결.
set -euo pipefail

ENV_NAME="py3dcal"
SERIAL="${1:-}"
DURATION="${2:-10}"
shift "$(( $# < 2 ? $# : 2 ))" || true   # 앞의 SERIAL/DURATION 소비, 나머지는 passthrough
EXTRA_ARGS=("$@")

# 저장소 루트로 이동(이 스크립트 위치 기준)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# --serial 인자 구성(미지정 시 자동 탐지)
SERIAL_ARG=()
if [[ -n "$SERIAL" ]]; then
  SERIAL_ARG=(--serial "$SERIAL")
fi

echo "==================================================="
echo " TouchNet vs DPT 추론 속도 벤치마크"
echo "  - 환경    : conda env '$ENV_NAME'"
echo "  - 센서    : ${SERIAL:-(자동 탐지)}"
echo "  - 측정시간: ${DURATION}s (모델 로딩/blank 캡처 제외)"
echo "==================================================="

# -u: 언버퍼 출력(실시간 로그). --duration 으로 측정 후 자동 종료되며 최종 요약을 출력한다.
conda run --no-capture-output -n "$ENV_NAME" python -u examples/stream_compare_touchnet_dpt.py \
  "${SERIAL_ARG[@]}" \
  --no-render \
  --duration "$DURATION" \
  "${EXTRA_ARGS[@]}"
