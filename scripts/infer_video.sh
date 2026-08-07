#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/output/rvm_video_semantic_13class/best_miou.pth}"
INPUT="${1:?Usage: bash scripts/infer_video.sh INPUT_VIDEO_OR_DIR [OUTPUT_DIR]}"
OUTPUT_DIR="${2:-${PROJECT_ROOT}/output/video_inference}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

cd "${PROJECT_ROOT}"
test -f "${CHECKPOINT}"
python inference_video_semantic.py \
  --checkpoint "${CHECKPOINT}" \
  --input "${INPUT}" \
  --output-dir "${OUTPUT_DIR}" \
  --recurrent \
  --scene-cut-threshold 0.35 \
  --amp \
  "${@:3}"
