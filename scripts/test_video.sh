#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/data/pub1/z00919662/dataset/VIPSeg_13cls_video}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/output/rvm_video_semantic_13class/best_miou.pth}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

cd "${PROJECT_ROOT}"
test -f "${CHECKPOINT}"
python test_video_semantic.py \
  --data-root "${DATA_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --images images/test \
  --annotations annotations/test \
  --batch-size 2 \
  --workers 8 \
  --amp \
  "$@"
