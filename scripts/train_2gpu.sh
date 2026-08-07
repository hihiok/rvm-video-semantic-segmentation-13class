#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/data/pub1/z00919662/dataset/VIPSeg_13cls_video}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/rvm_video_semantic_13class}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
RESUME="${RESUME:-}"

test -d "${DATA_ROOT}/images/train"
test -d "${DATA_ROOT}/annotations/train"
mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

EXTRA_ARGS=()
if [[ -n "${RESUME}" ]]; then
  EXTRA_ARGS+=(--resume "${RESUME}")
elif [[ -n "${INIT_CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--init-checkpoint "${INIT_CHECKPOINT}")
elif [[ -f "${OUTPUT_DIR}/last.pth" ]]; then
  EXTRA_ARGS+=(--resume "${OUTPUT_DIR}/last.pth")
fi

export CUDA_VISIBLE_DEVICES
python -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  train_video_semantic.py \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --variant mobilenetv3 \
  --input-size 512 \
  --clip-length 5 \
  --frame-stride 1 \
  --train-clip-step 5 \
  --epochs 100 \
  --batch-size 2 \
  --gradient-accumulation 2 \
  --workers 8 \
  --learning-rate 1e-4 \
  --backbone-learning-rate 1e-5 \
  --ce-weight 1.0 \
  --dice-weight 1.0 \
  --amp \
  --sync-bn \
  "${EXTRA_ARGS[@]}" \
  "$@"
