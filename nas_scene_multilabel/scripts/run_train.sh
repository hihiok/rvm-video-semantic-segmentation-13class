#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

RAW_ROOT="${RAW_ROOT:-/data/pub1/z00919662/scene_multilabel/datasets_raw}"
PLACES_ROOT="${PLACES_ROOT:-${RAW_ROOT}/places365}"
COCO_ROOT="${COCO_ROOT:-${RAW_ROOT}/coco2017}"
SEG_ROOT="${SEG_ROOT:-/data/pub1/z00919662/segmentation/datasets/COCO_ADE_13cls_16x9_640x360}"
DATA_ROOT="${DATA_ROOT:-/data/pub1/z00919662/scene_multilabel/nas_9label_partial_gt}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/pub1/z00919662/scene_multilabel/ultraface_slim_9label/output}"

INPUT_SIZE="${INPUT_SIZE:-224}"
BASE_CHANNEL="${BASE_CHANNEL:-16}"
BATCH_SIZE="${BATCH_SIZE:-256}"
WORKERS="${WORKERS:-4}"
EPOCHS="${EPOCHS:-60}"
LR="${LR:-0.001}"

python tools/model_info.py --input-size "${INPUT_SIZE}" --base-channel "${BASE_CHANNEL}"

python -u prepare_dataset.py \
  --places-root "${PLACES_ROOT}" \
  --coco-root "${COCO_ROOT}" \
  --seg-root "${SEG_ROOT}" \
  --output-root "${DATA_ROOT}" \
  --places-train-cap-per-class "${PLACES_TRAIN_CAP_PER_CLASS:-500}" \
  --seed 20260902

mkdir -p "${OUTPUT_DIR}"

ARGS=(
  --data-root "${DATA_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --input-size "${INPUT_SIZE}"
  --base-channel "${BASE_CHANNEL}"
  --batch-size "${BATCH_SIZE}"
  --workers "${WORKERS}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --cpu-threads 4
)

if [[ -n "${RESUME:-}" ]]; then
  ARGS+=(--resume "${RESUME}")
fi
if [[ -n "${INIT_BACKBONE:-}" ]]; then
  ARGS+=(--init-backbone "${INIT_BACKBONE}")
fi

python -u train.py "${ARGS[@]}"

python export_onnx.py \
  --checkpoint "${OUTPUT_DIR}/best_deploy.pth" \
  --output "${OUTPUT_DIR}/ultraface_slim_9label_224.onnx"
