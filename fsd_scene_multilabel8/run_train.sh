#!/usr/bin/env bash
set -euo pipefail

: "${FSD_ROOT:?Set FSD_ROOT to existing FSD repository root}"
DATA_ROOT="${DATA_ROOT:-/data/pub1/z00919662/dataset/FSD_8scene_multilabel_manifests}"
OUT="${OUT:-/data/pub1/z00919662/scene_multilabel/fsd_8label_640x360_v1/train_640x360}"
GPU="${GPU:-0}"
BATCH="${BATCH:-24}"
RESUME="${RESUME:-}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
mkdir -p "${OUT}"

ARGS=(
  --fsd-root "${FSD_ROOT}"
  --data-root "${DATA_ROOT}"
  --output-dir "${OUT}"
  --input-width 640
  --input-height 360
  --fd-bootstrap-size 640
  --epochs 200
  --batch-size "${BATCH}"
  --workers 4
  --lr 1e-2
  --weight-decay 1e-4
  --milestones 95,150
  --seed 20260904
  --cpu-threads 4
  --amp true
)
if [[ -n "${RESUME}" ]]; then
  ARGS+=(--resume-train-state "${RESUME}")
fi

python -u train_fsd_scene_multilabel8_640x360.py "${ARGS[@]}" \
  2>&1 | tee -a "${OUT}/train.log"
