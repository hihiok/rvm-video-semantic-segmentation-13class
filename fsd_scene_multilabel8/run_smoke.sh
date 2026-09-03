#!/usr/bin/env bash
set -euo pipefail

: "${FSD_ROOT:?Set FSD_ROOT to existing FSD repository root}"
DATA_ROOT="${DATA_ROOT:-/data/pub1/z00919662/dataset/FSD_8scene_multilabel_manifests}"
SMOKE_DATA_ROOT="${SMOKE_DATA_ROOT:-/data/pub1/z00919662/dataset/FSD_8scene_multilabel_smoke_manifests}"
OUT="${OUT:-/data/pub1/z00919662/scene_multilabel/fsd_8label_v1/smoke}"
GPU="${GPU:-0}"
BATCH="${BATCH:-24}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

rm -rf "${SMOKE_DATA_ROOT}" "${OUT}"
mkdir -p "${SMOKE_DATA_ROOT}" "${OUT}"
python -u make_smoke_manifest.py --data-root "${DATA_ROOT}" --output-root "${SMOKE_DATA_ROOT}" --per-state 24

python -u train_fsd_scene_multilabel8.py \
  --fsd-root "${FSD_ROOT}" \
  --data-root "${SMOKE_DATA_ROOT}" \
  --output-dir "${OUT}" \
  --input-size 240 \
  --epochs 1 \
  --batch-size "${BATCH}" \
  --workers 4 \
  --lr 1e-2 \
  --weight-decay 1e-4 \
  --milestones 95,150 \
  --seed 20260903 \
  --cpu-threads 4 \
  --amp true \
  2>&1 | tee "${OUT}/train.log"
