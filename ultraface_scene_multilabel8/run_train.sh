#!/usr/bin/env bash
set -euo pipefail
if [[ "${CONDA_DEFAULT_ENV:-}" != "Ultraface" ]]; then
  echo "ERROR: activate conda environment Ultraface before running"
  exit 2
fi
DATA_ROOT="${DATA_ROOT:-/data/pub1/z00919662/dataset/UltraFaceSlim_8scene_multilabel_manifests_640x360_v1}"
OUT="${OUT:-/data/pub1/z00919662/scene_multilabel/ultraface_slim_8label_640x360_v1/train}"
GPU="${GPU:-0}"; BATCH="${BATCH:-24}"; RESUME="${RESUME:-}"
export CUDA_VISIBLE_DEVICES="${GPU}" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
mkdir -p "${OUT}"
ARGS=(--data-root "${DATA_ROOT}" --output-dir "${OUT}" --epochs 200 --batch-size "${BATCH}" --workers 4 --lr 1e-2 --momentum 0.9 --weight-decay 1e-4 --milestones 95,150 --gamma 0.1 --cpu-threads 4 --amp)
if [[ -n "${RESUME}" ]]; then ARGS+=(--resume "${RESUME}"); fi
python -u train.py "${ARGS[@]}" 2>&1 | tee -a "${OUT}/train.log"
