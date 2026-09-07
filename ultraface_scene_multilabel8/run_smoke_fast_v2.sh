#!/usr/bin/env bash
set -euo pipefail
if [[ "${CONDA_DEFAULT_ENV:-}" != "Ultraface" ]]; then echo "ERROR: conda activate Ultraface first"; exit 2; fi
DATA_ROOT="${DATA_ROOT:-/data/pub1/z00919662/dataset/UltraFaceSlim_8scene_multilabel_manifests_640x360_fast_v2}"
OUT="${OUT:-/data/pub1/z00919662/scene_multilabel/ultraface_slim_8label_640x360_fast_v2/smoke}"
GPU="${GPU:-0}"; BATCH="${BATCH:-128}"; WORKERS="${WORKERS:-16}"; PREFETCH="${PREFETCH:-2}"
export CUDA_VISIBLE_DEVICES="${GPU}" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
rm -rf "${OUT}"; mkdir -p "${OUT}"
python -u train_fast_v2_uint8.py \
  --data-root "${DATA_ROOT}" --output-dir "${OUT}" \
  --epochs 1 --batch-size "${BATCH}" --workers "${WORKERS}" --prefetch-factor "${PREFETCH}" \
  --lr 1e-2 --momentum 0.9 --weight-decay 1e-4 --milestones 20,27 --gamma 0.1 \
  --cpu-threads 4 --amp --log-every 10 --max-train-steps 30 --max-eval-batches 8 \
  2>&1 | tee "${OUT}/train.log"
