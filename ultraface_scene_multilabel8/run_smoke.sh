#!/usr/bin/env bash
set -euo pipefail
source /data/pub1/z00919662/anaconda3/etc/profile.d/conda.sh
conda activate Ultraface
DATA_ROOT="${DATA_ROOT:-/data/pub1/z00919662/dataset/UltraFaceSlim_8scene_multilabel_manifests_640x360_v1}"
OUT="${OUT:-/data/pub1/z00919662/scene_multilabel/ultraface_slim_8label_640x360_v1/smoke}"
GPU="${GPU:-0}"; BATCH="${BATCH:-24}"
export CUDA_VISIBLE_DEVICES="${GPU}" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
rm -rf "${OUT}"; mkdir -p "${OUT}"
python -u train.py --data-root "${DATA_ROOT}" --output-dir "${OUT}" --epochs 1 --batch-size "${BATCH}" --workers 4 --lr 1e-2 --weight-decay 1e-4 --milestones 95,150 --cpu-threads 4 --amp --max-train-steps 20 --max-eval-batches 8 2>&1 | tee "${OUT}/train.log"
