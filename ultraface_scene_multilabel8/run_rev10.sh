#!/usr/bin/env bash
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == "ultraface_new" ]] || { echo 'Activate ultraface_new'; exit 2; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/mnt/ssd1/z00919662/datasets/NAS8_multilabel_weather_rev10_20260916_152810}"
: "${GPU_IDS:?Set GPU_IDS to one or two verified idle GPUs, e.g. 0,1}"
: "${RUN_ROOT:?Set RUN_ROOT to a NEW directory under NAS_scene_detection/runs}"
BATCH="${BATCH:-128}"; WORKERS="${WORKERS:-2}"
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NPROC="${#GPU_ARRAY[@]}"
[[ "$NPROC" == 1 || "$NPROC" == 2 ]] || { echo 'Use one or two GPUs'; exit 2; }
export CUDA_VISIBLE_DEVICES="$GPU_IDS" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
python -c 'import torch,cv2,numpy,onnx,onnxruntime; assert torch.cuda.is_available(); print("torch",torch.__version__,"CUDA",torch.version.cuda,"visible GPUs",torch.cuda.device_count())'
if [[ -z "${RESUME:-}" ]]; then
  [[ ! -e "$RUN_ROOT" ]] || { echo 'RUN_ROOT exists; choose new directory or explicit RESUME'; exit 2; }
  mkdir -p "$RUN_ROOT"
fi
ARGS=(--data-root "$DATA_ROOT" --accept-weak-weather --global-batch-size "$BATCH" --workers "$WORKERS" --amp)
LAUNCH=(python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" "$SCRIPT_DIR/train_rev10.py")
if [[ -z "${RESUME:-}" ]]; then
  "${LAUNCH[@]}" "${ARGS[@]}" --output-dir "$RUN_ROOT/smoke" --epochs 1 --smoke 2>&1 | tee "$RUN_ROOT/smoke.log"
  [[ -f "$RUN_ROOT/smoke/SMOKE_PASS.json" ]] || exit 2
  "${LAUNCH[@]}" "${ARGS[@]}" --output-dir "$RUN_ROOT/train" --epochs 60 --milestones 35,50 2>&1 | tee "$RUN_ROOT/train.log"
else
  [[ "$RESUME" == "$RUN_ROOT/train/last_train_state.pth" ]] || { echo 'Resume only own last_train_state.pth'; exit 2; }
  "${LAUNCH[@]}" "${ARGS[@]}" --output-dir "$RUN_ROOT/train" --epochs 60 --milestones 35,50 --resume "$RESUME" 2>&1 | tee -a "$RUN_ROOT/train.log"
fi
[[ -f "$RUN_ROOT/train/COMPLETE.json" ]] || exit 2
echo "TRAINING_COMPLETE $RUN_ROOT/train"
