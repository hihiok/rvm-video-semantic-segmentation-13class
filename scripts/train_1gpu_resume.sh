#!/usr/bin/env bash
set -euo pipefail

# Stable single-GPU Stage-12 launcher for RVM 13-class video semantic segmentation.
#
# The previous crash was caused by system cuDNN being injected through
# LD_LIBRARY_PATH. PyTorch 2.3.0 must use its compatible bundled CUDA/cuDNN
# runtime, so clear the external library overrides before importing torch.
#
# IMPORTANT: this server is shared/sensitive to high CPU load. Keep the
# DataLoader worker count and CPU math-library thread counts deliberately low.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/data/pub1/z00919662/dataset/VIPSeg_13cls_video}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/rvm_video_semantic_13class}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-2}"
WORKERS="${WORKERS:-2}"
CPU_THREADS="${CPU_THREADS:-1}"
RESUME="${RESUME:-}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/data/pub1/z00919662/segmentation/RobustVideoMatting-semantic-12class-512/output/rvm_semantic_12class_512_2gpu/best_miou.pth}"

TRAIN_IMAGES="${DATA_ROOT}/images/train"
TRAIN_ANNOTATIONS="${DATA_ROOT}/annotations/train"
VAL_IMAGES="${DATA_ROOT}/images/val"
VAL_ANNOTATIONS="${DATA_ROOT}/annotations/val"

for path in \
  "${TRAIN_IMAGES}" \
  "${TRAIN_ANNOTATIONS}" \
  "${VAL_IMAGES}" \
  "${VAL_ANNOTATIONS}"; do
  if [[ ! -d "${path}" ]]; then
    echo "ERROR: required dataset directory does not exist: ${path}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

# Critical cuDNN compatibility fix.
unset LD_LIBRARY_PATH || true
unset LD_PRELOAD || true

export CUDA_VISIBLE_DEVICES

# Keep CPU pressure low on the shared server. These variables prevent a single
# DataLoader/linear-algebra process from spawning a large CPU thread pool.
export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"
export VECLIB_MAXIMUM_THREADS="${CPU_THREADS}"
export TOKENIZERS_PARALLELISM=false

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${PROJECT_ROOT}/.venv/bin/python"
else
  PYTHON="${PYTHON:-python}"
fi

# Verify the selected runtime before starting a long job.
"${PYTHON}" - <<'PY'
import os
import torch

# Also cap PyTorch intra-op threads in this preflight process. The training
# process inherits the OMP/MKL/OpenBLAS thread limits above.
torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))

print("=== Runtime check ===")
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
print("cuda_available:", torch.cuda.is_available())
print("visible_devices:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
print("MKL_NUM_THREADS:", os.environ.get("MKL_NUM_THREADS"))
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available")
print("logical cuda:0:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
PY

EXTRA_ARGS=()
RESUME_SOURCE=""

if [[ -n "${RESUME}" ]]; then
  if [[ ! -f "${RESUME}" ]]; then
    echo "ERROR: explicit RESUME checkpoint does not exist: ${RESUME}" >&2
    exit 3
  fi
  EXTRA_ARGS+=(--resume "${RESUME}")
  RESUME_SOURCE="${RESUME}"
elif [[ -f "${OUTPUT_DIR}/last.pth" ]]; then
  EXTRA_ARGS+=(--resume "${OUTPUT_DIR}/last.pth")
  RESUME_SOURCE="${OUTPUT_DIR}/last.pth"
elif [[ -f "${INIT_CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--init-checkpoint "${INIT_CHECKPOINT}")
  RESUME_SOURCE="INIT:${INIT_CHECKPOINT}"
else
  echo "ERROR: no 13-class resume checkpoint found and init checkpoint is missing." >&2
  echo "Checked last.pth: ${OUTPUT_DIR}/last.pth" >&2
  echo "Checked init checkpoint: ${INIT_CHECKPOINT}" >&2
  exit 4
fi

echo "=== Single-GPU Stage-12 configuration ==="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "SOURCE=${RESUME_SOURCE}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION}"
echo "WORKERS=${WORKERS}"
echo "CPU_THREADS=${CPU_THREADS}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "MKL_NUM_THREADS=${MKL_NUM_THREADS}"
echo "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH-<unset>}"
echo "LD_PRELOAD=${LD_PRELOAD-<unset>}"

# Intentionally use plain Python: no torchrun, no DDP, no NCCL rendezvous.
# Do not pass --sync-bn; it is not needed for a single process/GPU.
# nice +10 ensures this training job yields CPU time to other server users.
set -o pipefail
nice -n 10 "${PYTHON}" train_video_semantic.py \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --variant mobilenetv3 \
  --input-size 512 \
  --clip-length 5 \
  --frame-stride 1 \
  --train-clip-step 5 \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --gradient-accumulation "${GRADIENT_ACCUMULATION}" \
  --workers "${WORKERS}" \
  --learning-rate 1e-4 \
  --backbone-learning-rate 1e-5 \
  --ce-weight 1.0 \
  --dice-weight 1.0 \
  --amp \
  "${EXTRA_ARGS[@]}" \
  "$@" \
  2>&1 | tee -a "${OUTPUT_DIR}/train.log"

RC=${PIPESTATUS[0]}
echo "TRAIN_EXIT_CODE=${RC}" | tee -a "${OUTPUT_DIR}/train.log"
exit "${RC}"
