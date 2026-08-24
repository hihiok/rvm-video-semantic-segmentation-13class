#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

VSPW_ROOT="${VSPW_ROOT:-/data/pub1/z00919662/segmentation/datasets/VSPW_13cls}"
SOURCE_PROJECT="${SOURCE_PROJECT:-/data/pub1/z00919662/segmentation/RobustVideoMatting-semantic-13class-512}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-${SOURCE_PROJECT}/output/rvm_semantic_13class_512_2gpu/best_miou.pth}"
STATIC_ROOT="${STATIC_ROOT:-${SOURCE_PROJECT}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/pub1/z00919662/segmentation/rvm-video-semantic-segmentation-13class/output/rvm_vspw_mixed_13class}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-20}"
STAGE3_EPOCHS="${STAGE3_EPOCHS:-60}"
VIDEO_BATCH_SIZE="${VIDEO_BATCH_SIZE:-2}"
STATIC_BATCH_SIZE="${STATIC_BATCH_SIZE:-8}"
if [[ -n "${INPUT_SIZE:-}" ]]; then
  INPUT_WIDTH="${INPUT_SIZE}"
  INPUT_HEIGHT="${INPUT_SIZE}"
else
  INPUT_WIDTH="${INPUT_WIDTH:-640}"
  INPUT_HEIGHT="${INPUT_HEIGHT:-360}"
fi
WORKERS="${WORKERS:-4}"
MAX_GPUS="${MAX_GPUS:-2}"
MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB:-18000}"
MASTER_PORT="${MASTER_PORT:-29637}"

if [[ ! -d "${VSPW_ROOT}/images/train" || ! -d "${VSPW_ROOT}/annotations/train" ]]; then
  printf 'ERROR: VSPW dataset is not ready: %s\n' "${VSPW_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${INIT_CHECKPOINT}" && -z "${RESUME:-}" ]]; then
  printf 'ERROR: first-stage checkpoint does not exist: %s\n' "${INIT_CHECKPOINT}" >&2
  exit 2
fi

if [[ ! -d "${STATIC_ROOT}" && "${AUTO_DISCOVER_STATIC_ROOT:-0}" == "1" ]]; then
  STATIC_ROOT="$(python tools/resolve_static_dataset.py \
    --checkpoint "${INIT_CHECKPOINT}" \
    --source-project "${SOURCE_PROJECT}" \
    --root-only)"
fi
if [[ ! -d "${STATIC_ROOT}" ]]; then
  printf 'ERROR: expected COCO+ADE13 static replay dataset does not exist: %s\n' "${STATIC_ROOT}" >&2
  printf 'Set STATIC_ROOT explicitly only if the confirmed dataset was moved.\n' >&2
  exit 2
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(python - "${MAX_GPUS}" "${MIN_FREE_GPU_MIB}" <<'PY'
import subprocess
import sys

maximum = int(sys.argv[1])
minimum_free = int(sys.argv[2])
output = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
candidates = []
for line in output.splitlines():
    index, free, utilization = (int(value.strip()) for value in line.split(","))
    if free >= minimum_free and utilization <= 25:
        candidates.append((free, index))
candidates.sort(reverse=True)
if not candidates:
    raise SystemExit(
        f"No GPU has at least {minimum_free} MiB free and utilization <= 25%; "
        "wait for a GPU or explicitly set CUDA_VISIBLE_DEVICES."
    )
print(",".join(str(index) for _, index in candidates[:maximum]))
PY
)"
fi
export CUDA_VISIBLE_DEVICES
NPROC_PER_NODE="$(python -c 'import os; print(len([x for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip()]))')"

mkdir -p "${OUTPUT_DIR}"
python tools/audit_vspw_mixed.py \
  --vspw-root "${VSPW_ROOT}" \
  --static-root "${STATIC_ROOT}" \
  --max-video-frames "${AUDIT_MAX_VIDEO_FRAMES:-20000}" \
  --max-static-images "${AUDIT_MAX_STATIC_IMAGES:-10000}" \
  --output-json "${OUTPUT_DIR}/dataset_audit.json"

args=(
  --data-root "${VSPW_ROOT}"
  --static-root "${STATIC_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --stage2-epochs "${STAGE2_EPOCHS}"
  --stage3-epochs "${STAGE3_EPOCHS}"
  --stage2-clip-length "${STAGE2_CLIP_LENGTH:-5}"
  --stage3-clip-length "${STAGE3_CLIP_LENGTH:-8}"
  --stage2-video-batches "${STAGE2_VIDEO_BATCHES:-1}"
  --stage2-static-batches "${STAGE2_STATIC_BATCHES:-1}"
  --stage3-video-batches "${STAGE3_VIDEO_BATCHES:-2}"
  --stage3-static-batches "${STAGE3_STATIC_BATCHES:-1}"
  --batch-size "${VIDEO_BATCH_SIZE}"
  --static-batch-size "${STATIC_BATCH_SIZE}"
  --workers "${WORKERS}"
  --input-width "${INPUT_WIDTH}"
  --input-height "${INPUT_HEIGHT}"
  --learning-rate "${LEARNING_RATE:-5e-5}"
  --backbone-learning-rate "${BACKBONE_LEARNING_RATE:-5e-6}"
  --static-validation-weight "${STATIC_VALIDATION_WEIGHT:-0.5}"
  --static-retention-tolerance "${STATIC_RETENTION_TOLERANCE:-0.03}"
  --save-every "${SAVE_EVERY:-10}"
)
if [[ -n "${RESUME:-}" ]]; then
  args+=(--resume "${RESUME}")
else
  args+=(--init-checkpoint "${INIT_CHECKPOINT}")
fi

printf 'VSPW_ROOT=%s\nSTATIC_ROOT=%s\nCUDA_VISIBLE_DEVICES=%s\nNPROC_PER_NODE=%s\nOUTPUT_DIR=%s\nINPUT_RESOLUTION=%sx%s\n' \
  "${VSPW_ROOT}" "${STATIC_ROOT}" "${CUDA_VISIBLE_DEVICES}" "${NPROC_PER_NODE}" "${OUTPUT_DIR}" \
  "${INPUT_WIDTH}" "${INPUT_HEIGHT}"
if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  exec torchrun --standalone --nnodes=1 --nproc_per_node "${NPROC_PER_NODE}" \
    --master_port "${MASTER_PORT}" train_vspw_mixed.py "${args[@]}"
fi
exec python -u train_vspw_mixed.py "${args[@]}"
