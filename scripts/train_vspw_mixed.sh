#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

VSPW_ROOT="${VSPW_ROOT:-/data/pub1/z00919662/segmentation/datasets/VSPW_13cls}"
SOURCE_PROJECT="${SOURCE_PROJECT:-/data/pub1/z00919662/segmentation/RobustVideoMatting-semantic-13class-512}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-${SOURCE_PROJECT}/output/rvm_semantic_13class_512_2gpu/best_miou.pth}"
ORIGINAL_STATIC_ROOT="${ORIGINAL_STATIC_ROOT:-${SOURCE_PROJECT}/data}"
PREPARED_STATIC_ROOT="${PREPARED_STATIC_ROOT:-/data/pub1/z00919662/segmentation/datasets/COCO_ADE_13cls_16x9_640x360}"
STATIC_ROOT="${STATIC_ROOT:-${PREPARED_STATIC_ROOT}}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/pub1/z00919662/segmentation/rvm-video-semantic-segmentation-13class/output/rvm_vspw_mixed_13class_640x360}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-20}"
STAGE3_EPOCHS="${STAGE3_EPOCHS:-60}"
VIDEO_BATCH_SIZE="${VIDEO_BATCH_SIZE:-2}"
STATIC_BATCH_SIZE="${STATIC_BATCH_SIZE:-8}"
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

if [[ ! -d "${STATIC_ROOT}" ]]; then
  printf 'ERROR: offline-prepared 16:9 static replay dataset does not exist: %s\n' "${STATIC_ROOT}" >&2
  printf 'Run bash scripts/prepare_static_16x9.sh before starting training.\n' >&2
  exit 2
fi
if [[ ! -f "${STATIC_ROOT}/PREPARED_16X9_MANIFEST.json" ]]; then
  printf 'ERROR: offline-prepared static dataset manifest is missing: %s/PREPARED_16X9_MANIFEST.json\n' "${STATIC_ROOT}" >&2
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
  --stage2-trainable-scope "${STAGE2_TRAINABLE_SCOPE:-all}"
  --stage3-trainable-scope "${STAGE3_TRAINABLE_SCOPE:-all}"
  --stage2-temporal-weight "${STAGE2_TEMPORAL_WEIGHT:-0.0}"
  --stage3-temporal-weight "${STAGE3_TEMPORAL_WEIGHT:-0.0}"
  --stage2-laplacian-weight "${STAGE2_LAPLACIAN_WEIGHT:-0.0}"
  --stage3-laplacian-weight "${STAGE3_LAPLACIAN_WEIGHT:-0.0}"
  --stage2-rvm-temporal-weight "${STAGE2_RVM_TEMPORAL_WEIGHT:-0.0}"
  --stage3-rvm-temporal-weight "${STAGE3_RVM_TEMPORAL_WEIGHT:-0.0}"
  --laplacian-levels "${LAPLACIAN_LEVELS:-5}"
  --rvm-temporal-beta "${RVM_TEMPORAL_BETA:-0.1}"
  --temporal-hidden-channels "${TEMPORAL_HIDDEN_CHANNELS:-16}"
  --temporal-adapter-scale "${TEMPORAL_ADAPTER_SCALE:-0.25}"
  --temporal-boundary-radius "${TEMPORAL_BOUNDARY_RADIUS:-2}"
  --temporal-temperature "${TEMPORAL_TEMPERATURE:-1.0}"
  --batch-size "${VIDEO_BATCH_SIZE}"
  --static-batch-size "${STATIC_BATCH_SIZE}"
  --workers "${WORKERS}"
  --input-width "${INPUT_WIDTH:-640}"
  --input-height "${INPUT_HEIGHT:-360}"
  --max-frame-gap "${MAX_FRAME_GAP:-1}"
  --learning-rate "${LEARNING_RATE:-5e-5}"
  --backbone-learning-rate "${BACKBONE_LEARNING_RATE:-5e-6}"
  --static-validation-weight "${STATIC_VALIDATION_WEIGHT:-0.5}"
  --static-retention-tolerance "${STATIC_RETENTION_TOLERANCE:-0.03}"
  --prediction-flip-penalty "${PREDICTION_FLIP_PENALTY:-0.1}"
  --save-every "${SAVE_EVERY:-10}"
)
if [[ "${TEMPORAL_RESIDUAL_ADAPTER:-0}" == "1" ]]; then
  args+=(--temporal-residual-adapter)
fi
if [[ -n "${CLASS_WEIGHTS:-}" ]]; then
  args+=(--class-weights "${CLASS_WEIGHTS}")
fi
if [[ -n "${RESUME:-}" ]]; then
  args+=(--resume "${RESUME}")
else
  args+=(--init-checkpoint "${INIT_CHECKPOINT}")
fi

printf 'VSPW_ROOT=%s\nORIGINAL_STATIC_ROOT=%s\nPREPARED_STATIC_ROOT=%s\nINPUT_WIDTH=%s\nINPUT_HEIGHT=%s\nCUDA_VISIBLE_DEVICES=%s\nNPROC_PER_NODE=%s\nOUTPUT_DIR=%s\n' \
  "${VSPW_ROOT}" "${ORIGINAL_STATIC_ROOT}" "${STATIC_ROOT}" "${INPUT_WIDTH:-640}" "${INPUT_HEIGHT:-360}" \
  "${CUDA_VISIBLE_DEVICES}" "${NPROC_PER_NODE}" "${OUTPUT_DIR}"
if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  exec torchrun --standalone --nnodes=1 --nproc_per_node "${NPROC_PER_NODE}" \
    --master_port "${MASTER_PORT}" train_vspw_mixed.py "${args[@]}"
fi
exec python -u train_vspw_mixed.py "${args[@]}"
