#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

VSPW_ROOT="${VSPW_ROOT:-/data/pub1/z00919662/segmentation/datasets/VSPW_13cls}"
STATIC_ROOT="${STATIC_ROOT:-/data/pub1/z00919662/segmentation/datasets/COCO_ADE_13cls_16x9_640x360}"
TEACHER_CACHE_ROOT="${TEACHER_CACHE_ROOT:-/data/pub1/z00919662/segmentation/datasets/OneFormer_ADE20K_RVM13_cache_45x80}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/pub1/z00919662/segmentation/rvm-video-semantic-segmentation-13class-rvm-residual-v1/output/rvm_vspw_rvm_residual_v1_13class_640x360}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/pub1/z00919662/segmentation/rvm-video-semantic-segmentation-13class-rvm-residual-v1/output/rvm_oneformer_distill_v1_13class_640x360}"

if [[ -z "${INIT_CHECKPOINT:-}" && -z "${RESUME:-}" ]]; then
  # Validates class order and the 13-channel head; never searches another run.
  INIT_CHECKPOINT="$(python tools/select_distillation_checkpoint.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" --print-path-only)"
fi
if [[ -z "${INIT_CHECKPOINT:-}" && -z "${RESUME:-}" ]]; then
  printf 'ERROR: no approved best checkpoint exists under %s\n' "${CHECKPOINT_DIR}" >&2
  printf 'Set INIT_CHECKPOINT explicitly after inspecting the run; last.pth is not selected automatically.\n' >&2
  exit 2
fi
if [[ -n "${INIT_CHECKPOINT:-}" && ! -f "${INIT_CHECKPOINT}" ]]; then
  printf 'ERROR: initialization checkpoint not found: %s\n' "${INIT_CHECKPOINT}" >&2
  exit 2
fi
if [[ -n "${RESUME:-}" && ! -f "${RESUME}" ]]; then
  printf 'ERROR: distillation resume checkpoint not found: %s\n' "${RESUME}" >&2
  exit 2
fi

if [[ "${SKIP_CACHE_AUDIT:-0}" != "1" ]]; then
  STATIC_TRAIN_IMAGES="$(python - "${STATIC_ROOT}" <<'PY'
import sys
from dataset import resolve_static_split
print(resolve_static_split(sys.argv[1], "train").image_root)
PY
)"
  python tools/audit_oneformer_cache.py \
    --image-root "${VSPW_ROOT}/images/train" \
    --cache-root "${TEACHER_CACHE_ROOT}/vspw/train"
  python tools/audit_oneformer_cache.py \
    --image-root "${STATIC_TRAIN_IMAGES}" \
    --cache-root "${TEACHER_CACHE_ROOT}/static/train"
fi

MAX_GPUS="${MAX_GPUS:-2}"
MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB:-18000}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(python - "${MAX_GPUS}" "${MIN_FREE_GPU_MIB}" <<'PY'
import subprocess, sys
maximum, minimum = int(sys.argv[1]), int(sys.argv[2])
output = subprocess.check_output([
    "nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
    "--format=csv,noheader,nounits"], text=True)
candidates = []
for line in output.splitlines():
    index, free, utilization = (int(item.strip()) for item in line.split(","))
    if free >= minimum and utilization <= 25:
        candidates.append((free, index))
candidates.sort(reverse=True)
if not candidates:
    raise SystemExit(f"No GPU has >= {minimum} MiB free and utilization <= 25%")
print(",".join(str(index) for _, index in candidates[:maximum]))
PY
)"
fi
export CUDA_VISIBLE_DEVICES
NPROC_PER_NODE="$(python -c 'import os; print(len([x for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip()]))')"

args=(
  --data-root "${VSPW_ROOT}"
  --static-root "${STATIC_ROOT}"
  --teacher-cache-root "${TEACHER_CACHE_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --input-width 640 --input-height 360
  --stage2-epochs "${STAGE2_EPOCHS:-10}"
  --stage3-epochs "${STAGE3_EPOCHS:-20}"
  --stage2-clip-length "${STAGE2_CLIP_LENGTH:-5}"
  --stage3-clip-length "${STAGE3_CLIP_LENGTH:-8}"
  --stage2-video-batches "${STAGE2_VIDEO_BATCHES:-1}"
  --stage2-static-batches "${STAGE2_STATIC_BATCHES:-1}"
  --stage3-video-batches "${STAGE3_VIDEO_BATCHES:-2}"
  --stage3-static-batches "${STAGE3_STATIC_BATCHES:-1}"
  --stage2-trainable-scope "${STAGE2_TRAINABLE_SCOPE:-all}"
  --stage3-trainable-scope "${STAGE3_TRAINABLE_SCOPE:-all}"
  --stage2-kd-weight "${STAGE2_KD_WEIGHT:-0.50}"
  --stage3-kd-weight "${STAGE3_KD_WEIGHT:-0.20}"
  --stage2-temporal-weight "${STAGE2_TEMPORAL_WEIGHT:-0.03}"
  --stage3-temporal-weight "${STAGE3_TEMPORAL_WEIGHT:-0.08}"
  --kd-temperature "${KD_TEMPERATURE:-2.0}"
  --kd-confidence-threshold "${KD_CONFIDENCE_THRESHOLD:-0.60}"
  --kd-background-weight "${KD_BACKGROUND_WEIGHT:-0.25}"
  --kd-disagreement-weight "${KD_DISAGREEMENT_WEIGHT:-0.25}"
  --learning-rate "${LEARNING_RATE:-2e-5}"
  --backbone-learning-rate "${BACKBONE_LEARNING_RATE:-2e-6}"
  --batch-size "${VIDEO_BATCH_SIZE:-2}"
  --static-batch-size "${STATIC_BATCH_SIZE:-4}"
  --workers "${WORKERS:-4}"
  --static-retention-tolerance "${STATIC_RETENTION_TOLERANCE:-0.02}"
)
if [[ -n "${RESUME:-}" ]]; then
  args+=(--resume "${RESUME}")
else
  args+=(--init-checkpoint "${INIT_CHECKPOINT}")
fi

mkdir -p "${OUTPUT_DIR}"
printf 'INIT_CHECKPOINT=%s\nRESUME=%s\nTEACHER_CACHE_ROOT=%s\nOUTPUT_DIR=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
  "${INIT_CHECKPOINT:-}" "${RESUME:-}" "${TEACHER_CACHE_ROOT}" "${OUTPUT_DIR}" "${CUDA_VISIBLE_DEVICES}"
if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  exec torchrun --standalone --nnodes=1 --nproc_per_node "${NPROC_PER_NODE}" \
    --master_port "${MASTER_PORT:-29647}" train_vspw_mixed.py "${args[@]}"
fi
exec python -u train_vspw_mixed.py "${args[@]}"
