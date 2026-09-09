#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

VSPW_ROOT="${VSPW_ROOT:-/data/pub1/z00919662/segmentation/datasets/VSPW_13cls}"
STATIC_ROOT="${STATIC_ROOT:-/data/pub1/z00919662/segmentation/datasets/COCO_ADE_13cls_16x9_640x360}"
TEACHER_CACHE_ROOT="${TEACHER_CACHE_ROOT:-/data/pub1/z00919662/segmentation/datasets/OneFormer_ADE20K_RVM13_cache_45x80}"
MAPPING="${MAPPING:-configs/oneformer_ade20k_to_13class.json}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${TEACHER_BATCH_SIZE:-1}"

if [[ ! -d "${VSPW_ROOT}/images/train" ]]; then
  printf 'ERROR: VSPW train images not found: %s\n' "${VSPW_ROOT}/images/train" >&2
  exit 2
fi
STATIC_TRAIN_IMAGES="$(python - "${STATIC_ROOT}" <<'PY'
import sys
from dataset import resolve_static_split
print(resolve_static_split(sys.argv[1], "train").image_root)
PY
)"

python tools/cache_oneformer_teacher.py \
  --image-root "${VSPW_ROOT}/images/train" \
  --cache-root "${TEACHER_CACHE_ROOT}/vspw/train" \
  --mapping "${MAPPING}" --device "${DEVICE}" --batch-size "${BATCH_SIZE}"
python tools/cache_oneformer_teacher.py \
  --image-root "${STATIC_TRAIN_IMAGES}" \
  --cache-root "${TEACHER_CACHE_ROOT}/static/train" \
  --mapping "${MAPPING}" --device "${DEVICE}" --batch-size "${BATCH_SIZE}"

python tools/audit_oneformer_cache.py \
  --image-root "${VSPW_ROOT}/images/train" \
  --cache-root "${TEACHER_CACHE_ROOT}/vspw/train" \
  --output-json "${TEACHER_CACHE_ROOT}/vspw_train_audit.json"
python tools/audit_oneformer_cache.py \
  --image-root "${STATIC_TRAIN_IMAGES}" \
  --cache-root "${TEACHER_CACHE_ROOT}/static/train" \
  --output-json "${TEACHER_CACHE_ROOT}/static_train_audit.json"
