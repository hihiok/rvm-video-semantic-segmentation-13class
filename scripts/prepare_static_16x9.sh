#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SOURCE_PROJECT="${SOURCE_PROJECT:-/data/pub1/z00919662/segmentation/RobustVideoMatting-semantic-13class-512}"
ORIGINAL_STATIC_ROOT="${ORIGINAL_STATIC_ROOT:-${SOURCE_PROJECT}/data}"
PREPARED_STATIC_ROOT="${PREPARED_STATIC_ROOT:-/data/pub1/z00919662/segmentation/datasets/COCO_ADE_13cls_16x9_640x360}"

if [[ ! -d "${ORIGINAL_STATIC_ROOT}" ]]; then
  printf 'ERROR: original COCO+ADE13 dataset does not exist: %s\n' "${ORIGINAL_STATIC_ROOT}" >&2
  exit 2
fi

exec python -u tools/prepare_static_16x9.py \
  --source-root "${ORIGINAL_STATIC_ROOT}" \
  --output-root "${PREPARED_STATIC_ROOT}" \
  --width "${INPUT_WIDTH:-640}" \
  --height "${INPUT_HEIGHT:-360}" \
  --crop-probability "${STATIC_CROP_PROBABILITY:-0.5}" \
  --min-foreground-retention "${STATIC_MIN_FOREGROUND_RETENTION:-0.45}" \
  --crop-attempts "${STATIC_CROP_ATTEMPTS:-12}" \
  --workers "${PREPARE_WORKERS:-8}" \
  --jpeg-quality "${STATIC_JPEG_QUALITY:-95}" \
  --seed "${STATIC_PREPARE_SEED:-1337}"
