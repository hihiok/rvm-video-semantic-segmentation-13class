#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

VSPW_ROOT="${VSPW_ROOT:-/data/pub1/z00919662/segmentation/datasets/VSPW_13cls}"
STATIC_ROOT="${STATIC_ROOT:-/data/pub1/z00919662/segmentation/datasets/COCO_ADE_13cls_16x9_640x360}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/rvm_vspw_temporal_preserve_13class_640x360}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_DIR}/best_spatial_preserved.pth}"
BENCHMARK_DIR="${BENCHMARK_DIR:-${OUTPUT_DIR}/guided_upsample_benchmark}"

test -f "${CHECKPOINT}"
mkdir -p "${BENCHMARK_DIR}"

common=(
  --checkpoint "${CHECKPOINT}"
  --base-scale "${GUIDED_BASE_SCALE:-0.5}"
  --guided-radius "${GUIDED_RADIUS:-1}"
  --guided-eps "${GUIDED_EPS:-1e-4}"
  --boundary-tolerance "${BOUNDARY_TOLERANCE:-2}"
  --max-samples "${BENCHMARK_MAX_SAMPLES:-500}"
  --batch-size "${BENCHMARK_BATCH_SIZE:-1}"
  --workers "${BENCHMARK_WORKERS:-2}"
  --device "${BENCHMARK_DEVICE:-cuda}"
)

python -u tools/benchmark_guided_upsample.py \
  "${common[@]}" \
  --images "${VSPW_ROOT}/images/val" \
  --annotations "${VSPW_ROOT}/annotations/val" \
  --output-json "${BENCHMARK_DIR}/vspw_val.json"

python -u tools/benchmark_guided_upsample.py \
  "${common[@]}" \
  --images "${STATIC_ROOT}/images/val" \
  --annotations "${STATIC_ROOT}/annotations/val" \
  --output-json "${BENCHMARK_DIR}/static_val.json"

printf 'Guided-upsample benchmark results: %s\n' "${BENCHMARK_DIR}"
