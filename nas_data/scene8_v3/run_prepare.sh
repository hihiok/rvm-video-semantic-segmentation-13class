#!/usr/bin/env bash
set -Eeuo pipefail
if [[ "${CONDA_DEFAULT_ENV:-}" != "ultraface" && "${CONDA_DEFAULT_ENV:-}" != "Ultraface" && "${CONDA_DEFAULT_ENV:-}" != "ultraface_new" ]]; then
  echo 'HUMAN_ACTION_REQUIRED: YES; activate the existing ultraface_new (or legacy ultraface/Ultraface) environment' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python -u "$HERE/prepare_all.py" "$@"
