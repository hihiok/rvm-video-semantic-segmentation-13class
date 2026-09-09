#!/usr/bin/env bash
set -Eeuo pipefail
if [[ "${CONDA_DEFAULT_ENV:-}" != "Ultraface" ]]; then
  echo 'STATUS: BLOCKED; HUMAN_ACTION_REQUIRED: YES; activate existing Ultraface environment.' >&2
  exit 2
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python -u "${HERE}/prepare.py" "$@"
