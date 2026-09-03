#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/rvm_vspw_rvm_residual_v1_13class_640x360}"

# Keep CPU usage conservative on the shared server.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export WORKERS="${WORKERS:-1}"

# The frozen Stage-1 path is evaluated independently per frame. Only this new
# low-resolution residual adapter is optimized; reset frames are exact Stage-1.
export TEMPORAL_RESIDUAL_ADAPTER=1
export TEMPORAL_HIDDEN_CHANNELS="${TEMPORAL_HIDDEN_CHANNELS:-16}"
export TEMPORAL_ADAPTER_SCALE="${TEMPORAL_ADAPTER_SCALE:-0.25}"
export STAGE2_TRAINABLE_SCOPE=temporal_residual
export STAGE3_TRAINABLE_SCOPE=temporal_residual

export STAGE2_EPOCHS="${STAGE2_EPOCHS:-10}"
export STAGE3_EPOCHS="${STAGE3_EPOCHS:-15}"
export STAGE2_CLIP_LENGTH="${STAGE2_CLIP_LENGTH:-5}"
export STAGE3_CLIP_LENGTH="${STAGE3_CLIP_LENGTH:-8}"
export STAGE2_VIDEO_BATCHES="${STAGE2_VIDEO_BATCHES:-1}"
export STAGE3_VIDEO_BATCHES="${STAGE3_VIDEO_BATCHES:-1}"

# A T=1 batch is strictly bypassed and has no trainable path. Static data is
# retained for validation, while all optimizer steps use annotated video clips.
export STAGE2_STATIC_BATCHES=0
export STAGE3_STATIC_BATCHES=0

# CE + Dice remain enabled by train_vspw_mixed.sh. Add the spatial boundary,
# stable-interior, and RVM derivative-matching objectives without distillation.
export STAGE2_LAPLACIAN_WEIGHT="${STAGE2_LAPLACIAN_WEIGHT:-0.05}"
export STAGE3_LAPLACIAN_WEIGHT="${STAGE3_LAPLACIAN_WEIGHT:-0.05}"
export STAGE2_TEMPORAL_WEIGHT="${STAGE2_TEMPORAL_WEIGHT:-0.02}"
export STAGE3_TEMPORAL_WEIGHT="${STAGE3_TEMPORAL_WEIGHT:-0.05}"
export STAGE2_RVM_TEMPORAL_WEIGHT="${STAGE2_RVM_TEMPORAL_WEIGHT:-0.05}"
export STAGE3_RVM_TEMPORAL_WEIGHT="${STAGE3_RVM_TEMPORAL_WEIGHT:-0.05}"
export LAPLACIAN_LEVELS="${LAPLACIAN_LEVELS:-5}"
export RVM_TEMPORAL_BETA="${RVM_TEMPORAL_BETA:-0.1}"
export TEMPORAL_BOUNDARY_RADIUS="${TEMPORAL_BOUNDARY_RADIUS:-2}"
export TEMPORAL_TEMPERATURE="${TEMPORAL_TEMPERATURE:-1.0}"

export STATIC_RETENTION_TOLERANCE="${STATIC_RETENTION_TOLERANCE:-0.0}"
export PREDICTION_FLIP_PENALTY="${PREDICTION_FLIP_PENALTY:-0.1}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export SAVE_EVERY="${SAVE_EVERY:-5}"

# This experiment intentionally uses unweighted CE and no teacher distillation.
unset CLASS_WEIGHTS

exec bash "${SCRIPT_DIR}/train_vspw_mixed.sh"
