#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Conservative follow-up curriculum: keep every spatial parameter and its
# BatchNorm statistics frozen, train only ConvGRUs, and require zero static-mIoU
# regression before a checkpoint can become best_spatial_preserved.pth.
export STAGE2_EPOCHS="${STAGE2_EPOCHS:-10}"
export STAGE3_EPOCHS="${STAGE3_EPOCHS:-15}"
export STAGE2_CLIP_LENGTH="${STAGE2_CLIP_LENGTH:-5}"
export STAGE3_CLIP_LENGTH="${STAGE3_CLIP_LENGTH:-8}"
export STAGE2_VIDEO_BATCHES="${STAGE2_VIDEO_BATCHES:-1}"
export STAGE2_STATIC_BATCHES="${STAGE2_STATIC_BATCHES:-1}"
export STAGE3_VIDEO_BATCHES="${STAGE3_VIDEO_BATCHES:-1}"
export STAGE3_STATIC_BATCHES="${STAGE3_STATIC_BATCHES:-1}"
export STAGE2_TRAINABLE_SCOPE="${STAGE2_TRAINABLE_SCOPE:-recurrent}"
export STAGE3_TRAINABLE_SCOPE="${STAGE3_TRAINABLE_SCOPE:-recurrent}"
export STAGE2_TEMPORAL_WEIGHT="${STAGE2_TEMPORAL_WEIGHT:-0.05}"
export STAGE3_TEMPORAL_WEIGHT="${STAGE3_TEMPORAL_WEIGHT:-0.10}"
export TEMPORAL_BOUNDARY_RADIUS="${TEMPORAL_BOUNDARY_RADIUS:-2}"
export TEMPORAL_TEMPERATURE="${TEMPORAL_TEMPERATURE:-1.0}"
export STATIC_RETENTION_TOLERANCE="${STATIC_RETENTION_TOLERANCE:-0.0}"
export PREDICTION_FLIP_PENALTY="${PREDICTION_FLIP_PENALTY:-0.1}"

# Intentionally leave CLASS_WEIGHTS unset: the established CE + Dice objective
# stays unweighted while the temporal-only experiment isolates recurrence.
unset CLASS_WEIGHTS

exec bash "${SCRIPT_DIR}/train_vspw_mixed.sh"
