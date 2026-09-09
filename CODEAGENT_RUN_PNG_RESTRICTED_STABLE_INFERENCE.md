# CodeAgent: restricted-class, scene-cut-aware, stable PNG inference

## Scope

Pull the prepared GitHub code and run inference on one business PNG sequence.
Do not train, edit source code, create commits, or push. Stop and report if a
preflight check or test fails.

Repository and branch:

```text
https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git
agent/rvm13-rvm-loss-temporal-residual-v1
```

The expected inference features are:

```text
--output-classes
--restricted-min-confidence
--temporal-ema-alpha
--temporal-hysteresis-margin
--scene-cut-method histogram
```

## Proxy and SSL

Never enable `set -x` and never print proxy credentials. The corporate proxy
URL must already be supplied securely in `CORPORATE_PROXY_URL`; do not commit it
to this public repository.

```bash
test -n "${CORPORATE_PROXY_URL:-}" || {
  echo "HUMAN_ACTION_REQUIRED: set CORPORATE_PROXY_URL securely"
  exit 1
}

export http_proxy="${CORPORATE_PROXY_URL}"
export https_proxy="${CORPORATE_PROXY_URL}"
export HTTP_PROXY="${CORPORATE_PROXY_URL}"
export HTTPS_PROXY="${CORPORATE_PROXY_URL}"

git config --global http.proxy "${CORPORATE_PROXY_URL}"
git config --global https.proxy "${CORPORATE_PROXY_URL}"
git config --global http.sslVerify false
```

The proxy/SSL settings are only for accessing GitHub behind the corporate
HTTPS-inspection proxy. Keep SSL verification enabled for unrelated workflows
when possible.

## Fixed paths

```bash
export PROJECT_ROOT="/data/pub1/z00919662/segmentation/rvm-video-semantic-segmentation-13class-rvm-residual-v1"

export MODEL_OUTPUT_DIR="${PROJECT_ROOT}/output/rvm_vspw_rvm_residual_v1_13class_640x360"

export CHECKPOINT="${MODEL_OUTPUT_DIR}/best_spatial_preserved.pth"

export PNG_DIR="/data/pub1/z00919662/segmentation/datasets/qishuai_test/00001_ori002_650_0924_1920x1080_p444_10bit_SR_YUV444_RC1.0"

export INFERENCE_ROOT="${MODEL_OUTPUT_DIR}/business_png_sky_water_mountain_stable_v1"

export PREVIEW_FPS=30
```

`PREVIEW_FPS` only controls MP4 playback speed. It does not alter model input or
temporal processing.

## Update the prepared code

```bash
git -C "${PROJECT_ROOT}" status --short
```

If the output is not empty, stop. Do not overwrite or stash another agent's
changes. When the worktree is clean:

```bash
git -C "${PROJECT_ROOT}" fetch origin agent/rvm13-rvm-loss-temporal-residual-v1
git -C "${PROJECT_ROOT}" checkout agent/rvm13-rvm-loss-temporal-residual-v1
git -C "${PROJECT_ROOT}" merge --ff-only origin/agent/rvm13-rvm-loss-temporal-residual-v1

cd "${PROJECT_ROOT}"

test -z "$(git status --short)"
test -f inference_png_sequence_semantic.py
test -f tests/test_png_sequence_inference_standalone.py

for option in \
  output-classes \
  restricted-min-confidence \
  temporal-ema-alpha \
  temporal-hysteresis-margin \
  scene-cut-method; do
  grep -q -- "--${option}" inference_png_sequence_semantic.py
done
```

## Environment and CPU limit

```bash
source /home/z00919662/anaconda3/etc/profile.d/conda.sh
conda activate ultraface

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

which python
python -c 'import cv2,torch; print("cv2",cv2.__version__); print("torch",torch.__version__); print("cuda",torch.cuda.is_available())'
```

Required environment:

```text
/home/z00919662/anaconda3/envs/ultraface/bin/python
torch 2.4.1+cu121
CUDA available
CPU thread limit 1
```

## Validate the code

```bash
cd "${PROJECT_ROOT}"

python tests/test_png_sequence_inference_standalone.py -v

python -m py_compile \
  inference_png_sequence_semantic.py \
  inference_video_semantic.py
```

Expected result:

```text
12 tests OK, zero skipped
py_compile PASS
```

If any test is skipped or fails, stop and report. Do not modify the code.

## Validate the checkpoint and PNG sequence

```bash
test -f "${CHECKPOINT}"
test -d "${PNG_DIR}"

find "${PNG_DIR}" -type f -iname '*.png' | wc -l
find "${PNG_DIR}" -type f -iname '*.png' -printf '%h\n' | sort -u
find "${PNG_DIR}" -type f -iname '*.png' | sort -V | head -n 5
find "${PNG_DIR}" -type f -iname '*.png' | sort -V | tail -n 5
```

All PNG files must be in one frame directory. If multiple directories are
reported, stop instead of mixing sequences.

```bash
CHECKPOINT="${CHECKPOINT}" python - <<'PY'
import os
import torch

path = os.environ["CHECKPOINT"]
try:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(path, map_location="cpu")

print("checkpoint", path)
print("epoch", checkpoint.get("epoch"))
print("training_stage", checkpoint.get("training_stage"))
print("input", checkpoint.get("input_width"), checkpoint.get("input_height"))
print("class_names", checkpoint.get("class_names"))
print("temporal_residual_adapter", checkpoint.get("temporal_residual_adapter"))

assert checkpoint.get("input_width") == 640
assert checkpoint.get("input_height") == 360
assert checkpoint.get("class_names") == [
    "background", "sky", "person", "plant", "building", "flower",
    "food", "water", "desert", "ice_or_snow", "text", "ball",
    "mountain",
]
assert checkpoint.get("temporal_residual_adapter") is True
PY
```

## Select an idle GPU

```bash
pgrep -af 'train_vspw_mixed.py|torchrun' || true

nvidia-smi \
  --query-gpu=index,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader
```

Choose one idle GPU without stopping any training process. Example:

```bash
export INFERENCE_GPU=0
```

Replace `0` with the actual idle GPU index.

## 32-frame smoke inference

The output mask IDs remain compatible with the original palette:

```text
0  other
1  sky
7  water
12 mountain
```

```bash
export SMOKE_DIR="${INFERENCE_ROOT}/smoke_32frames"

test ! -e "${SMOKE_DIR}"
mkdir -p "${SMOKE_DIR}"

CUDA_VISIBLE_DEVICES="${INFERENCE_GPU}" \
python -u inference_png_sequence_semantic.py \
  --checkpoint "${CHECKPOINT}" \
  --input-dir "${PNG_DIR}" \
  --output-dir "${SMOKE_DIR}" \
  --fps "${PREVIEW_FPS}" \
  --max-frames 32 \
  --device cuda \
  --input-width 640 \
  --input-height 360 \
  --resize-mode letterbox \
  --upsample-mode bilinear \
  --overlay-alpha 0.5 \
  --recurrent \
  --output-classes sky water mountain other \
  --restricted-min-confidence 0.20 \
  --temporal-ema-alpha 0.20 \
  --temporal-hysteresis-margin 0.08 \
  --scene-cut-method histogram \
  --scene-cut-threshold 0.35 \
  --reset-interval 0 \
  --save-masks \
  --no-amp \
  2>&1 | tee "${SMOKE_DIR}/inference.log"
```

Validate smoke artifacts and mask IDs:

```bash
test -s "${SMOKE_DIR}/${PNG_DIR##*/}.mp4"
test -s "${SMOKE_DIR}/${PNG_DIR##*/}.summary.json"

SMOKE_DIR="${SMOKE_DIR}" PNG_NAME="${PNG_DIR##*/}" python - <<'PY'
import json
import os
from pathlib import Path

import cv2
import numpy as np

root = Path(os.environ["SMOKE_DIR"])
name = os.environ["PNG_NAME"]
masks = sorted((root / f"{name}_masks").glob("*.png"))
assert len(masks) == 32, len(masks)

ids = set()
for path in masks:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert mask is not None and mask.ndim == 2, path
    ids.update(int(value) for value in np.unique(mask))
assert ids <= {0, 1, 7, 12}, ids

summary = json.loads((root / f"{name}.summary.json").read_text())
assert summary["frames"] == 32
assert summary["output_classes"] == ["sky", "water", "mountain", "other"]
print("mask_ids", sorted(ids))
print(json.dumps(summary, indent=2))
PY
```

Stop if the smoke run fails.

## Full business-sequence inference

Use a fresh Python process. Do not reuse state from the smoke run.

```bash
export FULL_DIR="${INFERENCE_ROOT}/full_sequence"

test ! -e "${FULL_DIR}"
mkdir -p "${FULL_DIR}"

CUDA_VISIBLE_DEVICES="${INFERENCE_GPU}" \
python -u inference_png_sequence_semantic.py \
  --checkpoint "${CHECKPOINT}" \
  --input-dir "${PNG_DIR}" \
  --output-dir "${FULL_DIR}" \
  --fps "${PREVIEW_FPS}" \
  --device cuda \
  --input-width 640 \
  --input-height 360 \
  --resize-mode letterbox \
  --upsample-mode bilinear \
  --overlay-alpha 0.5 \
  --recurrent \
  --output-classes sky water mountain other \
  --restricted-min-confidence 0.20 \
  --temporal-ema-alpha 0.20 \
  --temporal-hysteresis-margin 0.08 \
  --scene-cut-method histogram \
  --scene-cut-threshold 0.35 \
  --reset-interval 0 \
  --no-amp \
  2>&1 | tee "${FULL_DIR}/inference.log"
```

The histogram scene-cut detector must clear both:

```text
1. all four network recurrent hidden states
2. EMA and hysteresis post-processing state
```

## Validate outputs

```bash
export OUTPUT_MP4="${FULL_DIR}/${PNG_DIR##*/}.mp4"
export OUTPUT_JSONL="${FULL_DIR}/${PNG_DIR##*/}.jsonl"
export OUTPUT_SUMMARY="${FULL_DIR}/${PNG_DIR##*/}.summary.json"

test -s "${OUTPUT_MP4}"
test -s "${OUTPUT_JSONL}"
test -s "${OUTPUT_SUMMARY}"

ffprobe -v error \
  -show_entries stream=width,height,r_frame_rate,nb_frames \
  -show_entries format=duration,size \
  -of json "${OUTPUT_MP4}"

cat "${OUTPUT_SUMMARY}"
```

The summary reports scene-cut reset count and raw/stabilized low-resolution flip
rates. The stabilized flip rate should normally be lower than the raw flip rate.
Do not automatically change thresholds when this is not true; report the actual
numbers and preserve all artifacts.

## Final report

```text
STATUS:
GITHUB_BRANCH:
GITHUB_COMMIT:
SOURCE_CODE_MODIFIED: NO
WORKTREE_CLEAN:

PYTHON_ENVIRONMENT:
TORCH_VERSION:
CUDA_VISIBLE_DEVICES:
CPU_THREAD_LIMIT: 1

CHECKPOINT:
CHECKPOINT_EPOCH:
PNG_DIR:
PNG_FRAME_COUNT:
OUTPUT_CLASSES: sky, water, mountain, other
OUTPUT_MASK_IDS: other=0, sky=1, water=7, mountain=12
RESTRICTED_MIN_CONFIDENCE: 0.20
TEMPORAL_EMA_ALPHA: 0.20
TEMPORAL_HYSTERESIS_MARGIN: 0.08
SCENE_CUT_METHOD: histogram
SCENE_CUT_THRESHOLD: 0.35

TEST_STATUS:
SMOKE_STATUS:
OUTPUT_MP4:
OUTPUT_JSONL:
OUTPUT_SUMMARY:
OUTPUT_FRAME_COUNT:
SCENE_CUT_RESET_COUNT:
RAW_FLIP_RATE_LOWRES:
STABILIZED_FLIP_RATE_LOWRES:
NETWORK_MS_PER_FRAME:
WARNINGS:
HUMAN_ACTION_REQUIRED:
```

When inference and validations finish successfully:

```text
STATUS: SUCCESS
HUMAN_ACTION_REQUIRED: YES - visually inspect the MP4 for desired stability and acceptable lag
```

