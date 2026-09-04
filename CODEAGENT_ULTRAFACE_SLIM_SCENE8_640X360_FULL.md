# CodeAgent — Original UltraFace Slim 8-label Multi-label Scene Classification, 640x360

This runbook is self-contained. Do not rely on prior chat context.

## 0. Goal

Train an 8-label multi-label scene classifier using the **original UltraFace slim / Mb_Tiny backbone topology**, not the user's old FSD code.

Labels in exact order:

1. `night` — 夜景
2. `indoor` — 室内
3. `rain_snow` — 雨/雪
4. `office` — 办公场景
5. `outdoor` — 户外
6. `landscape` — 风景
7. `sports` — 运动
8. `objective_image` — 客观图（电脑合成 pattern、解析度卡、测试卡等）

This is multi-label, not softmax classification. One image may have several positives.

Label convention: `1=positive`, `0=confirmed negative`, `-1=unknown`. Unknown labels MUST be masked from loss and metrics.

Input is fixed to **RGB 3x360x640**, horizontal 16:9.

## 1. GitHub

Repository:

`https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git`

Branch:

`agent/ultraface-slim-scene-multilabel8-640x360-v1`

Checkout directory:

`/data/pub1/z00919662/segmentation/ultraface-slim-scene-multilabel8-640x360-v1`

### Mandatory repository update rule

Before doing any work:

```bash
export REPO_URL=https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git
export BRANCH=agent/ultraface-slim-scene-multilabel8-640x360-v1
export PROJECT_ROOT=/data/pub1/z00919662/segmentation/ultraface-slim-scene-multilabel8-640x360-v1

if [[ -d "$PROJECT_ROOT/.git" ]]; then
  cd "$PROJECT_ROOT"
  if [[ -n "$(git status --porcelain)" ]]; then
    echo 'ERROR: working tree is not clean'
    git status --short
    exit 2
  fi
  git fetch origin "$BRANCH"
  git checkout "$BRANCH"
  git merge --ff-only "origin/$BRANCH"
else
  git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$PROJECT_ROOT"
  cd "$PROJECT_ROOT"
fi

LOCAL_HEAD=$(git rev-parse HEAD)
REMOTE_HEAD=$(git rev-parse "origin/$BRANCH")
echo LOCAL_HEAD=$LOCAL_HEAD
echo REMOTE_HEAD=$REMOTE_HEAD
[[ "$LOCAL_HEAD" == "$REMOTE_HEAD" ]]
[[ -z "$(git status --porcelain)" ]]
```

Never stash, reset --hard, locally patch, or locally commit a fix. If prepared GitHub code needs modification, STOP and report it.

## 2. Environment — MUST use existing `Ultraface`

Do NOT create a new conda environment or venv.

Locate existing conda and activate environment **exactly named `Ultraface`**.

Preferred:

```bash
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  CONDA_BIN=$(find /data/pub1/z00919662 /mnt/ssd1/z00919662 -maxdepth 5 -type f -path '*/bin/conda' 2>/dev/null | head -1)
  if [[ -z "$CONDA_BIN" ]]; then
    echo 'ERROR: conda not found'
    exit 2
  fi
  source "$(dirname "$(dirname "$CONDA_BIN")")/etc/profile.d/conda.sh"
fi

conda env list
conda activate Ultraface
[[ "${CONDA_DEFAULT_ENV:-}" == "Ultraface" ]]

which python
python - <<'PY'
import sys, torch, cv2, numpy
print('python', sys.version)
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available())
print('cv2', cv2.__version__)
assert torch.cuda.is_available()
PY
```

Do not upgrade/downgrade torch or torchvision.

Set CPU limits:

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
```

## 3. Model definition

The model is in:

`ultraface_scene_multilabel8/model.py`

It is a standalone implementation of the original UltraFace slim `Mb_Tiny` backbone topology from upstream `Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB`, reference commit `dffdddda9794a50607cba8f318507a28c1c27cab`.

Architecture:

```text
RGB [B,3,360,640]
 -> original Mb_Tiny backbone (base_channel=16)
 -> AdaptiveAvgPool2d(1)
 -> Dropout(0.1)
 -> Linear(256,8)
 -> 8 raw logits
```

Important:

- no RFB
- no SSD extras
- no bbox regression head
- no face-detection classification head
- no FSD dependency
- no `fd_config.define_img_size()`
- no 640x480 conversion
- exact runtime/training tensor is `[B,3,360,640]`

Run static checks:

```bash
cd "$PROJECT_ROOT"
python -m compileall -q ultraface_scene_multilabel8 fsd_scene_multilabel8
bash -n ultraface_scene_multilabel8/run_smoke_ultraface_env.sh
bash -n ultraface_scene_multilabel8/run_train.sh

cd ultraface_scene_multilabel8
python model.py
```

Must print output shape `(1, 8)`. Any shape other than `[1,8]` is a HARD FAIL.

## 4. Existing datasets — READ ONLY

Use exactly:

```text
COCO:
/data/pub1/z00919662/segmentation/datasets/coco

Places365:
/data/pub1/z00919662/segmentation/datasets/places365

COCO+ADE20K 13-class segmentation:
/data/pub1/z00919662/segmentation/datasets/COCO_ADE_13cls_16x9_640x360

10_scenes:
/data/pub1/z00919662/dataset/10_scenes

Computer synthesized:
/data/pub1/z00919662/dataset/10_scenes/train/Computer_synthesized
```

Do NOT download any data.
Do NOT modify/copy/move/rename/resize-overwrite source images.
Do NOT modify existing labels, masks or annotations.

Only generate new JSONL manifests referencing original absolute paths.

Manifest root:

`/data/pub1/z00919662/dataset/UltraFaceSlim_8scene_multilabel_manifests_640x360_v1`

## 5. Manifest generation

The reusable source mapping tools are under `fsd_scene_multilabel8/`. Despite the directory name they are model-independent; they only create partial multi-label JSONL files.

```bash
export PLACES_ROOT=/data/pub1/z00919662/segmentation/datasets/places365
export COCO_ROOT=/data/pub1/z00919662/segmentation/datasets/coco
export SEG_ROOT=/data/pub1/z00919662/segmentation/datasets/COCO_ADE_13cls_16x9_640x360
export TEN_ROOT=/data/pub1/z00919662/dataset/10_scenes
export LABEL_ROOT=/data/pub1/z00919662/dataset/UltraFaceSlim_8scene_multilabel_manifests_640x360_v1

for p in "$PLACES_ROOT" "$COCO_ROOT" "$SEG_ROOT" "$TEN_ROOT"; do
  [[ -d "$p" ]] || { echo "MISSING $p"; exit 2; }
done
[[ -d "$TEN_ROOT/train/Computer_synthesized" ]] || { echo 'MISSING Computer_synthesized'; exit 2; }

rm -rf "$LABEL_ROOT"
mkdir -p "$LABEL_ROOT"
cd "$PROJECT_ROOT/fsd_scene_multilabel8"

python -u prepare_manifest.py \
  --places-root "$PLACES_ROOT" \
  --coco-root "$COCO_ROOT" \
  --seg-root "$SEG_ROOT" \
  --ten-scenes-root "$TEN_ROOT" \
  --output-root "$LABEL_ROOT" \
  --places-train-cap-per-class 500 \
  --coco-objective-neg-train-cap 5000 \
  --coco-objective-neg-eval-cap 1000 \
  --seed 20260904 \
  --snow-min-area 0.01 \
  --landscape-min-area 0.30 \
  2>&1 | tee "$LABEL_ROOT/prepare.log"

python -u finalize_manifest.py --data-root "$LABEL_ROOT" 2>&1 | tee "$LABEL_ROOT/finalize.log"
python -u audit_manifest.py --data-root "$LABEL_ROOT" 2>&1 | tee "$LABEL_ROOT/audit.log"
```

`Computer_synthesized` must provide `objective_image=1`.
Other 10_scenes folders may supplement relevant labels via prepared explicit mapping.

Do not convert unknown `-1` to `0`.

## 6. Dataset audit and leakage rule

Before training, report train/val/test for every label:

- positive
- negative
- unknown

Every label in every split must have at least one positive and one negative.

Leakage policy:

- train vs val underlying duplicate -> HARD FAIL
- train vs test underlying duplicate -> HARD FAIL
- val vs test duplicate -> WARNING ONLY, continue
- duplicate in same split -> WARNING ONLY

Do not stop solely for val/test overlap.

Also verify the manifest root contains labels/logs only; no copied image dataset.

## 7. GPU

Use one free GPU only. Do not kill another user's process.

```bash
nvidia-smi
```

Select a free physical GPU, then export e.g.:

```bash
export GPU=0
```

## 8. Smoke test

Activate `Ultraface`, then:

```bash
cd "$PROJECT_ROOT/ultraface_scene_multilabel8"
export DATA_ROOT="$LABEL_ROOT"
export GPU=<FREE_GPU_ID>
export BATCH=24
bash run_smoke_ultraface_env.sh
```

Smoke uses:

- 640x360
- RGB 3-channel
- batch 24
- workers 4
- AMP
- max 20 train steps
- max 8 eval batches

If CUDA OOM, retry only batch size in this order:

`24 -> 16 -> 8 -> 4`

Do NOT lower input resolution. Batch 4 OOM -> HARD FAIL.

Smoke PASS requires:

- input tensor `[B,3,360,640]`
- model output `[B,8]`
- forward/backward works
- masked BCE works
- validation works
- `last_train_state.pth` and `best_train_state.pth` written
- final calibration/test code completes

## 9. Formal training

Output root:

`/data/pub1/z00919662/scene_multilabel/ultraface_slim_8label_640x360_v1/train`

Use batch size proven by smoke.

```bash
cd "$PROJECT_ROOT/ultraface_scene_multilabel8"
export DATA_ROOT="$LABEL_ROOT"
export OUT=/data/pub1/z00919662/scene_multilabel/ultraface_slim_8label_640x360_v1/train
export GPU=<FREE_GPU_ID>
export BATCH=<SMOKE_PASS_BATCH>
bash run_train.sh
```

Formal training:

- original UltraFace slim Mb_Tiny backbone topology
- input 3x360x640
- masked BCEWithLogitsLoss
- SGD
- lr 1e-2
- momentum 0.9
- weight_decay 1e-4
- MultiStepLR milestones 95,150
- gamma 0.1
- epochs 200
- workers 4
- AMP

## 10. Resume

If interrupted, never restart from epoch 0 when a valid checkpoint exists.

```bash
export RESUME="$OUT/last_train_state.pth"
bash run_train.sh
```

## 11. Final evaluation

Thresholds are calibrated independently per label on validation and then fixed for test.

Report per label:

- threshold
- Precision
- Recall
- F1
- Accuracy
- Balanced Accuracy
- AP
- known / positive / negative counts

Report aggregate:

- macro-F1
- macro-balanced-accuracy
- macro-AP

Primary deployment checkpoint:

`$OUT/best_ultraface_slim_multilabel8_640x360.pth`

Also report `best_train_state.pth`, `last_train_state.pth`, `thresholds.json`, `test_per_class_calibrated.csv`, `test_summary.json`.

## 12. ONNX export

Do not modify the `Ultraface` environment just to install packages. First check:

```bash
python -c 'import onnx; print(onnx.__version__)'
```

If `onnx` already exists:

```bash
cd "$PROJECT_ROOT/ultraface_scene_multilabel8"
python export_onnx.py \
  --checkpoint "$OUT/best_ultraface_slim_multilabel8_640x360.pth" \
  --output "$OUT/ultraface_slim_scene8_640x360.onnx"

python - <<'PY'
import onnx
p='/data/pub1/z00919662/scene_multilabel/ultraface_slim_8label_640x360_v1/train/ultraface_slim_scene8_640x360.onnx'
m=onnx.load(p); onnx.checker.check_model(m)
print('ONNX_CHECK=PASS')
print('OPS=',sorted(set(n.op_type for n in m.graph.node)))
PY
```

If `onnx` is absent, report `ONNX_SKIPPED_MISSING_EXISTING_DEPENDENCY` as a warning. Do not pip-install/upgrade packages unless the user explicitly requests it.

## 13. HARD STOP rules

Stop and output `HUMAN_ACTION_REQUIRED: YES` if any occurs:

1. Git working tree not clean before sync.
2. Git fetch/checkout/ff-only update fails.
3. Local HEAD != origin branch HEAD.
4. Existing conda environment `Ultraface` is missing or cannot activate.
5. `torch.cuda.is_available()` is false in `Ultraface`.
6. Any required dataset root is missing.
7. `Computer_synthesized` is missing.
8. Manifest builder has a code/layout error.
9. Any of 8 labels lacks positive or negative supervision in train/val/test.
10. train/val underlying leakage exists.
11. train/test underlying leakage exists.
12. model.py does not produce `[1,8]` for `[1,3,360,640]`.
13. Batch 24/16/8/4 all OOM in smoke.
14. Non-finite loss or genuine source-code error occurs.
15. Prepared code must be changed to continue.

When blocked, do NOT patch code locally. Return failed command, full traceback, file/line, relevant paths, branch/commit, Python/Torch/CUDA versions.

Warnings that must NOT stop training:

- val/test overlap
- same-split duplicates
- batch 24 OOM if a smaller permitted batch passes
- ONNX package absent after successful training

## 14. Final report

Return:

```text
STATUS: PASS / FAIL
HUMAN_ACTION_REQUIRED: YES / NO

GITHUB_BRANCH:
GITHUB_COMMIT:
CONDA_ENV: Ultraface
PYTHON:
TORCH:
GPU:

MODEL: original UltraFace slim Mb_Tiny backbone + GAP + FC8
INPUT: [B,3,360,640]
PARAMETERS:
MACS/FLOPS_IF_MEASURED:

DATA_ROOTS:
PLACES_ROOT:
COCO_ROOT:
SEG_ROOT:
TEN_SCENES_ROOT:
COMPUTER_SYNTHESIZED_ROOT:

MANIFEST_ROOT:
train_records:
val_records:
test_records:

PER_CLASS_COVERAGE:
label | train pos/neg/unknown | val pos/neg/unknown | test pos/neg/unknown

LEAKAGE:
train/val underlying overlap:
train/test underlying overlap:
val/test underlying overlap:

SMOKE_STATUS:
SMOKE_BATCH:

TRAIN_EPOCHS_COMPLETED:
BEST_EPOCH:
BEST_VAL_MACRO_F1:

TEST_PER_CLASS:
label | threshold | precision | recall | F1 | accuracy | balanced_accuracy | AP | known(pos/neg)

MACRO_F1:
MACRO_BALANCED_ACCURACY:
MACRO_AP:

DEPLOY_CHECKPOINT:
THRESHOLDS_JSON:
TEST_CSV:
ONNX_STATUS:
ONNX_PATH:

WARNINGS:
```

If complete, output `HUMAN_ACTION_REQUIRED: NO`.
