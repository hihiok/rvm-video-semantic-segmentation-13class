# Task: train UltraFace-slim 9-label multi-label scene classifier using existing datasets only

## Goal

Use the already prepared GitHub code in this branch and the three existing datasets below. Do not download any new dataset and do not copy/move source images. Only generate label manifest files (`train.jsonl`, `val.jsonl`, `test.jsonl`) plus audit/statistics files, then train/evaluate/export the 9-label model.

Nine labels:

1. indoor / 室内
2. outdoor / 户外
3. landscape / 风景
4. sports / 运动
5. food / 美食
6. animal / 动物
7. building / 建筑
8. sky / 蓝天（semantic definition: visible sky of any color）
9. office / 办公

## Hard restrictions

- DO NOT download Places365, COCO, ADE20K, OpenImages, or any other dataset.
- DO NOT run `nas_scene_multilabel/scripts/download_datasets.sh`.
- DO NOT modify, rename, copy, delete, or reorganize source dataset images/masks/annotations.
- DO NOT create a duplicate image dataset. Label manifests must reference the existing absolute image paths.
- DO NOT change Python/model/training code unless a real code bug is found; if a bug is found, stop and report traceback + exact file/line instead of patching locally.
- DO NOT convert unknown label `-1` to negative `0`.
- DO NOT change the task to 9-way softmax. This is multi-label classification using independent sigmoid/BCE outputs.
- DO NOT add RFB, SSD extras, bbox heads, detection heads, or segmentation heads.
- DO NOT start RVM training.

## Repository

```bash
export REPO_URL="https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git"
export BRANCH="agent/nas-ultraface-slim-9label-v1"
export PROJECT_ROOT="/data/pub1/z00919662/segmentation/nas-ultraface-slim-9label-v1"
```

Clone/pull the prepared branch. If the repo already exists, require a clean working tree and fast-forward only.

```bash
if [[ ! -d "${PROJECT_ROOT}/.git" ]]; then
  git clone --branch "${BRANCH}" --single-branch "${REPO_URL}" "${PROJECT_ROOT}"
else
  cd "${PROJECT_ROOT}"
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "ERROR: repository has local modifications; do not overwrite them"
    git status --short
    exit 2
  fi
  git fetch origin "${BRANCH}"
  git checkout "${BRANCH}"
  git merge --ff-only "origin/${BRANCH}"
fi

cd "${PROJECT_ROOT}"
git branch --show-current
git rev-parse HEAD
```

Expected branch:

```text
agent/nas-ultraface-slim-9label-v1
```

## Existing dataset paths (MUST use these exact roots)

```bash
export COCO_ROOT="/data/pub1/z00919662/segmentation/datasets/coco"
export PLACES_ROOT="/data/pub1/z00919662/segmentation/datasets/places365"
export SEG_ROOT="/data/pub1/z00919662/segmentation/datasets/COCO_ADE_13cls_16x9_640x360"

export LABEL_ROOT="/data/pub1/z00919662/segmentation/datasets/NAS_9label_multilabel_manifests"
export OUTPUT_ROOT="/data/pub1/z00919662/segmentation/nas_scene_tagging/ultraface_slim_9label_v1"
```

These datasets already exist. Do not search for replacements unless a path is genuinely missing. Do not download anything.

## 1. Audit existing dataset structure only

Print directory metadata and a small number of example entries; do not enumerate millions of files unnecessarily.

```bash
for p in "${COCO_ROOT}" "${PLACES_ROOT}" "${SEG_ROOT}"; do
  if [[ ! -d "${p}" ]]; then
    echo "MISSING_DATASET_ROOT=${p}"
    exit 2
  fi
  echo "FOUND_DATASET_ROOT=${p}"
  du -sh "${p}" || true
done

find "${COCO_ROOT}" -maxdepth 3 -type f \( -name 'instances_train2017.json' -o -name 'instances_val2017.json' \) -print
find "${COCO_ROOT}" -maxdepth 2 -type d \( -name 'train2017' -o -name 'val2017' \) -print

find "${PLACES_ROOT}" -maxdepth 3 -type f \( -name 'categories_places365.txt' -o -name 'IO_places365.txt' -o -name 'places365_val.txt' \) -print
find "${PLACES_ROOT}" -maxdepth 3 -type d \( -name 'data_256' -o -name 'train' -o -name 'val_256' -o -name 'val' \) -print

for split in train val; do
  test -d "${SEG_ROOT}/images/${split}"
  test -d "${SEG_ROOT}/annotations/${split}"
done
```

If one dataset has a different but obvious internal layout, first test whether `prepare_dataset.py` already resolves it. Do not modify source folders. If code cannot resolve the layout, stop and report exact existing paths plus error; do not patch locally.

## 2. Environment and CPU limits

Reuse the existing CUDA PyTorch environment already used on this server. Do not replace torch/torchvision unless required.

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

which python
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
print('gpu_count:', torch.cuda.device_count())
assert torch.cuda.is_available()
PY
```

Install only missing lightweight Python dependencies from the prepared project requirements if needed. Do not download datasets as part of environment setup.

## 3. Static code checks and model information

```bash
cd "${PROJECT_ROOT}/nas_scene_multilabel"
python -m compileall -q .
bash -n scripts/run_train.sh
python tools/model_info.py --input-size 224 --base-channel 16
```

Expected model concept:

```text
UltraFace Mb_Tiny/slim convolution backbone
NO RFB
NO SSD extras
NO bbox/confidence detection heads
Adaptive Global Average Pooling
Linear(256 -> 9)
9 independent logits
```

Default target size is approximately 0.173M parameters and ~48.3M MACs at 224x224. Record actual values from `model_info.py`; do not hard-fail solely because profiler rounding differs slightly.

## 4. Build multi-label manifests ONLY

The label builder must read the three existing datasets and generate label files only.

Source responsibilities:

```text
Places365:
  indoor / outdoor
  landscape
  sports
  office

COCO2017:
  food
  animal

COCO_ADE_13cls_16x9_640x360 masks:
  building
  sky
  food (supplemental)
```

Unknown labels remain `-1` and are excluded from loss and metrics.

Do not infer unsupported negatives across datasets. Example: a Places365 image without an animal annotation is NOT automatically `animal=0`; it must stay `animal=-1`.

Create only the manifest directory:

```bash
rm -rf "${LABEL_ROOT}"
mkdir -p "${LABEL_ROOT}"

cd "${PROJECT_ROOT}/nas_scene_multilabel"
python -u prepare_dataset.py \
  --places-root "${PLACES_ROOT}" \
  --coco-root "${COCO_ROOT}" \
  --seg-root "${SEG_ROOT}" \
  --output-root "${LABEL_ROOT}" \
  --places-train-cap-per-class 500 \
  --seed 20260902 \
  --sky-min-area 0.01 \
  --building-min-area 0.02 \
  --food-min-area 0.01 \
  2>&1 | tee "${LABEL_ROOT}/prepare_dataset.log"
```

Expected generated files:

```text
${LABEL_ROOT}/train.jsonl
${LABEL_ROOT}/val.jsonl
${LABEL_ROOT}/test.jsonl
${LABEL_ROOT}/dataset_summary.json
```

No image file should be created under `${LABEL_ROOT}`.

Verify this explicitly:

```bash
find "${LABEL_ROOT}" -type f | sort
find "${LABEL_ROOT}" -type l -print

python - <<'PY'
import json, os
from pathlib import Path
root = Path('/data/pub1/z00919662/segmentation/datasets/NAS_9label_multilabel_manifests')
allowed = {'.jsonl', '.json', '.log'}
bad = [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() not in allowed]
print('unexpected_files:', bad[:20])
assert not bad, 'Label root must contain manifests/reports only; no images or copied masks'
for name in ['train.jsonl','val.jsonl','test.jsonl']:
    p = root/name
    assert p.exists()
    with p.open() as f:
        first = json.loads(next(f))
    print(name, first)
    assert os.path.isabs(first['image'])
    assert set(first['labels'].values()) <= {-1,0,1}
PY
```

## 5. Dataset coverage audit

Read:

```bash
cat "${LABEL_ROOT}/dataset_summary.json"
```

Report for train/val/test, for every class:

```text
positive count
negative count
unknown count
```

Every class must have at least one positive and one negative in all three splits. Prefer substantially more; if any class has zero positive or zero negative, stop and report the exact class/split and source availability.

Also report total images by source in each split.

Do NOT lower coverage requirements by editing code.

## 6. Inspect duplicate/source leakage risk

Because the manifests are assembled from multiple existing datasets, verify that exact absolute image paths do not appear across train/val/test.

```bash
python - <<'PY'
import json
from pathlib import Path
root=Path('/data/pub1/z00919662/segmentation/datasets/NAS_9label_multilabel_manifests')
sets={}
for s in ['train','val','test']:
    sets[s]={json.loads(x)['image'] for x in (root/f'{s}.jsonl').read_text().splitlines() if x.strip()}
for a,b in [('train','val'),('train','test'),('val','test')]:
    inter=sets[a]&sets[b]
    print(a,b,'overlap=',len(inter))
    assert not inter, f'exact image path leakage: {a}/{b}'
PY
```

## 7. Select one free GPU

```bash
nvidia-smi
```

Choose one genuinely free GPU only. Example:

```bash
export CUDA_VISIBLE_DEVICES=0
```

Do not kill other jobs. Training is single-GPU.

## 8. Smoke training

First run only 1 epoch. Use workers <= 4 and CPU thread limits above.

```bash
export TRAIN_MANIFEST="${LABEL_ROOT}/train.jsonl"
export VAL_MANIFEST="${LABEL_ROOT}/val.jsonl"
export TEST_MANIFEST="${LABEL_ROOT}/test.jsonl"
export SMOKE_OUT="${OUTPUT_ROOT}/smoke"
mkdir -p "${SMOKE_OUT}"

cd "${PROJECT_ROOT}/nas_scene_multilabel"
python -u train.py \
  --train-manifest "${TRAIN_MANIFEST}" \
  --val-manifest "${VAL_MANIFEST}" \
  --test-manifest "${TEST_MANIFEST}" \
  --output-dir "${SMOKE_OUT}" \
  --input-size 224 \
  --base-channel 16 \
  --epochs 1 \
  --batch-size 256 \
  --workers 4 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --amp \
  2>&1 | tee "${SMOKE_OUT}/train.log"
```

If batch size 256 OOMs, reducing only the runtime batch size to 128/64/32 is allowed. Do not change input resolution or model width for the first baseline.

Smoke must demonstrate:

```text
forward/backward works
masked BCE works
validation runs
checkpoints are written
per-class metrics can be computed
```

If there is a code bug, stop; do not patch locally.

## 9. Full training: 60 epochs

If smoke passes, remove/recreate only the final output folder, not datasets/manifests.

```bash
export FINAL_OUT="${OUTPUT_ROOT}/train_224_bc16"
mkdir -p "${FINAL_OUT}"

cd "${PROJECT_ROOT}/nas_scene_multilabel"
python -u train.py \
  --train-manifest "${TRAIN_MANIFEST}" \
  --val-manifest "${VAL_MANIFEST}" \
  --test-manifest "${TEST_MANIFEST}" \
  --output-dir "${FINAL_OUT}" \
  --input-size 224 \
  --base-channel 16 \
  --epochs 60 \
  --batch-size 256 \
  --workers 4 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --warmup-epochs 2 \
  --amp \
  2>&1 | tee "${FINAL_OUT}/train.log"
```

Use the batch size proven by smoke if 256 did not fit.

If interrupted and `last.pth` exists, resume from it using the code's prepared resume option. Do not restart from epoch 0 unless no valid checkpoint exists.

## 10. Threshold calibration and test metrics

The prepared code calibrates one threshold per class on validation data and evaluates those fixed thresholds on the independent test split.

Final report must include, for each of the nine classes:

```text
Precision
Recall
F1
Accuracy
Balanced Accuracy
Average Precision
calibrated threshold
```

Also report:

```text
macro-F1
macro-balanced-accuracy
macro-AP
micro-F1 over known label/image pairs
```

Do not report overall accuracy alone as the primary metric.

## 11. ONNX export

Export the best deploy checkpoint at 224x224.

```bash
cd "${PROJECT_ROOT}/nas_scene_multilabel"
python -u export_onnx.py \
  --checkpoint "${FINAL_OUT}/best_deploy.pth" \
  --output "${FINAL_OUT}/ultraface_slim_9label_224.onnx" \
  --input-size 224 \
  --base-channel 16
```

Verify the ONNX file exists and report its size.

## 12. Final report format

Return:

```text
STATUS: PASS / FAIL

GITHUB_BRANCH:
GITHUB_COMMIT:

COCO_ROOT: /data/pub1/z00919662/segmentation/datasets/coco
PLACES_ROOT: /data/pub1/z00919662/segmentation/datasets/places365
SEG_ROOT: /data/pub1/z00919662/segmentation/datasets/COCO_ADE_13cls_16x9_640x360
LABEL_ROOT: /data/pub1/z00919662/segmentation/datasets/NAS_9label_multilabel_manifests
OUTPUT_ROOT:

DOWNLOADS_PERFORMED: NO
SOURCE_DATA_MODIFIED: NO
SOURCE_IMAGES_COPIED: NO

DATASET_COUNTS:
train total/by-source
val total/by-source
test total/by-source

PER_CLASS_LABEL_COUNTS:
label | train pos/neg/unknown | val pos/neg/unknown | test pos/neg/unknown

MODEL:
input_size:
base_channel:
parameters:
MACs:

TRAINING:
epochs:
batch_size:
workers:
best_epoch:

PER_CLASS_TEST_METRICS:
label | threshold | precision | recall | F1 | accuracy | balanced_accuracy | AP

MACRO_F1:
MACRO_BALANCED_ACCURACY:
MACRO_AP:
MICRO_F1_KNOWN_PAIRS:

BEST_CHECKPOINT:
LAST_CHECKPOINT:
ONNX_PATH:
ONNX_SIZE:

WARNINGS:
HUMAN_ACTION_REQUIRED: YES / NO
```

If all steps complete:

```text
HUMAN_ACTION_REQUIRED: NO
```

If a real dataset-layout incompatibility or code bug blocks execution:

```text
HUMAN_ACTION_REQUIRED: YES
```

State the exact problem and do not locally change the prepared source code.
