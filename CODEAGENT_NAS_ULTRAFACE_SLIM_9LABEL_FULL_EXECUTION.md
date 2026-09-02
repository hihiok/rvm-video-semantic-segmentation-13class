# NAS 9-label UltraFace-slim multi-label scene classification — FULL EXECUTION RUNBOOK

This file is self-contained for a brand-new CodeAgent chat. Do not assume any prior conversation context.

## 0. Objective

Train a very small multi-label scene classifier for NAS photo tagging using an UltraFace Mb_Tiny/slim-style convolutional backbone with all face-detection-specific modules removed.

Required labels, in this exact order:

1. `indoor` / 室内
2. `outdoor` / 户外
3. `landscape` / 风景
4. `sports` / 运动
5. `food` / 美食
6. `animal` / 动物
7. `building` / 建筑
8. `sky` / 蓝天（产品名叫“蓝天”，但语义定义是“明显可见天空”，不要求蓝色）
9. `office` / 办公

This is **multi-label classification**, not 9-way single-label classification. One image may have multiple positive labels simultaneously.

Label encoding:

- `1` = positive
- `0` = negative
- `-1` = unknown / not supervised by this source

`-1` MUST be masked out of loss and metrics. Never convert unknown labels to negative.

## 1. Hard constraints

- DO NOT download any dataset.
- DO NOT run `nas_scene_multilabel/scripts/download_datasets.sh`.
- DO NOT copy, move, rename, delete, resize, or modify the source datasets.
- DO NOT create a duplicate image dataset.
- Only create label manifests and training outputs.
- DO NOT modify Python/model/config/training code locally. If a real code bug is found, stop and report traceback + file + line number. The user will update GitHub centrally.
- DO NOT add RFB, SSD extras, detection heads, bbox heads, segmentation heads, or recurrent modules.
- DO NOT change the model input size or width for the first baseline.
- DO NOT switch to softmax. Outputs are 9 independent logits trained with masked BCEWithLogitsLoss.
- DO NOT run RVM training.
- Use one GPU only.
- Keep CPU use modest: DataLoader workers <= 4 and CPU math threads = 4.

## 2. GitHub repository

Repository:

```text
https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git
```

Branch to use:

```text
agent/nas-ultraface-slim-9label-v1-fullrun
```

Project checkout path:

```bash
export REPO_URL="https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git"
export BRANCH="agent/nas-ultraface-slim-9label-v1-fullrun"
export PROJECT_ROOT="/data/pub1/z00919662/segmentation/nas-ultraface-slim-9label-v1-fullrun"
```

If Git access requires the existing corporate proxy, use the proxy configuration already present on the server/shell. Do not print proxy credentials or commit them anywhere. Git SSL verification may remain disabled if required by the internal HTTPS inspection environment.

Clone or update the prepared branch:

```bash
set -euo pipefail

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
echo "BRANCH=$(git branch --show-current)"
echo "COMMIT=$(git rev-parse HEAD)"
```

Verify the prepared files exist:

```bash
test -f nas_scene_multilabel/config.py
test -f nas_scene_multilabel/model.py
test -f nas_scene_multilabel/prepare_dataset.py
test -f nas_scene_multilabel/train.py
test -f nas_scene_multilabel/export_onnx.py
test -f nas_scene_multilabel/tools/model_info.py
```

## 3. Existing datasets — use exactly these roots

```bash
export COCO_ROOT="/data/pub1/z00919662/segmentation/datasets/coco"
export PLACES_ROOT="/data/pub1/z00919662/segmentation/datasets/places365"
export SEG_ROOT="/data/pub1/z00919662/segmentation/datasets/COCO_ADE_13cls_16x9_640x360"
```

Generated labels only:

```bash
export LABEL_ROOT="/data/pub1/z00919662/segmentation/datasets/NAS_9label_multilabel_manifests"
```

Training outputs:

```bash
export OUTPUT_ROOT="/data/pub1/z00919662/segmentation/nas_scene_tagging/ultraface_slim_9label_v1"
```

Do not search for alternative datasets unless one of these exact roots is missing.

## 4. Dataset supervision policy

Use the three datasets as partial-label sources.

### Places365

Provides supervision for:

- `indoor`
- `outdoor`
- `landscape`
- `sports`
- `office`

Use the official Places365 indoor/outdoor taxonomy when available. Curated scene-category mappings in `config.py` define landscape/sports/office positives and negatives.

Do not infer `food/animal/building/sky` from Places365. Those labels remain `-1` for Places images unless the prepared code explicitly assigns them from a reliable source.

### COCO 2017

Provides supervision for:

- `food`
- `animal`

The prepared mapping in `config.py` defines COCO food and animal categories.

Do not infer indoor/outdoor/landscape/building/sky/office from COCO object labels.

### Existing COCO+ADE 13-class semantic dataset

Provides image-level supervision converted from segmentation masks for:

- `building`
- `sky`
- `food` as supplemental supervision

Mask-to-image-level rules are already implemented in `prepare_dataset.py`:

- sky positive if sky area >= 1%
- building positive if building area >= 2%
- food positive if food area >= 1%
- zero pixels => negative
- small non-zero area below threshold => unknown (`-1`)

Do not change these thresholds in the first baseline.

## 5. Audit the existing dataset layouts

Do not enumerate every image unnecessarily. Check roots and key files/directories only.

```bash
for p in "${COCO_ROOT}" "${PLACES_ROOT}" "${SEG_ROOT}"; do
  if [[ ! -d "${p}" ]]; then
    echo "MISSING_DATASET_ROOT=${p}"
    exit 2
  fi
  echo "FOUND_DATASET_ROOT=${p}"
  du -sh "${p}" || true
done
```

COCO checks:

```bash
find "${COCO_ROOT}" -maxdepth 4 -type f \
  \( -name 'instances_train2017.json' -o -name 'instances_val2017.json' \) -print
find "${COCO_ROOT}" -maxdepth 3 -type d \
  \( -name 'train2017' -o -name 'val2017' \) -print
```

Places365 checks:

```bash
find "${PLACES_ROOT}" -maxdepth 4 -type f \
  \( -name 'categories_places365.txt' -o -name 'IO_places365.txt' -o -name 'places365_val.txt' -o -name 'val.txt' \) -print
find "${PLACES_ROOT}" -maxdepth 4 -type d \
  \( -name 'data_256' -o -name 'train' -o -name 'val_256' -o -name 'val' \) -print
```

13-class segmentation checks:

```bash
for split in train val; do
  test -d "${SEG_ROOT}/images/${split}"
  test -d "${SEG_ROOT}/annotations/${split}"
done
```

If the actual layout differs, first run the prepared label builder and see whether its resolver already supports it. If it fails because of layout incompatibility, stop and report the actual directory structure plus traceback. Do not patch locally.

## 6. Python/CUDA environment

Reuse a working CUDA PyTorch environment on this server. Do not replace a working torch/torchvision installation.

Limit CPU usage:

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
```

Check environment:

```bash
which python
python - <<'PY'
import sys, torch
print('python:', sys.version)
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
print('gpu_count:', torch.cuda.device_count())
assert torch.cuda.is_available(), 'CUDA PyTorch environment required'
PY
```

Install only missing lightweight dependencies if needed:

```bash
cd "${PROJECT_ROOT}/nas_scene_multilabel"
python -m pip install -r requirements.txt --upgrade-strategy only-if-needed
```

Do not install or download any pretrained model. This first baseline trains the prepared small classifier directly.

## 7. Static checks and model-size check

```bash
cd "${PROJECT_ROOT}/nas_scene_multilabel"
python -m compileall -q .
python tools/model_info.py --input-size 224 --base-channel 16
```

Architecture must remain:

```text
RGB 224x224
  -> UltraFace Mb_Tiny/slim convolution backbone
  -> no RFB
  -> no SSD extras
  -> no bbox/confidence heads
  -> AdaptiveAvgPool2d(1)
  -> Dropout(0.1 during training)
  -> Linear(256, 9)
  -> 9 independent logits
```

At `base_channel=16`, expected scale is approximately:

- parameters: ~0.173M
- MACs @224x224: ~48M

Record the actual values printed by `model_info.py`.

## 8. Generate multi-label manifests only

Delete/recreate only the generated label manifest directory. Never touch source data.

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

Expected output files:

```text
${LABEL_ROOT}/train.jsonl
${LABEL_ROOT}/val.jsonl
${LABEL_ROOT}/test.jsonl
${LABEL_ROOT}/dataset_summary.json
${LABEL_ROOT}/prepare_dataset.log
```

No source image should be copied into `${LABEL_ROOT}`.

Verify:

```bash
find "${LABEL_ROOT}" -maxdepth 2 -type f -print | sort

python - <<'PY'
import json, os
from pathlib import Path
root = Path('/data/pub1/z00919662/segmentation/datasets/NAS_9label_multilabel_manifests')
allowed = {'.jsonl', '.json', '.log'}
bad = [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() not in allowed]
print('unexpected_files:', bad[:20])
assert not bad, 'LABEL_ROOT must contain labels/reports only'
for name in ['train.jsonl', 'val.jsonl', 'test.jsonl']:
    p = root / name
    assert p.exists() and p.stat().st_size > 0
    with p.open('r', encoding='utf-8') as f:
        first = json.loads(next(f))
    assert os.path.isabs(first['image'])
    assert set(first['labels'].keys()) == {
        'indoor','outdoor','landscape','sports','food','animal','building','sky','office'
    }
    assert set(first['labels'].values()) <= {-1,0,1}
    print(name, first)
PY
```

## 9. Mandatory dataset audit

Print the prepared summary:

```bash
cat "${LABEL_ROOT}/dataset_summary.json"
```

For every class and every split, report:

- positives
- negatives
- unknowns

Hard requirement for baseline execution:

- train: each label must have at least one positive and one negative
- val: each label must have at least one positive and one negative
- test: each label must have at least one positive and one negative

If any label lacks positive or negative supervision in any split, stop. Do not change mappings or thresholds locally.

Also report number of images per source (`places365`, `coco2017`, `seg13`) in train/val/test.

### Exact absolute-path leakage check

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path('/data/pub1/z00919662/segmentation/datasets/NAS_9label_multilabel_manifests')
sets = {}
for split in ['train','val','test']:
    sets[split] = {
        json.loads(line)['image']
        for line in (root / f'{split}.jsonl').read_text(encoding='utf-8').splitlines()
        if line.strip()
    }
for a,b in [('train','val'),('train','test'),('val','test')]:
    overlap = sets[a] & sets[b]
    print(a, b, 'exact_path_overlap=', len(overlap))
    assert not overlap, f'exact image path leakage: {a}/{b}'
PY
```

### Additional COCO/SEG duplication sanity check

Because the 13-class static dataset may contain processed derivatives of COCO images, inspect potential cross-split duplicate basenames/IDs across sources. Do not fail simply because train contains the same underlying image twice within the same split, but **do fail and report** if the same identifiable COCO image appears in different train/val/test splits through different source paths.

Use manifest `source`, `image`, and filename/basename information for this audit. Do not modify code or manifests to hide leakage.

## 10. Select one free GPU

```bash
nvidia-smi
```

Choose one genuinely free GPU. Example only:

```bash
export CUDA_VISIBLE_DEVICES=0
```

Do not kill or interfere with other jobs.

Confirm:

```bash
python - <<'PY'
import torch
print(torch.cuda.get_device_name(0))
print('allocated_MB=', torch.cuda.memory_allocated(0)/1024/1024)
PY
```

## 11. Smoke training — 1 epoch

Use the actual CLI implemented by `train.py`: it takes `--data-root`, not separate manifest arguments. AMP is already enabled internally on CUDA; there is no `--amp` CLI flag.

```bash
export SMOKE_OUT="${OUTPUT_ROOT}/smoke_224_bc16"
rm -rf "${SMOKE_OUT}"
mkdir -p "${SMOKE_OUT}"

cd "${PROJECT_ROOT}/nas_scene_multilabel"
python -u train.py \
  --data-root "${LABEL_ROOT}" \
  --output-dir "${SMOKE_OUT}" \
  --input-size 224 \
  --base-channel 16 \
  --dropout 0.1 \
  --epochs 1 \
  --batch-size 256 \
  --workers 4 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --warmup-epochs 0 \
  --seed 20260902 \
  --cpu-threads 4 \
  --print-every 100 \
  2>&1 | tee "${SMOKE_OUT}/train.log"
```

If batch size 256 causes CUDA OOM, retry only by reducing batch size in this order:

```text
128 -> 64 -> 32
```

Do not change input size or base channel for this baseline.

Smoke PASS criteria:

- dataset loads successfully
- forward/backward succeeds
- masked BCE runs with unknown labels
- validation runs
- `last.pth` exists
- `best_macro_f1.pth` exists
- final val calibration/test stage completes without crash

If smoke fails because of a code bug, stop and return traceback; do not patch locally.

## 12. Formal training — 60 epochs

Use the batch size proven by smoke.

```bash
export FINAL_OUT="${OUTPUT_ROOT}/train_224_bc16"
mkdir -p "${FINAL_OUT}"

cd "${PROJECT_ROOT}/nas_scene_multilabel"
python -u train.py \
  --data-root "${LABEL_ROOT}" \
  --output-dir "${FINAL_OUT}" \
  --input-size 224 \
  --base-channel 16 \
  --dropout 0.1 \
  --epochs 60 \
  --batch-size 256 \
  --workers 4 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --warmup-epochs 2 \
  --seed 20260902 \
  --cpu-threads 4 \
  --print-every 100 \
  2>&1 | tee "${FINAL_OUT}/train.log"
```

Replace `--batch-size 256` with the smoke-proven smaller value only if needed.

Training behavior already implemented in `train.py`:

- optimizer: AdamW
- LR: linear warmup + cosine decay
- mixed precision: CUDA FP16 autocast + GradScaler
- loss: masked BCEWithLogitsLoss
- positive-class reweighting: `neg/pos`, clipped to `[0.5, 8.0]`
- unknown labels (`-1`) are masked out
- best checkpoint selected by validation macro-F1 at threshold 0.5
- after training, thresholds are calibrated per class on validation data
- calibrated thresholds are then evaluated on independent test data

## 13. Resume after interruption/reboot

If formal training is interrupted and `${FINAL_OUT}/last.pth` exists, resume rather than restarting.

First verify checkpoint exists:

```bash
test -f "${FINAL_OUT}/last.pth"
```

Resume command:

```bash
cd "${PROJECT_ROOT}/nas_scene_multilabel"
python -u train.py \
  --data-root "${LABEL_ROOT}" \
  --output-dir "${FINAL_OUT}" \
  --input-size 224 \
  --base-channel 16 \
  --dropout 0.1 \
  --epochs 60 \
  --batch-size 256 \
  --workers 4 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --warmup-epochs 2 \
  --seed 20260902 \
  --cpu-threads 4 \
  --print-every 100 \
  --resume "${FINAL_OUT}/last.pth" \
  2>&1 | tee -a "${FINAL_OUT}/train.log"
```

Again use the actual batch size from the original run.

Important: resume restores model and optimizer. Do not restart from epoch 0 when a valid `last.pth` exists.

## 14. Expected training outputs

Formal training should produce at least:

```text
${FINAL_OUT}/metrics.jsonl
${FINAL_OUT}/last.pth
${FINAL_OUT}/best_macro_f1.pth
${FINAL_OUT}/best_val_per_class_0p5.csv
${FINAL_OUT}/thresholds.json
${FINAL_OUT}/test_per_class_calibrated.csv
${FINAL_OUT}/test_summary.json
${FINAL_OUT}/best_deploy.pth
${FINAL_OUT}/train.log
```

Inspect:

```bash
cat "${FINAL_OUT}/test_summary.json"
cat "${FINAL_OUT}/thresholds.json"
column -s, -t < "${FINAL_OUT}/test_per_class_calibrated.csv" | head -20 || cat "${FINAL_OUT}/test_per_class_calibrated.csv"
```

For each class report:

- calibrated threshold
- known GT count
- positive GT count
- negative GT count
- Precision
- Recall
- F1
- Accuracy
- Balanced Accuracy
- Average Precision

Primary aggregate metrics:

- macro-F1
- macro-balanced-accuracy
- macro-AP

Do not use ordinary overall accuracy alone to judge success.

## 15. Export ONNX

The actual `export_onnx.py` CLI only requires checkpoint/output/opset. Input size and base channel are read from the checkpoint.

```bash
cd "${PROJECT_ROOT}/nas_scene_multilabel"
python -u export_onnx.py \
  --checkpoint "${FINAL_OUT}/best_deploy.pth" \
  --output "${FINAL_OUT}/ultraface_slim_9label_224.onnx" \
  --opset 13
```

Expected files:

```text
${FINAL_OUT}/ultraface_slim_9label_224.onnx
${FINAL_OUT}/ultraface_slim_9label_224.json
```

Verify:

```bash
ls -lh "${FINAL_OUT}/ultraface_slim_9label_224.onnx" "${FINAL_OUT}/ultraface_slim_9label_224.json"
python - <<'PY'
import onnx
p='/data/pub1/z00919662/segmentation/nas_scene_tagging/ultraface_slim_9label_v1/train_224_bc16/ultraface_slim_9label_224.onnx'
m=onnx.load(p)
onnx.checker.check_model(m)
print('ONNX_CHECK=PASS')
print('nodes=', len(m.graph.node))
print('ops=', sorted(set(n.op_type for n in m.graph.node)))
PY
```

Do not claim V516 100fps from GPU training speed or MAC count. V516 latency must be measured separately after chip conversion.

## 16. Final report required from CodeAgent

Return a concise but complete report in exactly this structure:

```text
STATUS: PASS / FAIL

GITHUB_BRANCH:
GITHUB_COMMIT:

PYTHON_VERSION:
TORCH_VERSION:
GPU:
CUDA_VISIBLE_DEVICES:

DOWNLOADS_PERFORMED: NO
SOURCE_DATA_MODIFIED: NO
SOURCE_IMAGES_COPIED: NO

COCO_ROOT: /data/pub1/z00919662/segmentation/datasets/coco
PLACES_ROOT: /data/pub1/z00919662/segmentation/datasets/places365
SEG_ROOT: /data/pub1/z00919662/segmentation/datasets/COCO_ADE_13cls_16x9_640x360
LABEL_ROOT: /data/pub1/z00919662/segmentation/datasets/NAS_9label_multilabel_manifests
OUTPUT_ROOT: /data/pub1/z00919662/segmentation/nas_scene_tagging/ultraface_slim_9label_v1

DATASET_COUNTS:
train total / by source
val total / by source
test total / by source

PER_CLASS_LABEL_COUNTS:
label | train pos/neg/unknown | val pos/neg/unknown | test pos/neg/unknown

LEAKAGE_AUDIT:
exact_path_overlap train/val:
exact_path_overlap train/test:
exact_path_overlap val/test:
COCO-vs-SEG cross-split duplicate risk:

MODEL:
input_size: 224
base_channel: 16
parameters:
MACs:
approx_FLOPs:

SMOKE:
status:
batch_size_used:

TRAINING:
epochs_target: 60
epochs_completed:
batch_size:
workers: 4
best_epoch:
best_val_macro_f1_0p5:

PER_CLASS_TEST_METRICS:
label | threshold | precision | recall | F1 | accuracy | balanced_accuracy | AP | known(pos/neg)

MACRO_F1:
MACRO_BALANCED_ACCURACY:
MACRO_AP:

BEST_CHECKPOINT:
LAST_CHECKPOINT:
THRESHOLDS_JSON:
TEST_METRICS_CSV:
ONNX_PATH:
ONNX_SIZE:
ONNX_CHECK: PASS / FAIL

WARNINGS:
HUMAN_ACTION_REQUIRED: YES / NO
```

## 17. When to stop and ask for human action

Return `HUMAN_ACTION_REQUIRED: YES` only if one of these occurs:

1. one of the three exact dataset roots is missing;
2. the existing dataset layout is incompatible with prepared code;
3. dataset supervision coverage is incomplete for a class/split;
4. cross-split source leakage is discovered and requires code/data-split changes;
5. prepared GitHub code throws a genuine bug;
6. no usable CUDA environment/GPU exists;
7. ONNX export/check fails due a code or environment issue that cannot be resolved without modifying prepared source.

When blocked, give exact paths, traceback, file/line, and observed directory layout. Do not locally edit the prepared source code.

If the entire pipeline completes successfully:

```text
HUMAN_ACTION_REQUIRED: NO
```
