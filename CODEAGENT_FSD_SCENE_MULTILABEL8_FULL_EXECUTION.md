# CODEAGENT — FSD 8-label multi-label scene classification — FULL EXECUTION

This document is self-contained for a new CodeAgent chat.

## 0. Goal

Train the existing FSD/UltraFace scene-classification network as an **8-label multi-label classifier**.

Fixed label order:

```text
0 night           夜景
1 indoor          室内
2 rain_snow       雨/雪
3 office          办公场景
4 outdoor         户外
5 landscape       风景
6 sports          运动
7 objective_image 客观图（电脑合成 pattern、解析度卡、test chart 等）
```

This is NOT an 8-way softmax task. One image may contain multiple labels, e.g.:

```text
night=1 + outdoor=1 + landscape=1
office=1 + indoor=1
```

Manifest label values:

```text
1  positive
0  confirmed negative
-1 unknown / not supervised by this source
```

`-1` MUST be ignored by loss and metrics.

## 1. Model requirement — MUST use existing FSD implementation

The user supplied the existing FSD scene-classification baseline. Keep the existing model factory and preprocessing path:

```python
create_Mb_Tiny_RFB_fd_3_scene_noRFB(2, 8)
YUV444TrainAugmentation_scene
YUV444TestTransform_scene
```

The reference FSD launcher used:

```text
input_size=240
batch_size=24
SGD lr=1e-2
weight_decay=1e-4
milestones=95,150
num_epochs=200
workers=4
```

The prepared code follows those defaults.

Do NOT replace the FSD model with the standalone `UltraFaceSlimMultiLabel` implementation from an earlier experiment.

Do NOT edit:

```text
vision/ssd/mb_tiny_RFB_fd_3.py
vision/ssd/data_preprocessing.py
existing train-compressed-sceneclassification-noRFB.py
existing FSD shell scripts
```

The new trainer imports the existing FSD modules at runtime.

## 2. GitHub staging repository

Repository:

```text
https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git
```

Branch:

```text
agent/fsd-scene-multilabel8-v1
```

Suggested checkout:

```bash
export REPO_URL="https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git"
export BRANCH="agent/fsd-scene-multilabel8-v1"
export PROJECT_ROOT="/data/pub1/z00919662/segmentation/fsd-scene-multilabel8-v1"
```

### Mandatory repository update rule

If `${PROJECT_ROOT}` does not exist:

```bash
git clone --branch "${BRANCH}" --single-branch "${REPO_URL}" "${PROJECT_ROOT}"
```

If it already exists:

```bash
cd "${PROJECT_ROOT}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "HUMAN_ACTION_REQUIRED: YES"
  echo "Reason: local repository has uncommitted changes"
  git status --short
  exit 2
fi

git fetch origin "${BRANCH}"
git checkout "${BRANCH}"
git merge --ff-only "origin/${BRANCH}"
```

Then verify:

```bash
cd "${PROJECT_ROOT}"
LOCAL_HEAD="$(git rev-parse HEAD)"
REMOTE_HEAD="$(git rev-parse origin/${BRANCH})"
CURRENT_BRANCH="$(git branch --show-current)"

echo "CURRENT_BRANCH=${CURRENT_BRANCH}"
echo "LOCAL_HEAD=${LOCAL_HEAD}"
echo "REMOTE_HEAD=${REMOTE_HEAD}"

[[ "${CURRENT_BRANCH}" == "${BRANCH}" ]]
[[ "${LOCAL_HEAD}" == "${REMOTE_HEAD}" ]]
[[ -z "$(git status --porcelain)" ]]
```

Do NOT:

```text
git stash
git reset --hard
commit local patches
force checkout over local changes
```

If clean fast-forward update cannot be completed, STOP.

## 3. Prepared GitHub files

Required:

```text
fsd_scene_multilabel8/config.py
fsd_scene_multilabel8/prepare_manifest.py
fsd_scene_multilabel8/audit_manifest.py
fsd_scene_multilabel8/make_smoke_manifest.py
fsd_scene_multilabel8/train_fsd_scene_multilabel8.py
fsd_scene_multilabel8/run_smoke.sh
fsd_scene_multilabel8/run_train.sh
fsd_scene_multilabel8/README.md
nas_scene_multilabel/metadata/places365_io_map.tsv
```

CodeAgent MUST NOT rewrite or patch these locally.

If prepared code has a genuine bug, STOP and report traceback + exact file/line.

## 4. Existing source datasets — READ ONLY

Use these exact paths:

```bash
export PLACES_ROOT="/data/pub1/z00919662/segmentation/datasets/places365"
export COCO_ROOT="/data/pub1/z00919662/segmentation/datasets/coco"
export SEG_ROOT="/data/pub1/z00919662/segmentation/datasets/COCO_ADE_13cls_16x9_640x360"
export TEN_SCENES_ROOT="/data/pub1/z00919662/dataset/10_scenes"
export COMPUTER_SYNTH_ROOT="/data/pub1/z00919662/dataset/10_scenes/train/Computer_synthesized"
```

Do NOT modify any existing:

```text
images
masks
annotations
folder names
existing label files
```

Do NOT download a new dataset.

Do NOT copy source images into a new dataset.

The only new dataset artifact is a set of JSONL label manifests referencing original absolute paths.

## 5. New manifest/output paths

```bash
export LABEL_ROOT="/data/pub1/z00919662/dataset/FSD_8scene_multilabel_manifests"
export SMOKE_LABEL_ROOT="/data/pub1/z00919662/dataset/FSD_8scene_multilabel_smoke_manifests"
export OUTPUT_ROOT="/data/pub1/z00919662/scene_multilabel/fsd_8label_v1"
```

`LABEL_ROOT` may contain only JSON/JSONL/log audit files. No source image copies.

## 6. Dataset-label policy

### 6.1 Places365

Use the actual local structure already discovered:

```text
places365/
└── versions/1/
    ├── train/<category-folder>/*.jpg
    └── val/<category-folder>/*.jpg
```

Folder names are mapped back to the official Places365 category taxonomy using the repository metadata.

Use Places365 for:

```text
indoor
outdoor
office
landscape
sports
conservative snow positives for rain_snow
```

All Places365 photographic images are valid:

```text
objective_image=0
```

Do not infer `night` from Places365.

### 6.2 10_scenes

This is the main source for product-specific classes not well represented by Places365.

Important exact folder:

```text
/data/pub1/z00919662/dataset/10_scenes/train/Computer_synthesized
```

Every image in this mapped folder must have:

```text
objective_image=1
night=0
indoor=0
rain_snow=0
office=0
outdoor=0
landscape=0
sports=0
```

Other 10_scenes categories may supplement:

```text
night
indoor
rain_snow
office
outdoor
landscape
sports
```

The prepared mapping is explicit/conservative. Unmapped folders MUST NOT automatically become negatives.

Generate and inspect:

```text
source_folder_audit.json
```

If a folder looks like one of the requested 8 categories but remains unmapped, STOP and return the complete folder list. Do not invent an alias locally.

### 6.3 COCO

COCO is used conservatively as real-photo negative supervision for:

```text
objective_image=0
```

Other 7 labels remain unknown for COCO-only records.

This avoids generating noisy scene labels from object annotations.

### 6.4 Existing COCO+ADE 13-class semantic data

Use only conservative image-level labels derived from existing masks:

```text
class 9 ice_or_snow -> rain_snow positive if area >= 1%
class 9 absent      -> rain_snow negative

plant/water/desert/ice_or_snow/mountain combined area >= 30%
                    -> landscape positive
```

No raw source mask is changed.

Important: semantic masks provide snow evidence, not generic rain evidence. Rain positives should come from `10_scenes` if available.

All these natural images are valid:

```text
objective_image=0
```

## 7. Initial source audit

Check exact roots:

```bash
for p in "${PLACES_ROOT}" "${COCO_ROOT}" "${SEG_ROOT}" "${TEN_SCENES_ROOT}" "${COMPUTER_SYNTH_ROOT}"; do
  if [[ ! -d "${p}" ]]; then
    echo "MISSING=${p}"
    echo "HUMAN_ACTION_REQUIRED: YES"
    exit 2
  fi
  echo "FOUND=${p}"
done
```

Print 10_scenes top-level structure without changing it:

```bash
find "${TEN_SCENES_ROOT}" -maxdepth 2 -type d -print | sort
```

Confirm Computer_synthesized contains images:

```bash
find "${COMPUTER_SYNTH_ROOT}" -maxdepth 2 -type f | head -20
```

Do not rename folders to match code.

## 8. Locate existing FSD repository

Preferred known location:

```text
/mnt/ssd1/z00919662/AI-face-detect/ultraface_3323_ref_param
```

Check:

```bash
PREFERRED_FSD="/mnt/ssd1/z00919662/AI-face-detect/ultraface_3323_ref_param"
```

A valid FSD root MUST contain all of:

```text
vision/ssd/mb_tiny_RFB_fd_3.py
vision/ssd/data_preprocessing.py
vision/ssd/config/fd_config.py
train-compressed-sceneclassification-noRFB.py
```

If preferred path is valid:

```bash
export FSD_ROOT="${PREFERRED_FSD}"
```

If not, search only likely user project roots:

```bash
find /mnt/ssd1/z00919662 /data/pub1/z00919662 \
  -maxdepth 7 -type f -name 'train-compressed-sceneclassification-noRFB.py' \
  -print 2>/dev/null
```

For every candidate, confirm `vision/ssd/mb_tiny_RFB_fd_3.py` exists.

If zero candidates exist: STOP.

If multiple independent FSD roots exist and there is no unambiguous current working repo: STOP and list them. Do not guess.

Do NOT modify the FSD repository source files.

## 9. Environment

Use the FSD project's existing working Python environment.

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

which python
python - <<'PY'
import sys, torch, cv2
print('python', sys.version)
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available())
print('cv2', cv2.__version__)
assert torch.cuda.is_available()
PY
```

Do not replace working PyTorch/CUDA packages.

## 10. Static checks

```bash
cd "${PROJECT_ROOT}/fsd_scene_multilabel8"
python -m compileall -q .
bash -n run_smoke.sh
bash -n run_train.sh
```

Then verify FSD imports without changing source:

```bash
python - <<'PY'
import sys
from pathlib import Path
fsd=Path("${FSD_ROOT}")
sys.path.insert(0,str(fsd))
from vision.ssd.config.fd_config import define_img_size
define_img_size(240)
from vision.ssd.mb_tiny_RFB_fd_3 import create_Mb_Tiny_RFB_fd_3_scene_noRFB
net=create_Mb_Tiny_RFB_fd_3_scene_noRFB(2,8)
print(type(net).__name__)
print('FSD_FACTORY_8LABEL=PASS')
PY
```

If factory instantiation fails, STOP. Do not edit FSD code.

## 11. Build new 8-label manifests

Remove/recreate only the generated manifest folder:

```bash
rm -rf "${LABEL_ROOT}"
mkdir -p "${LABEL_ROOT}"

cd "${PROJECT_ROOT}/fsd_scene_multilabel8"

python -u prepare_manifest.py \
  --places-root "${PLACES_ROOT}" \
  --coco-root "${COCO_ROOT}" \
  --seg-root "${SEG_ROOT}" \
  --ten-scenes-root "${TEN_SCENES_ROOT}" \
  --output-root "${LABEL_ROOT}" \
  --places-train-cap-per-class 500 \
  --coco-objective-neg-train-cap 5000 \
  --coco-objective-neg-eval-cap 1000 \
  --seed 20260903 \
  --snow-min-area 0.01 \
  --landscape-min-area 0.30 \
  2>&1 | tee "${LABEL_ROOT}/prepare.log"
```

Expected only:

```text
train.jsonl
val.jsonl
test.jsonl
dataset_summary.json
source_folder_audit.json
prepare.log
```

Verify no images were generated:

```bash
python - <<'PY'
from pathlib import Path
root=Path('/data/pub1/z00919662/dataset/FSD_8scene_multilabel_manifests')
allowed={'.jsonl','.json','.log'}
bad=[p for p in root.rglob('*') if p.is_file() and p.suffix.lower() not in allowed]
print('unexpected_files',bad[:20])
assert not bad
PY
```

## 12. Inspect source folder mappings

```bash
cat "${LABEL_ROOT}/source_folder_audit.json"
```

Specifically verify:

```text
Computer_synthesized -> objective_image
```

If Computer_synthesized is not mapped to `objective_image`, STOP.

Review `ten_scenes_unmapped_folders`.

If an unmapped folder clearly corresponds to one of the requested classes, STOP and return the folder list so ChatGPT can update GitHub mapping centrally.

Do NOT locally edit `config.py`.

## 13. Dataset audit and leakage rule

Run:

```bash
cd "${PROJECT_ROOT}/fsd_scene_multilabel8"
python -u audit_manifest.py --data-root "${LABEL_ROOT}" \
  2>&1 | tee "${LABEL_ROOT}/audit.log"
```

Hard coverage requirement implemented by prepared code:

```text
all 8 labels must have at least one positive and one negative in train/val/test
```

Additionally report low-data warnings:

```text
TRAIN positive < 100  -> WARNING
VAL positive < 20     -> WARNING
TEST positive < 20    -> WARNING
```

Low-count warning alone does NOT stop the run if coverage is non-zero.

### Leakage policy

HARD FAIL:

```text
same exact/identifiable underlying image in TRAIN and VAL
same exact/identifiable underlying image in TRAIN and TEST
```

WARNING ONLY:

```text
same underlying image in VAL and TEST
same underlying image twice inside one split through different sources
```

Val/test overlap must be disclosed in final metrics because validation is used for threshold calibration.

## 14. Choose one free GPU

```bash
nvidia-smi
```

Choose one genuinely free GPU. Do not kill other users/jobs.

Set:

```bash
export GPU=<free_gpu_index>
```

Training is single-GPU only.

## 15. Smoke test

The prepared smoke launcher first creates compact manifests preserving positive/negative coverage for all labels, then runs 1 epoch.

```bash
cd "${PROJECT_ROOT}/fsd_scene_multilabel8"

FSD_ROOT="${FSD_ROOT}" \
DATA_ROOT="${LABEL_ROOT}" \
SMOKE_DATA_ROOT="${SMOKE_LABEL_ROOT}" \
OUT="${OUTPUT_ROOT}/smoke" \
GPU="${GPU}" \
BATCH=24 \
bash run_smoke.sh
```

If batch 24 OOMs, retry only:

```text
BATCH=16
BATCH=8
```

Do not change input size or model factory.

Smoke PASS requires:

```text
FSD YUV444 transform successfully reads manifest images
network output shape is [B,8]
masked BCE is finite
backward works
validation works
per-class threshold calibration works
best_fsd_multilabel8.pth exists
thresholds.json exists
deployment_metadata.json exists
```

If the FSD transform API differs and prepared compatibility wrapper still fails, STOP and report both transform errors. Do not change preprocessing locally.

## 16. Formal 200-epoch training

After smoke PASS:

```bash
cd "${PROJECT_ROOT}/fsd_scene_multilabel8"

FSD_ROOT="${FSD_ROOT}" \
DATA_ROOT="${LABEL_ROOT}" \
OUT="${OUTPUT_ROOT}/train_240" \
GPU="${GPU}" \
BATCH=24 \
bash run_train.sh
```

Use the smaller batch proven by smoke if 24 did not fit.

Default model/training settings must remain:

```text
factory: create_Mb_Tiny_RFB_fd_3_scene_noRFB(2,8)
input_size: 240
loss: masked BCEWithLogitsLoss
optimizer: SGD
lr: 1e-2
weight_decay: 1e-4
momentum: 0.9
milestones: 95,150
epochs: 200
workers: 4
AMP: enabled
```

## 17. Resume rule

If formal training is interrupted:

```bash
ls -lh "${OUTPUT_ROOT}/train_240/last_train_state.pth"
```

If valid, resume instead of restarting:

```bash
cd "${PROJECT_ROOT}/fsd_scene_multilabel8"

FSD_ROOT="${FSD_ROOT}" \
DATA_ROOT="${LABEL_ROOT}" \
OUT="${OUTPUT_ROOT}/train_240" \
GPU="${GPU}" \
BATCH=24 \
RESUME="${OUTPUT_ROOT}/train_240/last_train_state.pth" \
bash run_train.sh
```

Use the actual batch size from the original run.

## 18. Final outputs

Formal training should produce:

```text
metrics.jsonl
last_train_state.pth
best_train_state.pth
best_val_per_class_0p5.csv
thresholds.json
test_per_class_calibrated.csv
test_summary.json
best_fsd_multilabel8.pth
deployment_metadata.json
train.log
```

`best_fsd_multilabel8.pth` is saved through the existing FSD model `save()` method and is intended to be loadable by the existing FSD `net.load()` flow.

## 19. FSD checkpoint load compatibility check

After training:

```bash
python - <<'PY'
import sys
from pathlib import Path
fsd=Path("${FSD_ROOT}")
sys.path.insert(0,str(fsd))
from vision.ssd.config.fd_config import define_img_size
define_img_size(240)
from vision.ssd.mb_tiny_RFB_fd_3 import create_Mb_Tiny_RFB_fd_3_scene_noRFB
net=create_Mb_Tiny_RFB_fd_3_scene_noRFB(2,8)
ckpt=Path('/data/pub1/z00919662/scene_multilabel/fsd_8label_v1/train_240/best_fsd_multilabel8.pth')
net.load(str(ckpt))
print('FSD_CHECKPOINT_LOAD=PASS')
PY
```

If this fails because of checkpoint format, STOP. Do not patch `net.load()`.

## 20. Metrics to report

For every class report:

```text
threshold
known samples
positive / negative samples
Precision
Recall
F1
Accuracy
Balanced Accuracy
AP
```

Aggregate:

```text
macro-F1
macro-balanced-accuracy
macro-AP
```

Do not use simple overall accuracy alone as the main success metric.

## 21. Deployment semantics

FSD model output is:

```text
8 raw logits
```

Runtime should conceptually use:

```text
sigmoid(logit[i]) >= threshold[i]
```

with thresholds from `thresholds.json`.

Do NOT apply an 8-way softmax.

## 22. Repository/code failure stop rule

CodeAgent MUST NOT locally patch prepared source code.

HARD STOP with `HUMAN_ACTION_REQUIRED: YES` if:

1. staging Git repo is dirty before update;
2. git fetch/checkout/ff-only fails;
3. local HEAD != remote branch HEAD after sync;
4. one required source dataset root is missing;
5. FSD repo cannot be located unambiguously;
6. FSD factory/import fails;
7. prepared Python/shell syntax check fails;
8. Places365 has unmapped category folders;
9. `Computer_synthesized` is not mapped to `objective_image`;
10. a clearly relevant 10_scenes class folder is unmapped and requires a mapping code change;
11. any label has zero positive or zero negative supervision in train/val/test;
12. TRAIN overlaps VAL or TEST at underlying-image level;
13. FSD transform compatibility wrapper fails;
14. FSD output is not `[B,8]`;
15. loss becomes NaN/Inf;
16. batch 24/16/8 all OOM;
17. formal training cannot resume from a valid last state after interruption;
18. FSD-compatible final checkpoint cannot be loaded by `net.load()`;
19. any other problem requires editing prepared `.py/.sh/config` files.

When stopped, return:

```text
HUMAN_ACTION_REQUIRED: YES
exact error
full traceback
file + line
observed paths/layout
command that failed
```

Do not improvise a code fix locally.

## 23. Warning-only conditions

Do NOT stop for:

```text
val/test duplicate images
same underlying image twice inside one split
some labels having relatively low but non-zero counts
batch 24 OOM if 16 or 8 works
```

Record them as warnings.

## 24. Final report format

```text
STATUS: PASS / FAIL
HUMAN_ACTION_REQUIRED: YES / NO

GITHUB_BRANCH:
GITHUB_COMMIT:
FSD_ROOT:
PYTHON:
TORCH:
GPU:

SOURCE_DATA_MODIFIED: NO
SOURCE_LABELS_MODIFIED: NO
SOURCE_IMAGES_COPIED: NO
DOWNLOADS_PERFORMED: NO

DATA_ROOTS:
Places365:
COCO:
COCO_ADE_13cls:
10_scenes:
Computer_synthesized:

MANIFEST_ROOT:
train_records:
val_records:
test_records:

10_SCENES_FOLDER_MAPPING:
...
UNMAPPED_10_SCENES_FOLDERS:
...

PER_CLASS_COUNTS:
label | train pos/neg/unknown | val pos/neg/unknown | test pos/neg/unknown
night
indoor
rain_snow
office
outdoor
landscape
sports
objective_image

LEAKAGE:
train_val_underlying_overlap:
train_test_underlying_overlap:
val_test_underlying_overlap_warning:

MODEL:
factory: create_Mb_Tiny_RFB_fd_3_scene_noRFB(2,8)
input_size_argument: 240
actual_fd_config_image_size:
output_shape:

SMOKE:
status:
batch:

TRAINING:
epochs_target: 200
epochs_completed:
batch:
best_epoch:
best_val_macro_f1_0p5:

TEST:
label | threshold | precision | recall | F1 | accuracy | balanced_accuracy | AP | known(pos/neg)

MACRO_F1:
MACRO_BALANCED_ACCURACY:
MACRO_AP:

BEST_FSD_CHECKPOINT:
FSD_CHECKPOINT_LOAD: PASS / FAIL
THRESHOLDS_JSON:
DEPLOYMENT_METADATA:

WARNINGS:
```

If all stages complete:

```text
HUMAN_ACTION_REQUIRED: NO
```
