# CODEAGENT — FSD 8-label multi-label scene classification — EXECUTE V2

This is the authoritative runbook. It supersedes earlier NAS 9-label / standalone UltraFace-slim instructions.

## 0. Task

Train the existing FSD/UltraFace scene model as an 8-label **multi-label** classifier.

Fixed output order:

```text
0 night            夜景
1 indoor           室内
2 rain_snow        雨/雪
3 office           办公场景
4 outdoor          户外
5 landscape        风景
6 sports           运动
7 objective_image  客观图（电脑合成 pattern、解析度卡、test chart 等）
```

One image may have multiple positives. Do NOT use softmax.

Manifest values:

```text
1  positive
0  confirmed negative
-1 unknown; ignored by loss/metrics
```

## 1. Must use the existing FSD network

Do NOT use the earlier standalone `UltraFaceSlimMultiLabel` model.

The prepared trainer imports the existing FSD implementation:

```python
create_Mb_Tiny_RFB_fd_3_scene_noRFB(2, 8)
YUV444TrainAugmentation_scene
YUV444TestTransform_scene
```

Keep the user's existing FSD source unchanged.

Reference behavior intentionally preserved from the supplied FSD shell/Python baseline:

```text
input_size = 240
batch_size = 24
optimizer = SGD
lr = 1e-2
weight_decay = 1e-4
milestones = 95,150
epochs = 200
workers = 4
```

The only conceptual training change is:

```text
single-label CrossEntropy
        ->
8 independent logits + masked BCEWithLogitsLoss
```

## 2. GitHub

```bash
export REPO_URL="https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git"
export BRANCH="agent/fsd-scene-multilabel8-v1"
export PROJECT_ROOT="/data/pub1/z00919662/segmentation/fsd-scene-multilabel8-v1"
```

### Mandatory repo sync rule

If repository does not exist:

```bash
git clone --branch "${BRANCH}" --single-branch "${REPO_URL}" "${PROJECT_ROOT}"
```

If it exists:

```bash
cd "${PROJECT_ROOT}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "HUMAN_ACTION_REQUIRED: YES"
  echo "Repository working tree is not clean"
  git status --short
  exit 2
fi

git fetch origin "${BRANCH}"
git checkout "${BRANCH}"
git merge --ff-only "origin/${BRANCH}"
```

Verify:

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

Forbidden:

```text
git stash
git reset --hard
force checkout
local source patch
local commit to repair prepared code
```

If fast-forward clean sync fails, STOP.

## 3. Prepared files

Verify:

```bash
cd "${PROJECT_ROOT}"
for f in \
  fsd_scene_multilabel8/config.py \
  fsd_scene_multilabel8/prepare_manifest.py \
  fsd_scene_multilabel8/finalize_manifest.py \
  fsd_scene_multilabel8/audit_manifest.py \
  fsd_scene_multilabel8/make_smoke_manifest.py \
  fsd_scene_multilabel8/train_fsd_scene_multilabel8.py \
  fsd_scene_multilabel8/run_smoke.sh \
  fsd_scene_multilabel8/run_train.sh \
  nas_scene_multilabel/metadata/places365_io_map.tsv; do
  test -f "$f" || { echo "MISSING_PREPARED_FILE=$f"; exit 2; }
done
```

Do NOT rewrite these files locally.

## 4. Existing datasets — read only

Use exactly:

```bash
export PLACES_ROOT="/data/pub1/z00919662/segmentation/datasets/places365"
export COCO_ROOT="/data/pub1/z00919662/segmentation/datasets/coco"
export SEG_ROOT="/data/pub1/z00919662/segmentation/datasets/COCO_ADE_13cls_16x9_640x360"
export TEN_SCENES_ROOT="/data/pub1/z00919662/dataset/10_scenes"
export COMPUTER_SYNTH_ROOT="/data/pub1/z00919662/dataset/10_scenes/train/Computer_synthesized"
```

Do NOT:

```text
download another dataset
modify source images
modify source masks
modify source annotations
modify existing label files
rename source folders
copy source images into a new dataset
```

New generated labels only:

```bash
export LABEL_ROOT="/data/pub1/z00919662/dataset/FSD_8scene_multilabel_manifests"
export SMOKE_LABEL_ROOT="/data/pub1/z00919662/dataset/FSD_8scene_multilabel_smoke_manifests"
export OUTPUT_ROOT="/data/pub1/z00919662/scene_multilabel/fsd_8label_v1"
```

## 5. Source label policy

### Places365

Actual local layout is supported directly:

```text
places365/versions/1/train/<category-folder>/...
places365/versions/1/val/<category-folder>/...
```

Use official Places IO taxonomy + explicit category mappings for:

```text
indoor
outdoor
office
landscape
sports
snow-like rain_snow positives
```

Do not infer night.

### 10_scenes

This is the main source for product-specific categories, especially:

```text
night
rain_snow
objective_image
```

Exact important source:

```text
/data/pub1/z00919662/dataset/10_scenes/train/Computer_synthesized
```

It MUST map to:

```text
objective_image = 1
all other 7 labels = 0
```

Other 10_scenes folders are used only through explicit aliases in prepared `config.py`.

Unmapped folders are NOT silently turned into negatives.

### COCO

Use sampled real photos only as:

```text
objective_image = 0
```

Do not manufacture the other 7 scene labels from COCO object annotations.

### Existing COCO+ADE 13-class masks

Conservative supervision only:

```text
ice_or_snow area >= 1% -> rain_snow = 1
ice_or_snow absent     -> rain_snow = 0

plant/water/desert/ice_or_snow/mountain combined area >= 30%
                       -> landscape = 1
```

This provides snow evidence, not generic rain evidence.

## 6. Important objective-image balancing rule

`prepare_manifest.py` initially records safe photographic negatives from several sources for auditability.

Immediately after generation, MUST run `finalize_manifest.py`.

It changes **only the new JSONL manifests**, never source datasets.

It removes `objective_image=0` supervision from the very large Places365/SEG13 record pools by changing those manifest fields to `-1`.

Final objective-image supervision therefore comes mainly from:

```text
positive: Computer_synthesized / explicitly mapped objective folders
negative: sampled COCO real photos + non-objective 10_scenes photos
```

This prevents hundreds of thousands of photographic negatives from swamping the objective-image positive class.

## 7. Source audit

```bash
for p in "${PLACES_ROOT}" "${COCO_ROOT}" "${SEG_ROOT}" "${TEN_SCENES_ROOT}" "${COMPUTER_SYNTH_ROOT}"; do
  if [[ ! -d "${p}" ]]; then
    echo "MISSING_DATASET_ROOT=${p}"
    echo "HUMAN_ACTION_REQUIRED: YES"
    exit 2
  fi
  echo "FOUND_DATASET_ROOT=${p}"
done

find "${TEN_SCENES_ROOT}" -maxdepth 2 -type d -print | sort
find "${COMPUTER_SYNTH_ROOT}" -maxdepth 2 -type f | head -20
```

Do not rename source directories to fit mappings.

## 8. Locate FSD repo

Preferred path:

```bash
PREFERRED_FSD="/mnt/ssd1/z00919662/AI-face-detect/ultraface_3323_ref_param"
```

A valid root contains:

```text
train-compressed-sceneclassification-noRFB.py
vision/ssd/mb_tiny_RFB_fd_3.py
vision/ssd/data_preprocessing.py
vision/ssd/config/fd_config.py
```

If preferred root is valid:

```bash
export FSD_ROOT="${PREFERRED_FSD}"
```

Otherwise search:

```bash
find /mnt/ssd1/z00919662 /data/pub1/z00919662 \
  -maxdepth 7 -type f -name 'train-compressed-sceneclassification-noRFB.py' \
  -print 2>/dev/null
```

If zero valid FSD roots: STOP.

If multiple independent valid FSD repos exist and no unambiguous active repo can be determined: STOP and list them. Do not guess.

Do not change anything inside `${FSD_ROOT}`.

## 9. Environment / CPU limits

Reuse the working FSD environment.

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

which python
python - <<'PY'
import sys, torch, cv2
print('python',sys.version)
print('torch',torch.__version__)
print('cuda',torch.cuda.is_available())
print('cv2',cv2.__version__)
assert torch.cuda.is_available()
PY
```

Do not replace a working torch/CUDA installation.

## 10. Static prepared-code checks

```bash
cd "${PROJECT_ROOT}/fsd_scene_multilabel8"
python -m compileall -q .
bash -n run_smoke.sh
bash -n run_train.sh
```

If syntax fails, STOP. Do not patch locally.

## 11. FSD factory check

```bash
FSD_ROOT="${FSD_ROOT}" python - <<'PY'
import os,sys
from pathlib import Path
fsd=Path(os.environ['FSD_ROOT'])
sys.path.insert(0,str(fsd))
from vision.ssd.config.fd_config import define_img_size
define_img_size(240)
from vision.ssd.mb_tiny_RFB_fd_3 import create_Mb_Tiny_RFB_fd_3_scene_noRFB
net=create_Mb_Tiny_RFB_fd_3_scene_noRFB(2,8)
print(type(net).__name__)
print('FSD_FACTORY_8LABEL=PASS')
PY
```

If this fails, STOP.

## 12. Generate our own multi-label manifests

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

Then immediately finalize supervision balance:

```bash
python -u finalize_manifest.py \
  --data-root "${LABEL_ROOT}" \
  2>&1 | tee "${LABEL_ROOT}/finalize.log"
```

This modifies only newly generated manifests.

## 13. Verify generated label-only dataset

Allowed generated files/extensions:

```text
.jsonl
.json
.log
```

Run:

```bash
python - <<'PY'
from pathlib import Path
root=Path('/data/pub1/z00919662/dataset/FSD_8scene_multilabel_manifests')
allowed={'.jsonl','.json','.log'}
bad=[p for p in root.rglob('*') if p.is_file() and p.suffix.lower() not in allowed]
print('unexpected_files',bad[:20])
assert not bad
for x in ['train.jsonl','val.jsonl','test.jsonl','dataset_summary.json','source_folder_audit.json','finalize_manifest_summary.json']:
    p=root/x
    print(x,p.exists(),p.stat().st_size if p.exists() else -1)
    assert p.exists() and p.stat().st_size>0
PY
```

No jpg/png/mask may appear under LABEL_ROOT.

## 14. Inspect 10_scenes mapping

```bash
cat "${LABEL_ROOT}/source_folder_audit.json"
```

Must confirm:

```text
Computer_synthesized -> objective_image
```

Inspect `ten_scenes_unmapped_folders`.

If an unmapped folder clearly corresponds to night/indoor/rain-snow/office/outdoor/landscape/sports/objective-image, STOP and return the complete folder list.

Do NOT edit alias mappings locally.

## 15. Manifest coverage + leakage audit

```bash
cd "${PROJECT_ROOT}/fsd_scene_multilabel8"
python -u audit_manifest.py --data-root "${LABEL_ROOT}" \
  2>&1 | tee "${LABEL_ROOT}/audit.log"
```

Hard requirement:

```text
all 8 labels have non-zero positive and non-zero negative supervision in train/val/test
```

Additional low-count warning thresholds, do not stop solely for these:

```text
train positive < 100 -> WARNING
val positive < 20    -> WARNING
test positive < 20   -> WARNING
```

Leakage rules:

```text
TRAIN vs VAL underlying overlap  -> HARD FAIL
TRAIN vs TEST underlying overlap -> HARD FAIL
VAL vs TEST overlap              -> WARNING ONLY
same underlying image twice inside one split -> WARNING ONLY
```

## 16. Confirm final objective supervision counts

After `finalize_manifest.py`, print objective counts by source:

```bash
python - <<'PY'
import json
from collections import defaultdict
from pathlib import Path
root=Path('/data/pub1/z00919662/dataset/FSD_8scene_multilabel_manifests')
for split in ['train','val','test']:
    d=defaultdict(lambda:{'pos':0,'neg':0,'unk':0})
    for line in (root/f'{split}.jsonl').read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        r=json.loads(line); v=int(r['labels']['objective_image']); s=r['source']
        d[s]['pos' if v==1 else 'neg' if v==0 else 'unk']+=1
    print(split,dict(d))
PY
```

Confirm Places365/SEG13 objective supervision is mostly/entirely unknown after finalization.

## 17. Choose one free GPU

```bash
nvidia-smi
```

Choose one genuinely free GPU and set:

```bash
export GPU=<index>
```

Do not kill other jobs.

## 18. Smoke test

Smoke uses compact coverage-preserving manifests automatically.

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

If OOM:

```text
retry BATCH=16
then BATCH=8
```

Do not change input size or architecture.

Smoke must prove:

```text
existing FSD YUV444 scene preprocessing works with our manifest loader
input reaches the existing FSD model
output shape is [B,8]
masked BCE loss is finite
backward succeeds
val/test evaluation succeeds
threshold calibration succeeds
best_fsd_multilabel8.pth is produced
```

If transform wrapper fails with both call styles, STOP and report both errors.

## 19. Formal training

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

Use smoke-proven smaller batch if required.

Fixed baseline:

```text
factory = create_Mb_Tiny_RFB_fd_3_scene_noRFB(2,8)
input_size = 240
loss = masked BCEWithLogitsLoss
SGD lr = 1e-2
momentum = 0.9
weight_decay = 1e-4
milestones = 95,150
epochs = 200
workers = 4
AMP = true
```

## 20. Resume interrupted formal training

If interrupted and valid checkpoint exists:

```bash
LAST="${OUTPUT_ROOT}/train_240/last_train_state.pth"
test -s "${LAST}"
```

Resume:

```bash
cd "${PROJECT_ROOT}/fsd_scene_multilabel8"

FSD_ROOT="${FSD_ROOT}" \
DATA_ROOT="${LABEL_ROOT}" \
OUT="${OUTPUT_ROOT}/train_240" \
GPU="${GPU}" \
BATCH=24 \
RESUME="${LAST}" \
bash run_train.sh
```

Do not restart epoch 0 when a valid resume state exists.

## 21. Expected outputs

```text
${OUTPUT_ROOT}/train_240/metrics.jsonl
${OUTPUT_ROOT}/train_240/last_train_state.pth
${OUTPUT_ROOT}/train_240/best_train_state.pth
${OUTPUT_ROOT}/train_240/best_val_per_class_0p5.csv
${OUTPUT_ROOT}/train_240/thresholds.json
${OUTPUT_ROOT}/train_240/test_per_class_calibrated.csv
${OUTPUT_ROOT}/train_240/test_summary.json
${OUTPUT_ROOT}/train_240/best_fsd_multilabel8.pth
${OUTPUT_ROOT}/train_240/deployment_metadata.json
```

## 22. FSD checkpoint load verification

```bash
FSD_ROOT="${FSD_ROOT}" python - <<'PY'
import os,sys
from pathlib import Path
fsd=Path(os.environ['FSD_ROOT']); sys.path.insert(0,str(fsd))
from vision.ssd.config.fd_config import define_img_size
define_img_size(240)
from vision.ssd.mb_tiny_RFB_fd_3 import create_Mb_Tiny_RFB_fd_3_scene_noRFB
net=create_Mb_Tiny_RFB_fd_3_scene_noRFB(2,8)
ck=Path('/data/pub1/z00919662/scene_multilabel/fsd_8label_v1/train_240/best_fsd_multilabel8.pth')
net.load(str(ck))
print('FSD_CHECKPOINT_LOAD=PASS')
PY
```

If this fails, STOP. Do not change `net.load()`.

## 23. Metrics

Report per class:

```text
threshold
known count
positive / negative
precision
recall
F1
accuracy
balanced accuracy
AP
```

Aggregate:

```text
macro-F1
macro-balanced-accuracy
macro-AP
```

Do not use simple overall accuracy as the primary metric.

Deployment semantics:

```text
8 raw logits
-> sigmoid independently
-> compare each with its threshold from thresholds.json
```

No softmax.

## 24. HARD STOP rules

Immediately output `HUMAN_ACTION_REQUIRED: YES` and STOP if:

1. staging repository has local modifications before update;
2. git fetch/checkout/ff-only fails;
3. local HEAD != remote branch HEAD;
4. any exact required dataset root is missing;
5. FSD repo cannot be located unambiguously;
6. prepared source syntax check fails;
7. FSD import/factory fails;
8. Places365 folder mapping fails;
9. Computer_synthesized does not map to objective_image;
10. a clearly relevant 10_scenes folder is unmapped and requires code mapping;
11. any of 8 labels has zero positive or zero negative in train/val/test;
12. TRAIN overlaps VAL or TEST at underlying-image level;
13. FSD scene transform fails through both compatibility call styles;
14. network output shape is not [B,8];
15. loss is NaN/Inf;
16. batches 24,16,8 all OOM;
17. training interruption cannot resume from a valid last state;
18. final FSD checkpoint cannot be loaded via existing net.load();
19. any fix requires editing prepared `.py/.sh/config` code.

When stopped, return:

```text
HUMAN_ACTION_REQUIRED: YES
FAILED_COMMAND:
FULL_TRACEBACK:
FILE_AND_LINE:
OBSERVED_LAYOUT/PATHS:
WHY_CODE_CHANGE_IS_REQUIRED:
```

Do NOT locally patch code.

## 25. Warning-only rules

Do NOT stop for:

```text
VAL/TEST overlap
same underlying image twice within the same split
low but non-zero positive counts
batch 24 OOM if 16/8 works
```

Report warnings.

## 26. Final report

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
places365:
coco:
coco_ade_13cls:
10_scenes:
computer_synthesized:

MANIFEST_ROOT:
train_records:
val_records:
test_records:

10_SCENES_FOLDER_MAPPING:
UNMAPPED_10_SCENES_FOLDERS:

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

OBJECTIVE_LABEL_COUNTS_BY_SOURCE:

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

If all stages finish:

```text
HUMAN_ACTION_REQUIRED: NO
```
