# CodeAgent Full Runbook — FSD/UltraFace 8-label multi-label scene classification @ 640x360

## 0. Goal

Prepare multi-label manifests from the user's existing datasets and train an 8-label scene classifier using the user's existing FSD/UltraFace scene network.

This is a new 8-label product taxonomy. Do not continue the old 9-label taxonomy.

Exact output order:

0. `night` — 夜景
1. `indoor` — 室内
2. `rain_snow` — 雨/雪
3. `office` — 办公场景
4. `outdoor` — 户外
5. `landscape` — 风景
6. `sports` — 运动
7. `objective_image` — 客观图（电脑合成 pattern、解析度卡、测试卡等）

This is multi-label classification. One image may contain multiple labels, e.g. `night=1,outdoor=1,landscape=1` or `office=1,indoor=1`.

Label values in generated manifests:

- `1` = confirmed positive
- `0` = confirmed negative
- `-1` = unknown / not supervised by this source

`-1` MUST be ignored by loss and metrics. Never convert unknown to negative.

## 1. User FSD reference behavior that MUST be preserved

The user's reference scene trainer uses the FSD factory:

`create_Mb_Tiny_RFB_fd_3_scene_noRFB`

and the existing FSD scene preprocessing:

- `YUVTrainAugmentation_scene`
- `YUVTestTransform_scene`

The old reference is single-label CrossEntropy. This task changes only the scene objective/data interface to 8 independent logits + masked BCEWithLogitsLoss.

The user's old reference uses scalar `input_size=240` and produces scene tensors shaped `[B,1,240,320]`. For this task the real input MUST be explicit horizontal 16:9:

`[B,1,360,640]`

Do not silently train 640x480, 480x360, 360x480, 320x240, or any square resolution.

## 2. GitHub source of truth

Repository:

`https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git`

Branch:

`agent/fsd-scene-multilabel8-640x360-v1`

Checkout path:

`/data/pub1/z00919662/segmentation/fsd-scene-multilabel8-640x360-v1`

### Strict repository update rule

Run:

```bash
export REPO_URL="https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git"
export BRANCH="agent/fsd-scene-multilabel8-640x360-v1"
export PROJECT_ROOT="/data/pub1/z00919662/segmentation/fsd-scene-multilabel8-640x360-v1"

if [[ ! -d "${PROJECT_ROOT}/.git" ]]; then
  git clone --branch "${BRANCH}" --single-branch "${REPO_URL}" "${PROJECT_ROOT}"
else
  cd "${PROJECT_ROOT}"
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "HUMAN_ACTION_REQUIRED: YES"
    echo "STOP_REASON: project working tree is not clean"
    git status --short
    exit 2
  fi
  git fetch origin "${BRANCH}"
  git checkout "${BRANCH}"
  git merge --ff-only "origin/${BRANCH}"
fi

cd "${PROJECT_ROOT}"
LOCAL_HEAD="$(git rev-parse HEAD)"
REMOTE_HEAD="$(git rev-parse origin/${BRANCH})"
echo "LOCAL_HEAD=${LOCAL_HEAD}"
echo "REMOTE_HEAD=${REMOTE_HEAD}"
test "${LOCAL_HEAD}" = "${REMOTE_HEAD}"
test -z "$(git status --porcelain)"
```

Forbidden Git behavior:

- no `git stash`
- no `git reset --hard`
- no local patch
- no local commit to work around errors
- no editing prepared `.py/.sh/.md/config` files

If prepared code must change, STOP and report. ChatGPT will update GitHub centrally.

## 3. Existing datasets — READ ONLY

Use these exact roots:

```bash
export COCO_ROOT="/data/pub1/z00919662/segmentation/datasets/coco"
export PLACES_ROOT="/data/pub1/z00919662/segmentation/datasets/places365"
export SEG_ROOT="/data/pub1/z00919662/segmentation/datasets/COCO_ADE_13cls_16x9_640x360"
export TEN_SCENES_ROOT="/data/pub1/z00919662/dataset/10_scenes"
export COMPUTER_SYNTH_ROOT="/data/pub1/z00919662/dataset/10_scenes/train/Computer_synthesized"
```

`SEG_ROOT` is the user's existing COCO+ADE20K-derived 13-class static segmentation dataset and is already 640x360.

Do NOT:

- rename images
- resize/overwrite images
- modify masks
- modify annotations
- modify existing labels
- reorganize folders
- create a duplicate source image dataset
- download any new dataset

Only generate new JSONL manifests that reference original absolute image paths.

New manifest root:

```bash
export MANIFEST_ROOT="/data/pub1/z00919662/dataset/FSD_8scene_multilabel_manifests_640x360_v1"
```

Training output root:

```bash
export OUTPUT_ROOT="/data/pub1/z00919662/scene_multilabel/fsd_8label_640x360_v1"
```

## 4. Dataset-source policy

### Places365

Use for:

- indoor / outdoor through official Places365 IO taxonomy metadata stored in this GitHub branch
- landscape through conservative curated category mapping
- sports through conservative sports-scene/venue mapping
- office through office/workplace-related mapping
- snow positives only for clearly snowy/icy Places categories

Do not infer generic rain from Places365.

The user's local Places365 layout is supported:

`/data/pub1/z00919662/segmentation/datasets/places365/versions/1/train/<category>/...`

`/data/pub1/z00919662/segmentation/datasets/places365/versions/1/val/<category>/...`

Folder names are resolved against the repository taxonomy rather than guessed.

### COCO

Use only as sampled real-photo negative supervision for:

`objective_image=0`

Do not infer the other seven scene labels from COCO object categories in this baseline.

### COCO/ADE 13-class semantic dataset

Use masks conservatively for:

- `rain_snow=1` only when `ice_or_snow` mask area >= 1%
- zero ice/snow pixels may provide `rain_snow=0`
- `landscape=1` when combined natural semantic area (`plant/water/desert/ice_or_snow/mountain`) >= 30%

Do not claim snow mask supplies generic rain examples. Rain examples should come from `10_scenes` if present.

### 10_scenes

Use as an extra source for the eight product classes when mapped folder names exist.

The required computer-synthesized positive root is:

`/data/pub1/z00919662/dataset/10_scenes/train/Computer_synthesized`

Every mapped `Computer_synthesized` image must have:

`objective_image=1`

and the other seven labels are confirmed `0` for this synthetic/test-pattern class.

Other mapped 10_scenes folders may supplement night/indoor/rain_snow/office/outdoor/landscape/sports.

Unmapped 10_scenes folders must NOT automatically become negatives. They must remain unused and be listed in `source_folder_audit.json`.

If an unmapped folder clearly looks relevant to one of the eight product labels, STOP and report the folder list rather than inventing a mapping locally.

## 5. Source audit

Verify all roots before doing anything else:

```bash
for p in \
  "${COCO_ROOT}" \
  "${PLACES_ROOT}" \
  "${SEG_ROOT}" \
  "${TEN_SCENES_ROOT}" \
  "${COMPUTER_SYNTH_ROOT}"; do
  if [[ ! -d "${p}" ]]; then
    echo "HUMAN_ACTION_REQUIRED: YES"
    echo "MISSING=${p}"
    exit 2
  fi
  echo "FOUND=${p}"
  du -sh "${p}" || true
done
```

Show the 10_scenes category directories without modifying anything:

```bash
find "${TEN_SCENES_ROOT}" -maxdepth 2 -mindepth 2 -type d -print | sort
```

Confirm Computer_synthesized has images:

```bash
find "${COMPUTER_SYNTH_ROOT}" -type f \
  \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.webp' \) \
  | head -20
```

## 6. Locate the existing FSD repository

Do not clone or replace the user's FSD project. Reuse the existing one.

First test the known likely location:

```bash
CANDIDATE_FSD="/mnt/ssd1/z00919662/AI-face-detect/ultraface_3323_ref_param"
if [[ -f "${CANDIDATE_FSD}/vision/ssd/mb_tiny_RFB_fd_3.py" ]]; then
  export FSD_ROOT="${CANDIDATE_FSD}"
fi
```

If not found, locate it conservatively:

```bash
if [[ -z "${FSD_ROOT:-}" ]]; then
  find /mnt/ssd1/z00919662 /data/pub1/z00919662 \
    -maxdepth 7 -type f -path '*/vision/ssd/mb_tiny_RFB_fd_3.py' -print 2>/dev/null | head -20
fi
```

Choose the repository that also contains:

- `vision/ssd/data_preprocessing.py`
- `vision/ssd/config/fd_config.py`
- `create_Mb_Tiny_RFB_fd_3_scene_noRFB`
- `YUVTrainAugmentation_scene`
- `YUVTestTransform_scene`

Print only the chosen path:

```bash
echo "FSD_ROOT=${FSD_ROOT}"
```

Do not modify the FSD repository.

## 7. Python/CUDA environment

Use the same working environment used by FSD on this server whenever possible.

Set CPU limits:

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
```

Verify:

```bash
which python
python - <<'PY'
import sys, torch, cv2, numpy
print('python:', sys.version)
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
print('gpu_count:', torch.cuda.device_count())
print('opencv:', cv2.__version__)
assert torch.cuda.is_available()
PY
```

Do not upgrade/downgrade working torch or CUDA packages just for this task.

## 8. Static checks

```bash
cd "${PROJECT_ROOT}/fsd_scene_multilabel8"
python -m compileall -q .
bash -n run_smoke.sh
bash -n run_train.sh
```

If syntax fails, STOP. Do not patch locally.

## 9. Mandatory FSD 640x360 compatibility smoke BEFORE dataset training

Run:

```bash
cd "${PROJECT_ROOT}/fsd_scene_multilabel8"
python -u check_fsd_640x360.py --fsd-root "${FSD_ROOT}"
```

Required output:

```text
TRANSFORMED_SHAPE= (1, 360, 640)
MODEL_OUTPUT_SHAPE= (1, 8)
FSD_640X360_FACTORY_TEST=PASS
```

This check uses the same reference FSD scene transform names:

- `YUVTestTransform_scene`
- `create_Mb_Tiny_RFB_fd_3_scene_noRFB(2,8)`

If the transform does not produce `[1,360,640]`, or the existing FSD scene head cannot output `[1,8]` from `[1,1,360,640]`, STOP with the full traceback.

Do NOT edit FSD to make it fit.

## 10. Generate new multi-label manifests

Only the generated manifest directory may be recreated:

```bash
rm -rf "${MANIFEST_ROOT}"
mkdir -p "${MANIFEST_ROOT}"

cd "${PROJECT_ROOT}/fsd_scene_multilabel8"

python -u prepare_manifest.py \
  --places-root "${PLACES_ROOT}" \
  --coco-root "${COCO_ROOT}" \
  --seg-root "${SEG_ROOT}" \
  --ten-scenes-root "${TEN_SCENES_ROOT}" \
  --output-root "${MANIFEST_ROOT}" \
  --places-train-cap-per-class 500 \
  --coco-objective-neg-train-cap 5000 \
  --coco-objective-neg-eval-cap 1000 \
  --seed 20260904 \
  --snow-min-area 0.01 \
  --landscape-min-area 0.30 \
  2>&1 | tee "${MANIFEST_ROOT}/prepare_manifest.log"
```

Then run the prepared finalizer. It modifies only our newly generated JSONL files; it never touches source images/labels:

```bash
python -u finalize_manifest.py \
  --data-root "${MANIFEST_ROOT}" \
  2>&1 | tee "${MANIFEST_ROOT}/finalize_manifest.log"
```

Expected files include:

- `train.jsonl`
- `val.jsonl`
- `test.jsonl`
- `dataset_summary.json`
- `source_folder_audit.json`
- `finalize_manifest_summary.json`

Verify no image/mask was copied into the manifest root:

```bash
python - <<'PY'
from pathlib import Path
root=Path('/data/pub1/z00919662/dataset/FSD_8scene_multilabel_manifests_640x360_v1')
allowed={'.jsonl','.json','.log'}
bad=[p for p in root.rglob('*') if p.is_file() and p.suffix.lower() not in allowed]
print('unexpected_files=', bad[:20])
assert not bad
PY
```

## 11. Manifest audit and leakage rule

Run:

```bash
cd "${PROJECT_ROOT}/fsd_scene_multilabel8"
python -u audit_manifest.py --data-root "${MANIFEST_ROOT}" \
  2>&1 | tee "${MANIFEST_ROOT}/audit_manifest.log"
```

Required:

`AUDIT=PASS`

Each of the eight labels must have at least one positive and one negative in train, val, and test.

Leakage policy:

- TRAIN vs VAL same underlying image: HARD FAIL
- TRAIN vs TEST same underlying image: HARD FAIL
- VAL vs TEST overlap: WARNING ONLY; user explicitly allows it
- duplicate records inside the same split: WARNING/diagnostic only unless they create contradictory labels

If val/test overlap exists, final report must state that validation-calibrated thresholds make test metrics not strictly independent and potentially optimistic.

## 12. Inspect 10_scenes mapping before training

Print:

```bash
cat "${MANIFEST_ROOT}/source_folder_audit.json"
```

Confirm `Computer_synthesized` maps to `objective_image`.

If `ten_scenes_unmapped_folders` contains a folder whose name clearly corresponds to night, indoor, rain/snow, office, outdoor, landscape, sports, objective/test-pattern/resolution-chart content, STOP and report it. Do not locally add aliases.

## 13. Select one free GPU

```bash
nvidia-smi
```

Choose one genuinely free GPU. Do not kill another user's process.

Example only:

```bash
export GPU=0
export CUDA_VISIBLE_DEVICES="${GPU}"
```

Training is single-GPU.

## 14. Short 640x360 smoke training

Use the prepared limited-step smoke. This is intentionally not a full large-data epoch.

```bash
cd "${PROJECT_ROOT}/fsd_scene_multilabel8"
export DATA_ROOT="${MANIFEST_ROOT}"
export SMOKE_DATA_ROOT="/data/pub1/z00919662/dataset/FSD_8scene_multilabel_smoke_manifests_640x360_v1"
export OUT="${OUTPUT_ROOT}/smoke"
export BATCH=24

bash run_smoke.sh
```

Smoke uses:

- input = 640x360
- FSD tensor = `[B,1,360,640]`
- factory = `create_Mb_Tiny_RFB_fd_3_scene_noRFB(2,8)`
- batch = 24
- workers = 4
- FP16 AMP
- only 20 train steps
- only 8 validation/test batches

If batch 24 OOMs, retry only:

`24 -> 16 -> 8 -> 4`

Do not change input resolution.

If batch 4 also OOMs, STOP.

Smoke PASS requires:

- transform shape is correct
- forward `[B,1,360,640] -> [B,8]`
- masked BCE handles `-1`
- backward/optimizer work
- validation works
- threshold calibration/test code works
- checkpoint files are produced

## 15. Formal 640x360 training

If smoke passes, use the working batch size from smoke.

```bash
cd "${PROJECT_ROOT}/fsd_scene_multilabel8"
export DATA_ROOT="${MANIFEST_ROOT}"
export OUT="${OUTPUT_ROOT}/train_640x360"
export BATCH=24
unset RESUME || true

bash run_train.sh
```

If smoke required batch 16/8/4, set `BATCH` to that value instead.

Formal training baseline:

- input = 640x360 horizontal 16:9
- model = existing FSD `create_Mb_Tiny_RFB_fd_3_scene_noRFB(2,8)`
- existing FSD YUV scene transforms
- multi-label masked BCEWithLogitsLoss
- SGD
- lr = 1e-2
- momentum = 0.9
- weight decay = 1e-4
- milestones = 95,150
- gamma = 0.1
- epochs = 200
- workers = 4
- AMP = true
- gradient clipping = 5.0

The optimizer/training schedule intentionally follows the user's reference FSD scene-training style, except for the multi-label objective, CPU-worker limit, and explicit 640x360 input.

## 16. Resume rule

The formal output must write:

`last_train_state.pth`

If training is interrupted and this checkpoint is valid, resume instead of restarting:

```bash
export DATA_ROOT="${MANIFEST_ROOT}"
export OUT="${OUTPUT_ROOT}/train_640x360"
export BATCH=<same batch used before interruption>
export RESUME="${OUT}/last_train_state.pth"
cd "${PROJECT_ROOT}/fsd_scene_multilabel8"
bash run_train.sh
```

The trainer checks checkpoint input resolution and must reject a checkpoint from a different resolution.

## 17. Final evaluation

The trainer selects the best validation macro-F1 checkpoint using threshold 0.5 during training.

After training it:

1. calibrates one threshold per label on validation;
2. fixes those thresholds;
3. evaluates on test.

Report per class:

- calibrated threshold
- known positive/negative counts
- Precision
- Recall
- F1
- Accuracy
- Balanced Accuracy
- AP

Report aggregates:

- macro-F1
- macro-balanced-accuracy
- macro-AP

Do not use ordinary overall accuracy as the primary acceptance metric for this multi-label task.

Expected final files:

- `best_train_state.pth`
- `last_train_state.pth`
- `best_val_per_class_0p5.csv`
- `thresholds.json`
- `test_per_class_calibrated.csv`
- `test_summary.json`
- `deployment_metadata.json`
- `best_fsd_multilabel8_640x360.pth`
- `metrics.jsonl`
- `train.log`

## 18. Mandatory final FSD checkpoint load test

After formal training:

```bash
cd "${PROJECT_ROOT}/fsd_scene_multilabel8"
python - <<'PY'
import os, sys, torch
from pathlib import Path

fsd=Path(os.environ['FSD_ROOT']).resolve()
out=Path('/data/pub1/z00919662/scene_multilabel/fsd_8label_640x360_v1/train_640x360')
sys.path.insert(0,str(fsd))
from vision.ssd.config.fd_config import define_img_size
define_img_size(640)
from vision.ssd.mb_tiny_RFB_fd_3 import create_Mb_Tiny_RFB_fd_3_scene_noRFB
net=create_Mb_Tiny_RFB_fd_3_scene_noRFB(2,8)
ckpt=out/'best_fsd_multilabel8_640x360.pth'
assert ckpt.exists(), ckpt
net.load(str(ckpt))
net.eval()
with torch.no_grad():
    y=net(torch.zeros(1,1,360,640))
print('FINAL_LOAD_OUTPUT_SHAPE=',tuple(y.shape))
assert tuple(y.shape)==(1,8)
print('FINAL_FSD_LOAD_TEST=PASS')
PY
```

## 19. Failure / stop rules

Output `HUMAN_ACTION_REQUIRED: YES` and STOP if any of these occurs:

1. Git working tree is dirty before update.
2. fetch/checkout/ff-only fails.
3. local HEAD != remote branch HEAD after sync.
4. one of the exact source dataset roots is missing.
5. Computer_synthesized root is missing/empty.
6. existing FSD repo cannot be found.
7. FSD reference functions `YUVTrainAugmentation_scene` or `YUVTestTransform_scene` are missing.
8. `create_Mb_Tiny_RFB_fd_3_scene_noRFB` is missing.
9. `check_fsd_640x360.py` does not output both `(1,360,640)` and `(1,8)`.
10. prepared Python/shell code has a bug.
11. a relevant 10_scenes folder is unmapped and needs a new mapping decision.
12. any of the eight labels has zero positive or zero negative in train/val/test.
13. TRAIN-vs-VAL underlying leakage exists.
14. TRAIN-vs-TEST underlying leakage exists.
15. CUDA environment unavailable.
16. no free GPU available.
17. smoke still OOM at batch 4.
18. smoke has non-finite loss or incorrect output shape.
19. formal training needs source-code modification to continue.
20. final FSD checkpoint cannot reload and output `[1,8]` for `[1,1,360,640]`.

Do NOT stop for:

- val/test overlap only — warning is enough;
- batch 24 OOM when 16/8/4 works;
- same-split duplicate source derivatives without contradictory labels;
- slow training by itself.

Do not locally patch prepared code when stopping. Report full traceback, command, file/line, environment, and relevant paths.

## 20. Final report format

Return:

```text
STATUS: PASS / FAIL
HUMAN_ACTION_REQUIRED: YES / NO

GITHUB_BRANCH:
GITHUB_COMMIT:
PROJECT_ROOT:
FSD_ROOT:

PYTHON_VERSION:
TORCH_VERSION:
GPU:
CUDA_VISIBLE_DEVICES:

SOURCE_DATA_MODIFIED: NO
SOURCE_LABELS_MODIFIED: NO
SOURCE_IMAGES_COPIED: NO
NEW_DATA_DOWNLOADED: NO

INPUT:
width: 640
height: 360
channels: 1
scene_tensor: [B,1,360,640]

MODEL_FACTORY:
create_Mb_Tiny_RFB_fd_3_scene_noRFB(2,8)
MODEL_PARAMS:

LABELS:
night
indoor
rain_snow
office
outdoor
landscape
sports
objective_image

DATA_ROOTS:
COCO_ROOT:
PLACES_ROOT:
SEG_ROOT:
TEN_SCENES_ROOT:
COMPUTER_SYNTH_ROOT:
MANIFEST_ROOT:

DATASET_COUNTS:
train total / by source
val total / by source
test total / by source

PER_CLASS_LABEL_COUNTS:
label | train pos/neg/unknown | val pos/neg/unknown | test pos/neg/unknown

10_SCENES_MAPPING:
Computer_synthesized mapping:
unmapped folders:

LEAKAGE:
train/val underlying overlap:
train/test underlying overlap:
val/test underlying overlap:

FSD_640X360_FACTORY_TEST: PASS / FAIL
TRANSFORMED_SHAPE:
MODEL_OUTPUT_SHAPE:

SMOKE:
status:
batch_size:

FORMAL_TRAINING:
epochs_target: 200
epochs_completed:
batch_size:
workers: 4
best_epoch:
best_val_macro_f1_0p5:

TEST:
label | threshold | precision | recall | F1 | accuracy | balanced_accuracy | AP | known(pos/neg)

MACRO_F1:
MACRO_BALANCED_ACCURACY:
MACRO_AP:

BEST_FSD_CHECKPOINT:
LAST_TRAIN_STATE:
THRESHOLDS_JSON:
TEST_METRICS_CSV:
DEPLOYMENT_METADATA:
FINAL_FSD_LOAD_TEST: PASS / FAIL

WARNINGS:
```

If val/test overlap exists, include:

`VAL_TEST_OVERLAP_WARNING: accepted by user; validation-calibrated thresholds mean test metrics are not strictly independent and may be optimistic.`

If everything completes successfully:

`HUMAN_ACTION_REQUIRED: NO`
