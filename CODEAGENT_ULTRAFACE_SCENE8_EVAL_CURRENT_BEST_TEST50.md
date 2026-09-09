# CodeAgent: evaluate current best UltraFace scene8 checkpoint + visualize 50 test images

This task is evaluation-only. Do NOT stop, restart, modify, or interfere with the currently running 200-epoch training.

## 1. GitHub

Repository:

`https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git`

Branch:

`agent/ultraface-slim-scene8-eval50-v1`

The branch contains:

`ultraface_scene_multilabel8/eval_current_best_and_test50.py`

This evaluation code uses exactly the same original UltraFace slim Mb_Tiny scene8 model definition as V1 training.

Repository synchronization rules:

- working tree must be clean before update;
- fetch the specified branch;
- checkout the specified branch;
- only fast-forward update is allowed;
- do not stash;
- do not reset --hard;
- do not locally patch or commit source-code fixes;
- if the prepared code has a bug, STOP and return the traceback/file/line.

Recommended independent checkout:

```bash
export REPO_URL="https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git"
export BRANCH="agent/ultraface-slim-scene8-eval50-v1"
export PROJECT_ROOT="/data/pub1/z00919662/scene_multilabel/rvm-video-semantic-segmentation-13class-eval50"

if [[ ! -d "${PROJECT_ROOT}/.git" ]]; then
  git clone --branch "${BRANCH}" --single-branch "${REPO_URL}" "${PROJECT_ROOT}"
else
  cd "${PROJECT_ROOT}"
  test -z "$(git status --porcelain)" || { echo "DIRTY_WORKTREE"; exit 2; }
  git fetch origin "${BRANCH}"
  git checkout "${BRANCH}"
  git merge --ff-only "origin/${BRANCH}"
fi

cd "${PROJECT_ROOT}"
echo BRANCH=$(git branch --show-current)
echo HEAD=$(git rev-parse HEAD)
```

## 2. Environment

Use the existing conda environment only:

`Ultraface`

Do not create a new environment and do not upgrade/downgrade torch/torchvision.

```bash
conda activate Ultraface
python - <<'PY'
import torch, cv2, numpy
print('python ok')
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available())
print('cv2', cv2.__version__)
PY
```

## 3. Current training files — READ ONLY

Training root:

`/data/pub1/z00919662/scene_multilabel/ultraface_slim_8label_640x360_v1/train`

Current best checkpoint:

`/data/pub1/z00919662/scene_multilabel/ultraface_slim_8label_640x360_v1/train/best_train_state.pth`

Manifest root:

`/data/pub1/z00919662/dataset/UltraFaceSlim_8scene_multilabel_manifests_640x360_v1`

Do NOT modify or delete anything in either directory.

Do NOT use `last_train_state.pth` for this evaluation. Use the current validation-best checkpoint `best_train_state.pth`.

## 4. Do not interfere with active training

The active 200-epoch training is currently on GPU 1. It must keep running.

Before evaluation:

```bash
nvidia-smi
ps -ef | grep -E 'train.py|watchdog' | grep -v grep || true
```

Rules:

- never kill the training process;
- never kill its DataLoader workers;
- never restart its watchdog;
- never use GPU 1 for evaluation while training is active;
- choose another genuinely idle GPU if available;
- if no other GPU is free, use CPU for evaluation rather than disturbing GPU 1.

Example if GPU 0 is idle:

```bash
export CUDA_VISIBLE_DEVICES=0
export EVAL_DEVICE=cuda:0
```

If no idle GPU exists:

```bash
export CUDA_VISIBLE_DEVICES=""
export EVAL_DEVICE=cpu
```

## 5. Snapshot the current best checkpoint

The training process may later replace `best_train_state.pth` if a better epoch is found. The evaluation must use one internally consistent snapshot.

Output root:

`/data/pub1/z00919662/scene_multilabel/ultraface_slim_8label_640x360_v1/eval_current_best_test50_20260909`

```bash
export TRAIN_ROOT="/data/pub1/z00919662/scene_multilabel/ultraface_slim_8label_640x360_v1/train"
export DATA_ROOT="/data/pub1/z00919662/dataset/UltraFaceSlim_8scene_multilabel_manifests_640x360_v1"
export OUT="/data/pub1/z00919662/scene_multilabel/ultraface_slim_8label_640x360_v1/eval_current_best_test50_20260909"
mkdir -p "${OUT}"
```

The Python evaluation script itself copies the supplied checkpoint into:

`${OUT}/best_train_state_snapshot.pth`

and evaluates that snapshot only.

## 6. Static code check

```bash
cd "${PROJECT_ROOT}/ultraface_scene_multilabel8"
python -m compileall -q model.py eval_current_best_and_test50.py
python model.py
```

Expected model test:

`input (1, 3, 360, 640) output (1, 8)`

## 7. Run full validation calibration + full test evaluation + test50 visualization

If using an idle GPU:

```bash
cd "${PROJECT_ROOT}/ultraface_scene_multilabel8"

python -u eval_current_best_and_test50.py \
  --checkpoint "${TRAIN_ROOT}/best_train_state.pth" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUT}" \
  --device "${EVAL_DEVICE}" \
  --batch-size 128 \
  --workers 8 \
  --num-vis 50 \
  --seed 20260909 \
  --amp \
  2>&1 | tee "${OUT}/eval.log"
```

If using CPU, omit `--amp` and reduce batch size to 32:

```bash
cd "${PROJECT_ROOT}/ultraface_scene_multilabel8"

python -u eval_current_best_and_test50.py \
  --checkpoint "${TRAIN_ROOT}/best_train_state.pth" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUT}" \
  --device cpu \
  --batch-size 32 \
  --workers 8 \
  --num-vis 50 \
  --seed 20260909 \
  2>&1 | tee "${OUT}/eval.log"
```

## 8. Evaluation protocol

Do NOT use 0.5 threshold as the final test threshold.

Protocol must be:

1. load current `best_train_state.pth` snapshot;
2. run the COMPLETE validation manifest;
3. independently calibrate one threshold per class on validation by maximizing F1;
4. freeze those 8 thresholds;
5. run the COMPLETE test manifest;
6. calculate test-set metrics using the frozen validation thresholds;
7. separately select 50 test images for qualitative visualization.

The 50 visualization images should be chosen with deterministic class-positive coverage where possible, then filled with random test samples. This is more useful than a purely random sample for rare labels.

The eight labels are:

1. night
2. indoor
3. rain_snow
4. office
5. outdoor
6. landscape
7. sports
8. objective_image

For metrics, unknown GT (`-1`) must be excluded per class.

## 9. Required quantitative outputs

The script should create:

- `${OUT}/best_train_state_snapshot.pth`
- `${OUT}/thresholds.json`
- `${OUT}/val_per_class_calibrated.csv`
- `${OUT}/test_per_class_calibrated.csv`
- `${OUT}/summary.json`
- `${OUT}/eval.log`

Print a table with these columns for all 8 classes:

`class | threshold | Precision | Recall | F1 | Balanced Accuracy | AP | known | positive | negative`

Also report:

- checkpoint epoch;
- checkpoint stored best macro-F1@0.5;
- calibrated test macro-Precision;
- calibrated test macro-Recall;
- calibrated test macro-F1;
- calibrated test macro-Balanced-Accuracy;
- calibrated test macro-AP.

Sort an additional short diagnostic table by F1 ascending so it is obvious which classes are dragging down overall performance.

## 10. Required 50-image qualitative outputs

The script should create:

`${OUT}/test50_vis/`

with 50 annotated JPEGs.

For each image, visualization must show:

- original image;
- known positive GT labels (`GT+`);
- predicted positive labels (`PRED+`);
- score for all 8 classes;
- calibrated threshold for all 8 classes;
- predicted 0/1 for each class;
- known GT 0/1 or `?` for unknown GT.

Do not modify the original test image.

Also create:

- `${OUT}/test50_predictions.csv`
- `${OUT}/test50_contact_sheet.jpg`

`test50_predictions.csv` must contain source image path, source dataset, GT labels, predicted labels, all 8 scores, all 8 thresholds, and all 8 GT/pred states.

## 11. Inspect failures, not just successes

After inference, automatically identify from the selected 50 images:

- false-positive labels;
- false-negative labels;
- images with the largest number of known-label mistakes.

Report at least the top 10 qualitative failure examples by filename/path and explain which known labels were wrong.

Unknown GT (`-1`) is not an error and must not be counted as FP/FN.

## 12. Important interpretation note

The manifest policy previously allowed validation/test overlap. Therefore final test metrics are useful engineering metrics but are not a perfectly independent benchmark. Keep this warning in the report:

`VAL_TEST_OVERLAP_WARNING: validation/test overlap is accepted by the user; thresholds are calibrated on validation, therefore test metrics may be optimistic.`

## 13. Failure rules

Do not modify GitHub prepared code locally.

STOP with `HUMAN_ACTION_REQUIRED: YES` only if:

- repository synchronization fails;
- `Ultraface` environment cannot run the model;
- current best checkpoint is missing/corrupt;
- manifest is missing/corrupt;
- prepared evaluation code has a real bug;
- both GPU and CPU evaluation are impossible.

If a GPU evaluation hits CUDA OOM, retry batch 64 then 32. Do not stop the training and do not use GPU 1.

If evaluation completes:

`HUMAN_ACTION_REQUIRED: NO`

## 14. Final CodeAgent response format

Return the following concise report:

```text
STATUS: PASS / FAIL
HUMAN_ACTION_REQUIRED: NO / YES

CHECKPOINT_SNAPSHOT:
CHECKPOINT_EPOCH:
CHECKPOINT_BEST_MACRO_F1_0P5:
EVAL_DEVICE:

TEST PER-CLASS:
class | threshold | precision | recall | F1 | BA | AP | known(pos/neg)
night
indoor
rain_snow
office
outdoor
landscape
sports
objective_image

TEST MACRO:
macro_precision:
macro_recall:
macro_f1:
macro_balanced_accuracy:
macro_ap:

LOWEST_F1_CLASSES:
1.
2.
3.

TEST50:
visualization_dir:
contact_sheet:
predictions_csv:

TOP_10_FAILURES:
1.
...
10.

VAL_TEST_OVERLAP_WARNING: validation/test overlap is accepted by the user; thresholds are calibrated on validation, therefore test metrics may be optimistic.
```
