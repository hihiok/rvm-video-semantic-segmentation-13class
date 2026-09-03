# Resume task: Places365 folder-layout fix, then continue NAS 9-label training

The previous run stopped because the local Places365 dataset does not contain `categories_places365.txt` or `IO_places365.txt` and is arranged as:

```text
/data/pub1/z00919662/segmentation/datasets/places365/
└── versions/1/
    ├── train/<category-folder>/*.jpg
    ├── val/<category-folder>/*.jpg
    ├── train.txt
    └── val.txt
```

GitHub has now been updated to support this exact layout. Do not modify code locally.

## What changed

The prepared code now:

- uses the category folder name as the Places365 source class;
- recognizes `versions/1/train/<category>` and `versions/1/val/<category>` directly;
- no longer requires local `categories_places365.txt`;
- no longer requires local `IO_places365.txt`;
- bundles the official Places365 `IO_places365.txt` taxonomy in the repository as metadata;
- maps flattened local category folders back to official canonical category names, e.g.:
  - `field-cultivated` -> `field/cultivated`
  - `office-cubicles` -> `office_cubicles`
  - `basketball-court-indoor` -> `basketball_court/indoor`
  - `desert-sand` -> `desert/sand`
  - `lake-natural` -> `lake/natural`
- still uses the existing curated canonical sets in `config.py` for landscape/sports/office;
- still uses the official Places365 taxonomy for indoor/outdoor. Do not infer indoor/outdoor from folder-name substrings.

No dataset download is needed.

## Hard restrictions

- DO NOT download any dataset.
- DO NOT download or manually create taxonomy files in the Places365 dataset directory.
- DO NOT rename/reorganize Places365 folders.
- DO NOT copy source images.
- DO NOT modify Python/config files locally.
- If the updated GitHub code still fails, stop and report traceback + exact folder names involved.
- Unknown labels remain `-1` and MUST stay masked from loss/metrics.
- This remains 9-label multi-label classification, not 9-way softmax.

## Repository

```bash
export REPO_URL="https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git"
export BRANCH="agent/nas-ultraface-slim-9label-v1-fullrun"
export PROJECT_ROOT="/data/pub1/z00919662/segmentation/nas-ultraface-slim-9label-v1-fullrun"
```

Update the existing checkout. Require a clean working tree:

```bash
set -euo pipefail

cd "${PROJECT_ROOT}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: local Git working tree is not clean"
  git status --short
  exit 2
fi

git fetch origin "${BRANCH}"
git checkout "${BRANCH}"
git merge --ff-only "origin/${BRANCH}"

echo "GITHUB_BRANCH=$(git branch --show-current)"
echo "GITHUB_COMMIT=$(git rev-parse HEAD)"
```

Verify new files:

```bash
test -f nas_scene_multilabel/metadata/IO_places365.txt
test -f nas_scene_multilabel/tools/test_places_folder_mapping.py
test -f nas_scene_multilabel/prepare_dataset.py
```

## 1. Static and mapping tests

```bash
cd "${PROJECT_ROOT}/nas_scene_multilabel"

python -m compileall -q .
python tools/test_places_folder_mapping.py
```

Required output:

```text
PLACES_FOLDER_MAPPING_TEST=PASS
```

If this fails, stop. Do not patch code locally.

## 2. Confirm actual local Places365 folders

```bash
export PLACES_ROOT="/data/pub1/z00919662/segmentation/datasets/places365"

 test -d "${PLACES_ROOT}/versions/1/train"
 test -d "${PLACES_ROOT}/versions/1/val"

printf 'train_category_dirs='; find "${PLACES_ROOT}/versions/1/train" -mindepth 1 -maxdepth 1 -type d | wc -l
printf 'val_category_dirs='; find "${PLACES_ROOT}/versions/1/val" -mindepth 1 -maxdepth 1 -type d | wc -l

find "${PLACES_ROOT}/versions/1/train" -mindepth 1 -maxdepth 1 -type d | sort | head -30
```

Do not require `categories_places365.txt` or local `IO_places365.txt` anymore.

## 3. Rebuild the 9-label manifests

Use only the existing datasets:

```bash
export COCO_ROOT="/data/pub1/z00919662/segmentation/datasets/coco"
export PLACES_ROOT="/data/pub1/z00919662/segmentation/datasets/places365"
export SEG_ROOT="/data/pub1/z00919662/segmentation/datasets/COCO_ADE_13cls_16x9_640x360"
export LABEL_ROOT="/data/pub1/z00919662/segmentation/datasets/NAS_9label_multilabel_manifests"
export OUTPUT_ROOT="/data/pub1/z00919662/segmentation/nas_scene_tagging/ultraface_slim_9label_v1"
```

Only recreate the generated manifest directory:

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

The log should show resolved Places365 paths similar to:

```text
PLACES_TRAIN_ROOT=/data/pub1/z00919662/segmentation/datasets/places365/versions/1/train
PLACES_VAL_ROOT=/data/pub1/z00919662/segmentation/datasets/places365/versions/1/val
PLACES_IO_SOURCE=.../nas_scene_multilabel/metadata/IO_places365.txt
```

It must also report non-zero recognized category directory counts.

If any actual Places365 folder cannot be mapped to an official canonical category, the code intentionally stops with `Unrecognized Places365 category folders`. In that case return the complete list/preview and stop; do not guess or locally edit mappings.

## 4. Audit generated labels

Required files:

```bash
test -s "${LABEL_ROOT}/train.jsonl"
test -s "${LABEL_ROOT}/val.jsonl"
test -s "${LABEL_ROOT}/test.jsonl"
test -s "${LABEL_ROOT}/dataset_summary.json"

cat "${LABEL_ROOT}/dataset_summary.json"
```

Report for every label in train/val/test:

```text
positive / negative / unknown
```

Labels:

```text
indoor
outdoor
landscape
sports
food
animal
building
sky
office
```

Every label must have at least one positive and one negative in all three splits. If not, stop and report the exact split/class/counts.

Confirm no images were copied into LABEL_ROOT:

```bash
python - <<'PY'
from pathlib import Path
root=Path('/data/pub1/z00919662/segmentation/datasets/NAS_9label_multilabel_manifests')
image_ext={'.jpg','.jpeg','.png','.bmp','.webp'}
bad=[p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in image_ext]
print('copied_images=', len(bad))
assert not bad
PY
```

## 5. Exact-path leakage check

```bash
python - <<'PY'
import json
from pathlib import Path
root=Path('/data/pub1/z00919662/segmentation/datasets/NAS_9label_multilabel_manifests')
sets={}
for split in ['train','val','test']:
    sets[split]={
        json.loads(x)['image']
        for x in (root/f'{split}.jsonl').read_text(encoding='utf-8').splitlines()
        if x.strip()
    }
for a,b in [('train','val'),('train','test'),('val','test')]:
    inter=sets[a]&sets[b]
    print(a,b,'exact_path_overlap=',len(inter))
    assert not inter, f'exact path leakage: {a}/{b}'
PY
```

Also repeat the previously requested COCO-vs-SEG derivative leakage sanity check. If the same identifiable underlying COCO image lands in different train/val/test splits through different source paths, stop and report it.

## 6. Continue model execution after dataset audit passes

Once dataset generation + coverage + leakage audits pass, continue the already prepared full runbook:

```text
CODEAGENT_NAS_ULTRAFACE_SLIM_9LABEL_FULL_EXECUTION.md
```

Continue from the GPU selection / smoke training section onward. Do NOT rerun dataset downloads (there are none).

Key model settings remain:

```text
input_size = 224
base_channel = 16
UltraFace Mb_Tiny/slim backbone
NO RFB
NO SSD extras
NO bbox/conf heads
AdaptiveAvgPool2d(1)
Linear(256 -> 9)
masked BCEWithLogitsLoss
single GPU
workers <= 4
60 formal epochs after 1-epoch smoke
```

Remember the actual training CLI uses:

```bash
python train.py \
  --data-root "${LABEL_ROOT}" \
  --output-dir ...
```

There is no `--amp` CLI argument; AMP is enabled internally on CUDA.

If smoke passes, run the formal 60-epoch training, validation threshold calibration, independent test evaluation, and ONNX export exactly as specified in the full runbook.

## 7. Final response

If successful, return the complete final report requested by the full runbook, including:

- GitHub branch + commit
- resolved Places365 train/val roots
- taxonomy source path
- number of recognized Places365 category directories
- train/val/test image counts by source
- per-class pos/neg/unknown counts
- leakage audit
- model params/MACs
- smoke result
- 60-epoch training result
- per-class calibrated test Precision/Recall/F1/Accuracy/Balanced Accuracy/AP
- macro-F1 / macro-balanced-accuracy / macro-AP
- best checkpoint
- ONNX path/check

If all complete:

```text
HUMAN_ACTION_REQUIRED: NO
```

If blocked by a genuine code/layout issue:

```text
HUMAN_ACTION_REQUIRED: YES
```

Report the exact error and do not locally modify prepared source code.
