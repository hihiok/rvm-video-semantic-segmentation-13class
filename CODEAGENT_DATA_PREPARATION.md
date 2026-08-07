# CodeAgent Task: Prepare VIPSeg/VSPW for 13-Class RVM Video Segmentation

## Goal

Prepare a true video dataset with continuous frame folders and masks using this immutable class order:

```text
0 background
1 sky
2 person
3 plant
4 building
5 flower
6 food
7 water
8 desert
9 ice_or_snow
10 text
11 ball
12 mountain
```

The primary source is official VIPSeg. Do not treat its original panoptic values as semantic IDs. Use the converter in this repository.

## Required paths

```bash
REPO=/data/pub1/z00919662/segmentation/rvm-video-semantic-segmentation-13class
DATA_BASE=/data/pub1/z00919662/dataset
VIPSEG_DOWNLOAD=${DATA_BASE}/VIPSeg_download
VIPSEG_META=${DATA_BASE}/VIPSeg-Dataset-meta
OUTPUT=${DATA_BASE}/VIPSeg_13cls_video
```

Do not delete or overwrite any existing dataset. If `OUTPUT` already exists, inspect it and run in resume mode without `--overwrite`.

## 1. Clone code and official metadata

```bash
mkdir -p /data/pub1/z00919662/segmentation /data/pub1/z00919662/dataset

if [ ! -d "${REPO}/.git" ]; then
  git clone https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git "${REPO}"
else
  git -C "${REPO}" pull --ff-only
fi

if [ ! -d "${VIPSEG_META}/.git" ]; then
  git clone https://github.com/VIPSeg-Dataset/VIPSeg-Dataset.git "${VIPSEG_META}"
else
  git -C "${VIPSEG_META}" pull --ff-only
fi
```

Confirm metadata:

```bash
test -f "${VIPSEG_META}/train.txt"
test -f "${VIPSEG_META}/val.txt"
test -f "${VIPSEG_META}/test.txt"
test -f "${VIPSEG_META}/panoVIPSeg_categories.json"
wc -l "${VIPSEG_META}/train.txt" "${VIPSEG_META}/val.txt" "${VIPSEG_META}/test.txt"
```

Expected split sizes are 2806 train videos, 343 val videos, and 387 test videos.

## 2. Download VIPSeg from the official release

VIPSeg is released for non-commercial research. Download only through the official links and preserve its license/README.

Official Google Drive file ID:

```text
1B13QUiE82xf7N6nVHclb4ErN-Zuai-sZ
```

Try:

```bash
python -m pip install --upgrade gdown
mkdir -p "${VIPSEG_DOWNLOAD}"
gdown --fuzzy 'https://drive.google.com/file/d/1B13QUiE82xf7N6nVHclb4ErN-Zuai-sZ/view?usp=sharing' \
  -O "${VIPSEG_DOWNLOAD}/VIPSeg_release.zip"
unzip -q "${VIPSEG_DOWNLOAD}/VIPSeg_release.zip" -d "${VIPSEG_DOWNLOAD}"
```

If the official download is already present, do not download it again. Locate the extracted root that directly contains `images/` and `panomasks/`:

```bash
find "${VIPSEG_DOWNLOAD}" -maxdepth 4 -type d -name panomasks
```

Set it explicitly, for example:

```bash
VIPSEG_ROOT=${VIPSEG_DOWNLOAD}/VIPSeg_720P
test -d "${VIPSEG_ROOT}/images"
test -d "${VIPSEG_ROOT}/panomasks"
```

If Google Drive is blocked, report the exact failure and retain all downloaded partial files. Do not use unofficial mirrors or bypass access controls.

## 3. Create the Python environment

```bash
cd "${REPO}"
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy Pillow tqdm pytest opencv-python easyocr
```

## 4. Unit-test label decoding

```bash
cd "${REPO}"
PYTHONPATH=. pytest -q tests/test_label_mapping.py
```

This must pass before processing the full dataset. It verifies VOID, stuff IDs, thing instance IDs, background mapping, ball, and mountain.

## 5. Convert VIPSeg

Use hardlinks to avoid duplicating image storage when source and output are on the same filesystem. The converter automatically falls back to copying if hardlinks are unavailable.

```bash
cd "${REPO}"
python tools/prepare_vipseg_13class.py \
  --vipseg-root "${VIPSEG_ROOT}" \
  --metadata-root "${VIPSEG_META}" \
  --output-root "${OUTPUT}" \
  --mapping configs/vipseg_to_13class.json \
  --copy-mode hardlink \
  --workers 16
```

The command is resumable. Existing converted masks/images are reused unless `--overwrite` is deliberately added.

## 6. OCR-filter `painting_or_poster` before adding it to text

The strict mapping in step 5 deliberately excludes VIPSeg's combined `painting_or_poster` source class. Now examine each poster/painting instance and add the entire carrier to `text=10` only when EasyOCR finds text inside that instance.

Check whether CUDA PyTorch is available:

```bash
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

Run on GPU when available:

```bash
cd "${REPO}"
python tools/add_text_posters_with_easyocr.py \
  --vipseg-root "${VIPSEG_ROOT}" \
  --metadata-root "${VIPSEG_META}" \
  --converted-root "${OUTPUT}" \
  --splits train,val,test \
  --languages ch_sim,en \
  --gpu \
  --min-confidence 0.20 \
  --min-characters 2 \
  --min-polygon-overlap 0.50 \
  --min-text-area-ratio 0.001 \
  --qa-limit-per-decision 200
```

If CUDA is unavailable, replace `--gpu` with `--no-gpu`. Do not broaden the base mapping to all posters.

The OCR stage must create:

```text
${OUTPUT}/painting_or_poster_ocr_audit.jsonl
${OUTPUT}/painting_or_poster_ocr_summary.json
${OUTPUT}/qa_poster_ocr/accepted/
${OUTPUT}/qa_poster_ocr/rejected/
```

Manually inspect at least 100 accepted and 100 rejected crops (or all if fewer exist). Check that accepted detections lie on the annotated carrier, not on nearby subtitles, signs, or watermarks. If thresholds are changed, rerun with `--overwrite-audit`; the script reconstructs all `painting_or_poster` decisions so stale accepted masks are removed.

Record acceptance rate by split. A high acceptance rate by itself is not success; precision is more important because pure paintings must remain background.

## 7. Validate every converted frame

```bash
python tools/check_video_dataset.py \
  --data-root "${OUTPUT}" \
  --splits train,val,test \
  --output-json "${OUTPUT}/validation_report.json"
```

Acceptance criteria:

1. Every image has exactly one same-stem mask under the same video ID.
2. Image and mask resolutions match.
3. Every mask contains only `0..12` and optional `255`.
4. Train and val each contain multiple consecutive frames per video.
5. Class 12 (`mountain`) has non-zero pixels and occurs in non-zero frames.
6. `dataset_summary.json` and `validation_report.json` exist.
7. OCR audit and summary files exist; only OCR-accepted poster instances were added to `text=10`.
8. No original VIPSeg files were modified.

Print a concise table with video count, frame count, pixels per class, and frames containing each class. Explicitly report if any rare class has zero samples.

## 8. Visual spot check

Select at least 20 frames across at least 5 videos, including mountain, person, water, plant, and ball when available. Save overlays under:

```text
${OUTPUT}/qa_overlays/
```

Do not modify masks during this QA step. Report suspicious mappings, especially:

- beaches mapped to `desert` because VIPSeg provides only `sand`;
- billboard/bulletin-board carriers without readable text still mapped to `text`;
- OCR-accepted `painting_or_poster` instances whose detected text is actually outside the carrier;
- structures that may be broader than the desired `building` definition.

## Optional: Add VSPW

VSPW is a suitable extra source because it contains dense video semantic labels with the same broad 124-category vocabulary. Its standard layout is:

```text
VSPW_ROOT/
├── train.txt / val.txt / test.txt
└── data/<video_id>/origin + mask
```

After obtaining VSPW from its official source:

```bash
VSPW_ROOT=${DATA_BASE}/VSPW_480p
VSPW_OUTPUT=${DATA_BASE}/VSPW_13cls_video

python tools/prepare_vspw_13class.py \
  --vspw-root "${VSPW_ROOT}" \
  --categories-json "${VIPSEG_META}/panoVIPSeg_categories.json" \
  --output-root "${VSPW_OUTPUT}" \
  --mapping configs/vipseg_to_13class.json \
  --workers 16

python tools/check_video_dataset.py --data-root "${VSPW_OUTPUT}" --splits train,val
```

Do not mix splits by moving videos. If combining VIPSeg and VSPW, keep source prefixes in video IDs (for example `vipseg_<id>` and `vspw_<id>`) to prevent collisions.

## Final handoff

Report:

- actual `VIPSEG_ROOT` and `OUTPUT`;
- official archive filename and checksum;
- split video/frame counts;
- per-class pixel and frame occurrence counts;
- full validation result;
- QA overlay paths;
- poster OCR acceptance statistics, audit path, thresholds, and manual QA result;
- any download, license, mapping, or zero-class issue.
