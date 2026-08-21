# CodeAgent task: build filtered VSPW 13-class train/val dataset

## Required outcome

Build the final dataset at exactly:

```text
/data/pub1/z00919662/segmentation/datasets/VSPW_13cls
```

Use the **official VSPW 480p `origin` images and matching `mask` files**. Do not use VIPSeg images. Convert labels to the fixed 13-class taxonomy and omit every frame whose mapped mask has no foreground target pixel.

The filter is exact:

- keep: mapped mask contains at least one pixel with ID `1..12`;
- drop: mapped mask contains only `background=0` and/or `ignore=255`;
- `background=0` alone does not make a frame useful.

Do not create or edit Python, mapping, or training code. The implementation and tests are already on branch `agent/vspw-13class-filtered-dataset-v1`; only check out and run them.

## 0. Network/proxy setup

Run these commands before any Git or download command:

```bash
export http_proxy="http://z00919662:Zzhs12345%21@proxyhk.huawei.com:8080"
export https_proxy="http://z00919662:Zzhs12345%21@proxyhk.huawei.com:8080"
export https_proxy="http://z00919662:Zzhs12345%21@proxyhk.huawei.com:8080"
export HTTPS_PROXY="http://z00919662:Zzhs12345%21@proxyhk.huawei.com:8080"
git config --global http.proxy http://z00919662:Zzhs12345%21@proxy.server.com:8080
git config --global https.proxy http://z00919662:Zzhs12345%21@proxy.server.com:8080
git config --global https.proxy https://z00919662:Zzhs12345%21@proxyhk.huawei.com:8080
git config --global http.proxy http://z00919662:Zzhs12345%21@proxyhk.huawei.com:8080
git config --global http.sslVerify false
```

Do not copy proxy credentials into the final report.

## 1. Check out the prepared code

Repository:

```text
https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git
```

Use branch:

```text
agent/vspw-13class-filtered-dataset-v1
```

If an existing clone has local modifications, do not overwrite them; create a separate clone/worktree. Then run:

```bash
git fetch origin agent/vspw-13class-filtered-dataset-v1
git switch agent/vspw-13class-filtered-dataset-v1
python -m pip install -r requirements.txt
python -m pytest -q tests/test_prepare_vspw_13class.py tests/test_label_mapping.py
```

Stop if the tests fail. Do not patch around a failure.

## 2. Resolve the complete official VSPW source

Known extracted label location:

```text
/data/pub1/z00919662/segmentation/datasets/VSPW_480p_labels
```

Known report:

```text
/data/pub1/z00919662/segmentation/datasets/VSPW_480p_labels/VSPW_LABEL_DOWNLOAD_REPORT.md
```

The converter requires one root containing all of these:

```text
train.txt
val.txt
data/<video_id>/origin/<frame image>
data/<video_id>/mask/<same frame>.png
```

First inspect the known label location. If `origin` directories are already present, use that root directly. Otherwise locate the previously downloaded 14-GB official archive using the report and files below `/data/pub1/z00919662/segmentation/datasets/`, then extract its full contents to:

```text
/data/pub1/z00919662/segmentation/datasets/VSPW_480p_source
```

Support the archive's actual format (`tar`, `tar.gz`, or `zip`). Do not redownload it when the existing archive is valid. After extraction, set `VSPW_ROOT` to the directory that directly contains `train.txt`, `val.txt`, and `data/`.

Before conversion, verify:

```bash
test -f "$VSPW_ROOT/train.txt"
test -f "$VSPW_ROOT/val.txt"
test -d "$VSPW_ROOT/data"
find "$VSPW_ROOT/data" -type d -name origin -print -quit
find "$VSPW_ROOT/data" -type d -name mask -print -quit
```

Find the already available category metadata from the VIPSeg metadata checkout (not from the VIPSeg images):

```bash
CATEGORIES_JSON=$(find /data/pub1/z00919662/segmentation -type f -name panoVIPSeg_categories.json -print -quit)
test -n "$CATEGORIES_JSON"
test -f "$CATEGORIES_JSON"
```

## 3. Protect any existing output

Set:

```bash
OUTPUT_ROOT=/data/pub1/z00919662/segmentation/datasets/VSPW_13cls
```

If `$OUTPUT_ROOT/_SUCCESS` already exists, audit the existing dataset with step 5 before deciding whether a rebuild is necessary. If the directory is incomplete, move it to a timestamped sibling backup; do not delete it. The converter intentionally refuses to merge with stale output unless `--overwrite` is explicitly supplied.

## 4. Convert and filter

From the checked-out repository, run:

```bash
python tools/prepare_vspw_13class.py \
  --vspw-root "$VSPW_ROOT" \
  --categories-json "$CATEGORIES_JSON" \
  --output-root /data/pub1/z00919662/segmentation/datasets/VSPW_13cls \
  --splits train,val \
  --copy-mode hardlink \
  --minimum-target-pixels 1 \
  --workers 16
```

`hardlink` saves disk space while keeping the output image files valid if the extracted source tree is later removed. The converter falls back to a real copy only when source and output are on different filesystems. Converted masks are newly written one-channel PNGs.

Dropped frames break temporal continuity. The converter therefore starts a new `segment_NNNN` directory after each dropped run, so the video loader cannot treat frames across a removed interval as adjacent.

## 5. Full validation

Run a full scan, not a sample:

```bash
python tools/check_video_dataset.py \
  --data-root /data/pub1/z00919662/segmentation/datasets/VSPW_13cls \
  --splits train,val \
  --max-frames 0 \
  --require-target \
  --output-json /data/pub1/z00919662/segmentation/datasets/VSPW_13cls/validation_report.json
```

Then verify:

```bash
test -f /data/pub1/z00919662/segmentation/datasets/VSPW_13cls/_SUCCESS
test -f /data/pub1/z00919662/segmentation/datasets/VSPW_13cls/dataset_summary.json
test -f /data/pub1/z00919662/segmentation/datasets/VSPW_13cls/validation_report.json
```

Acceptance criteria:

1. Every retained image has exactly one same-resolution mask at the mirrored relative path.
2. Every mask is one channel and contains only `0..12` plus optional `255`.
3. Every retained mask contains at least one target ID `1..12`.
4. `kept_frames + dropped_no_target_frames == source_frames` for both train and val.
5. No source video ID occurs in both train and val.
6. Images and masks remain at the official raw resolution; do not resize them during preparation.
7. The output contains `_SUCCESS` only after conversion completed.

## 6. Final report

Write:

```text
/data/pub1/z00919662/segmentation/datasets/VSPW_13cls/VSPW_13CLASS_PREPARATION_REPORT.md
```

Report, separately for train and val:

- source videos and videos with retained frames;
- source frames, kept frames, dropped no-target frames, and keep rate;
- number of temporal segments;
- frame counts containing each class ID `1..12`;
- image/mask resolution distribution;
- invalid/mismatched/corrupted file counts;
- exact source root, output root, branch, and commit SHA;
- test and validation command outcomes.

Do not claim success unless every acceptance criterion passes. Do not start training in this task.

## Final directory shape

```text
/data/pub1/z00919662/segmentation/datasets/VSPW_13cls/
├── _SUCCESS
├── dataset_summary.json
├── validation_report.json
├── VSPW_13CLASS_PREPARATION_REPORT.md
├── images/
│   ├── train/<video_id>/segment_0000/<frame>.jpg
│   └── val/<video_id>/segment_0000/<frame>.jpg
├── annotations/
│   ├── train/<video_id>/segment_0000/<frame>.png
│   └── val/<video_id>/segment_0000/<frame>.png
└── metadata/
    ├── class_mapping.json
    ├── frame_filter_train.tsv
    └── frame_filter_val.tsv
```
