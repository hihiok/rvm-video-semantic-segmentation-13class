# CODEAGENT — RVM13 RANDOM 5 TRAIN LABEL VISUALIZATION

## Goal

The current model inference quality looks poor. Before changing the model, visually inspect the ground-truth labels themselves.

Randomly select exactly 5 training videos from the processed RVM13 VIPSeg dataset and generate label-visualization MP4s.

This task does NOT run model inference.

Each output frame must show:

```text
Original frame | GT color mask | GT mask overlay on original
```

## Paths

Project:

```bash
/mnt/ssd1/z00919662/segmentation/rvm-video-semantic-segmentation-13class
```

Images:

```bash
/mnt/ssd1/z00919662/segmentation/dataset/VIPSeg_13cls_video/images/train
```

Masks:

```bash
/mnt/ssd1/z00919662/segmentation/dataset/VIPSeg_13cls_video/annotations/train
```

Environment:

```bash
/mnt/ssd1/z00919662/anaconda3/envs/rvm_video13_legacy
```

Prepared script:

```bash
tools/visualize_rvm13_train_labels.py
```

Branch:

```text
agent/rvm13-label-vis-v1
```

Script commit:

```text
3f6b169df87c1e5354d0c355aa2d5562c9176047
```

## CPU safety

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

Do not use multiprocessing. Run with `nice -n 10`.

## Checkout

```bash
cd /mnt/ssd1/z00919662/segmentation/rvm-video-semantic-segmentation-13class

git status --short
git fetch origin
```

If working tree is clean:

```bash
git checkout agent/rvm13-label-vis-v1
git pull --ff-only origin agent/rvm13-label-vis-v1
```

If unrelated local changes must be preserved, do not hard reset. Bring in only:

```bash
git checkout 3f6b169df87c1e5354d0c355aa2d5562c9176047 -- tools/visualize_rvm13_train_labels.py
```

## Activate environment

```bash
source /mnt/ssd1/z00919662/anaconda3/bin/activate
conda activate /mnt/ssd1/z00919662/anaconda3/envs/rvm_video13_legacy
unset LD_LIBRARY_PATH
unset LD_PRELOAD
```

## Run

Use deterministic random sampling:

```text
num videos = 5
seed = 20260819
```

Output:

```bash
OUTPUT_DIR=/mnt/ssd1/z00919662/segmentation/rvm-video-semantic-segmentation-13class/output/train_label_visualization_random5_seed20260819
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
```

Run:

```bash
nice -n 10 python tools/visualize_rvm13_train_labels.py \
  --image-root /mnt/ssd1/z00919662/segmentation/dataset/VIPSeg_13cls_video/images/train \
  --mask-root /mnt/ssd1/z00919662/segmentation/dataset/VIPSeg_13cls_video/annotations/train \
  --output-dir "$OUTPUT_DIR" \
  --num-videos 5 \
  --seed 20260819 \
  --fps 25 \
  --overlay-alpha 0.45 \
  --max-frames 0 \
  2>&1 | tee "$OUTPUT_DIR/visualize.log"
```

`--max-frames 0` means all frames of each selected video.

## Expected output

Exactly 5 MP4 files plus:

```text
label_visualization_manifest.csv
visualize.log
```

Each MP4 uses the canonical 13-class palette and includes a legend of classes present.

The script validates:

```text
mask is single-channel
mask IDs are only 0..12 or 255
image/mask resolution matches
video frame resolution is consistent
generated MP4 is readable
```

If any validation fails, do not repair the dataset automatically. Report the exact file/video.

## Static previews

Extract one middle frame from each MP4 into `previews/` using OpenCV. Expect exactly 5 JPGs.

## Visual audit

For each selected video assess:

```text
alignment
semantic correctness
boundary quality
temporal consistency
```

Status each as:

```text
GOOD
QUESTIONABLE
CLEAR LABEL PROBLEM
```

Inspect tail classes carefully if they happen to appear:

```text
flower
food
desert
ice_or_snow
text
ball
mountain
```

Do not claim the full dataset is clean based only on 5 random videos.

## Final report

Create:

```bash
/mnt/ssd1/z00919662/segmentation/rvm-video-semantic-segmentation-13class/TRAIN_RANDOM5_LABEL_VIS_REPORT.md
```

Print:

```text
=== RVM13 RANDOM5 TRAIN LABEL VISUALIZATION ===

GITHUB_BRANCH:
agent/rvm13-label-vis-v1

SEED:
20260819

SELECTED_VIDEOS:
1. ...
2. ...
3. ...
4. ...
5. ...

MP4_COUNT:
5

PREVIEW_JPG_COUNT:
5

OUTPUT_DIR:
/mnt/ssd1/z00919662/segmentation/rvm-video-semantic-segmentation-13class/output/train_label_visualization_random5_seed20260819

CLEAR_LABEL_PROBLEMS_FOUND:
YES / NO

STATUS:
PASS / FAIL
```
