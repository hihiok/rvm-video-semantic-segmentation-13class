# CodeAgent Full Pipeline: VIPSeg Preparation and 13-Class RVM Video Training

## Mission

Execute the complete pipeline for the repository below, from official dataset download through recurrent video training, evaluation, and MP4 inference.

```text
Repository: https://github.com/hihiok/rvm-video-semantic-segmentation-13class
Code path:  /data/pub1/z00919662/segmentation/rvm-video-semantic-segmentation-13class
Data base:  /data/pub1/z00919662/dataset
```

This is a **video semantic segmentation** task. Do not flatten videos into unrelated frames for training. The model must receive continuous clips, preserve the four RVM ConvGRU states between chronological frames inside each clip, and reset state between unrelated clips/videos.

Required class order is immutable:

```text
0  background
1  sky
2  person
3  plant
4  building
5  flower
6  food
7  water
8  desert
9  ice_or_snow
10 text
11 ball
12 mountain
```

`255` is VOID/padding and must be ignored by the loss and metrics. It is not a class.

## Non-negotiable execution rules

1. Work only under the paths declared below. Do not delete, rename, or overwrite the original VIPSeg archive, extracted source data, or an existing checkpoint.
2. Every stage must be resumable. Reuse an existing clone, archive, extraction, conversion, OCR audit, and training checkpoint when valid.
3. Do not use unofficial dataset mirrors or bypass access controls. VIPSeg is for non-commercial research; retain the official license and README.
4. Never write proxy usernames, passwords, tokens, or cookies into source files, logs, or this repository. Inherit already configured proxy environment variables when present.
5. Do not start full training until label unit tests, full dataset validation, and the recurrent forward/backward smoke test all pass.
6. Never launch two training jobs against the same output directory. If a tmux session or `last.pth` exists, inspect it and resume safely.
7. `painting_or_poster` is excluded from the base text mapping. Add it to `text=10` only after the required OCR instance filter passes.
8. Record actual commands, paths, versions, counts, checksums, thresholds, metrics, warnings, and failures in the final handoff. Do not claim that a long-running stage completed when it is still running.

## Stage 0 — Define paths and inspect existing work

Use these paths unless the host clearly requires a different mounted data root. If a path must change, keep the same directory roles and report the final absolute paths.

```bash
export REPO=/data/pub1/z00919662/segmentation/rvm-video-semantic-segmentation-13class
export DATA_BASE=/data/pub1/z00919662/dataset
export VIPSEG_DOWNLOAD=${DATA_BASE}/VIPSeg_download
export VIPSEG_META=${DATA_BASE}/VIPSeg-Dataset-meta
export VIPSEG_OUTPUT=${DATA_BASE}/VIPSeg_13cls_video
export TRAIN_OUTPUT=${REPO}/output/rvm_video_semantic_13class
export TRAIN_SESSION=rvm_video13

mkdir -p /data/pub1/z00919662/segmentation "${DATA_BASE}" "${VIPSEG_DOWNLOAD}"
```

Before changing anything, inspect:

```bash
test -d /data/pub1/z00919662
df -h "${DATA_BASE}"
nvidia-smi
python -V
```

Report insufficient disk space or unavailable GPUs before a large download or full training. Do not delete other datasets to make space.

## Stage 1 — Clone or update code and official metadata

```bash
if [ ! -d "${REPO}/.git" ]; then
  git clone https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git "${REPO}"
else
  git -C "${REPO}" status --short
  git -C "${REPO}" pull --ff-only
fi

if [ ! -d "${VIPSEG_META}/.git" ]; then
  git clone https://github.com/VIPSeg-Dataset/VIPSeg-Dataset.git "${VIPSEG_META}"
else
  git -C "${VIPSEG_META}" pull --ff-only
fi

cd "${REPO}"
git rev-parse HEAD
test -f tools/prepare_vipseg_13class.py
test -f tools/add_text_posters_with_easyocr.py
test -f train_video_semantic.py
test -f scripts/train_2gpu.sh
```

Do not discard local user changes if `git status --short` is non-empty. Preserve them and report the conflict instead of resetting the repository.

Confirm the official metadata:

```bash
test -f "${VIPSEG_META}/train.txt"
test -f "${VIPSEG_META}/val.txt"
test -f "${VIPSEG_META}/test.txt"
test -f "${VIPSEG_META}/panoVIPSeg_categories.json"
wc -l "${VIPSEG_META}/train.txt" "${VIPSEG_META}/val.txt" "${VIPSEG_META}/test.txt"
```

Expected split sizes are 2,806 train videos, 343 validation videos, and 387 test videos. If the actual official metadata differs, keep the official files unchanged and report the difference.

## Stage 2 — Create the Python environment

Prefer a working CUDA-compatible PyTorch already installed on the server. Do not replace a working CUDA build with a CPU-only build.

```bash
cd "${REPO}"

if [ ! -d .venv ]; then
  python -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install gdown easyocr

python -V
python -c 'import torch, torchvision; print("torch", torch.__version__); print("torchvision", torchvision.__version__); print("cuda", torch.cuda.is_available()); print("gpu_count", torch.cuda.device_count())'
```

If the installed PyTorch cannot use CUDA, install a PyTorch build compatible with the server driver/CUDA environment, then rerun the version command. Record the exact installation command and versions.

## Stage 3 — Download and locate official VIPSeg

Official Google Drive file ID currently used by the project:

```text
1B13QUiE82xf7N6nVHclb4ErN-Zuai-sZ
```

Do not redownload a complete archive. Otherwise run:

```bash
mkdir -p "${VIPSEG_DOWNLOAD}"

if [ ! -s "${VIPSEG_DOWNLOAD}/VIPSeg_release.zip" ]; then
  gdown --fuzzy \
    'https://drive.google.com/file/d/1B13QUiE82xf7N6nVHclb4ErN-Zuai-sZ/view?usp=sharing' \
    -O "${VIPSEG_DOWNLOAD}/VIPSeg_release.zip"
fi

sha256sum "${VIPSEG_DOWNLOAD}/VIPSeg_release.zip" | tee "${VIPSEG_DOWNLOAD}/VIPSeg_release.zip.sha256"
unzip -t "${VIPSEG_DOWNLOAD}/VIPSeg_release.zip"
```

Extract only when no valid extracted root is present:

```bash
find "${VIPSEG_DOWNLOAD}" -maxdepth 5 -type d -name panomasks -print
```

If no result is returned:

```bash
unzip -q "${VIPSEG_DOWNLOAD}/VIPSeg_release.zip" -d "${VIPSEG_DOWNLOAD}"
find "${VIPSEG_DOWNLOAD}" -maxdepth 5 -type d -name panomasks -print
```

Set `VIPSEG_ROOT` to the directory that directly contains both `images/` and `panomasks/`. Example only:

```bash
export VIPSEG_ROOT=${VIPSEG_DOWNLOAD}/VIPSeg_720P
test -d "${VIPSEG_ROOT}/images"
test -d "${VIPSEG_ROOT}/panomasks"
```

If Google Drive is blocked, record the exact error and preserve partial downloads. Do not switch to an unofficial mirror.

## Stage 4 — Verify source label decoding before full conversion

VIPSeg masks are panoptic IDs, not ordinary semantic class-index masks:

- raw `0` is VOID and becomes output `255`;
- stuff regions store the source category ID;
- thing regions store `category_id × 100 + instance_id`;
- annotated non-target categories become `background=0`.

Run the mandatory unit tests:

```bash
cd "${REPO}"
source .venv/bin/activate
PYTHONPATH=. pytest -q tests/test_label_mapping.py
```

Stop before full conversion if any unit test fails.

## Stage 5 — Convert VIPSeg to continuous 13-class video folders

Use the strict mapping. It includes mountain and excludes `painting_or_poster` until OCR approval.

```bash
cd "${REPO}"
source .venv/bin/activate

python tools/prepare_vipseg_13class.py \
  --vipseg-root "${VIPSEG_ROOT}" \
  --metadata-root "${VIPSEG_META}" \
  --output-root "${VIPSEG_OUTPUT}" \
  --mapping configs/vipseg_to_13class.json \
  --copy-mode hardlink \
  --workers 16
```

The converter is resumable and must not use `--overwrite` during a normal resume. Hardlink mode automatically falls back to copying when required.

Expected layout:

```text
VIPSeg_13cls_video/
├── images/
│   ├── train/<video_id>/<frame>.jpg
│   ├── val/<video_id>/<frame>.jpg
│   └── test/<video_id>/<frame>.jpg
├── annotations/
│   ├── train/<video_id>/<frame>.png
│   ├── val/<video_id>/<frame>.png
│   └── test/<video_id>/<frame>.png
├── class_mapping.json
└── dataset_summary.json
```

Frames inside each video must remain chronologically ordered and must not be randomly renamed or moved across splits.

## Stage 6 — OCR-filter `painting_or_poster` and add accepted carriers to text

The strict mapping already maps reliable text carriers such as billboard/bulletin-board according to the repository configuration. It deliberately excludes the mixed `painting_or_poster` source class.

Run EasyOCR per poster/painting instance. A carrier is assigned completely to `text=10` only when detected text passes all gates:

- confidence at least `0.20`;
- at least `2` effective alphanumeric characters;
- OCR polygon overlap with the carrier at least `0.50`;
- accepted text area / carrier area at least `0.001`.

Use GPU when available:

```bash
cd "${REPO}"
source .venv/bin/activate

python tools/add_text_posters_with_easyocr.py \
  --vipseg-root "${VIPSEG_ROOT}" \
  --metadata-root "${VIPSEG_META}" \
  --converted-root "${VIPSEG_OUTPUT}" \
  --splits train,val,test \
  --languages ch_sim,en \
  --gpu \
  --min-confidence 0.20 \
  --min-characters 2 \
  --min-polygon-overlap 0.50 \
  --min-text-area-ratio 0.001 \
  --qa-limit-per-decision 200
```

If CUDA is unavailable for EasyOCR, replace `--gpu` with `--no-gpu`. The audit is resumable; rerun the same command after interruption. Do not use `--overwrite-audit` unless thresholds are intentionally changed and the whole decision set must be rebuilt.

Required outputs:

```text
${VIPSEG_OUTPUT}/painting_or_poster_ocr_audit.jsonl
${VIPSEG_OUTPUT}/painting_or_poster_ocr_summary.json
${VIPSEG_OUTPUT}/qa_poster_ocr/accepted/
${VIPSEG_OUTPUT}/qa_poster_ocr/rejected/
```

Manually inspect at least 100 accepted and 100 rejected crops, or every crop if fewer exist. Accepted text must lie on the annotated carrier, not on nearby subtitles, signs, UI, or watermarks. Precision is more important than acceptance rate. Record the inspection count and error examples.

## Stage 7 — Validate the complete converted dataset

```bash
cd "${REPO}"
source .venv/bin/activate

python tools/check_video_dataset.py \
  --data-root "${VIPSEG_OUTPUT}" \
  --splits train,val,test \
  --output-json "${VIPSEG_OUTPUT}/validation_report.json"
```

Training is allowed only if all of the following are true:

1. Every image has one same-stem mask in the same split/video directory.
2. Image and mask resolutions match.
3. Mask values are only `0..12` and optional `255`.
4. Train and validation contain multi-frame chronological videos.
5. `mountain=12` has non-zero pixels and occurs in non-zero frames.
6. `dataset_summary.json`, `validation_report.json`, and both OCR audit files exist.
7. Original VIPSeg files are unchanged.

Print and save a table with video count, frame count, pixel count per class, and frames containing each class. Explicitly flag zero or extremely rare classes. Keep these known risks in the report:

- VIPSeg `sand → desert` is a weak proxy and includes beaches;
- `text` is carrier-level, not character-stroke segmentation;
- `ball` is the ball itself and is often a very small target;
- OCR false positives can come from subtitles or nearby text outside the carrier.

## Stage 8 — Optional VSPW supplement

This stage is optional and must not block the VIPSeg training run. Use it only if the official VSPW dataset is already available or can be obtained from its official release.

```bash
export VSPW_ROOT=${DATA_BASE}/VSPW_480p
export VSPW_OUTPUT=${DATA_BASE}/VSPW_13cls_video

python tools/prepare_vspw_13class.py \
  --vspw-root "${VSPW_ROOT}" \
  --categories-json "${VIPSEG_META}/panoVIPSeg_categories.json" \
  --output-root "${VSPW_OUTPUT}" \
  --mapping configs/vipseg_to_13class.json \
  --workers 16

python tools/check_video_dataset.py \
  --data-root "${VSPW_OUTPUT}" \
  --splits train,val \
  --output-json "${VSPW_OUTPUT}/validation_report.json"
```

Do not merge VSPW into VIPSeg by simply copying folders with colliding video IDs. The current production command below trains on `VIPSEG_OUTPUT`. If multi-source training is requested later, add an explicit multi-root dataset loader or create a verified merged root with source-prefixed video IDs and preserved train/val separation.

## Stage 9 — Verify recurrent forward/backward behavior

Full training requires CUDA and must pass this gate:

```bash
cd "${REPO}"
source .venv/bin/activate
python tools/smoke_video_forward.py --device cuda --time 3 --size 64
```

Acceptance criteria:

- input represents `[1,3,3,64,64]`;
- logits are exactly `[1,3,13,64,64]`;
- four recurrent state tensors are returned;
- `gru_gradient_l1 > 0`, proving the recurrent path receives gradient through time.

Stop if the output is single-frame, has 12 channels, or the ConvGRU gradient is zero.

## Stage 10 — Select and inspect initialization weights

Preferred initialization is the best existing 12-class semantic checkpoint. Locate candidates without modifying them:

```bash
find /data/pub1/z00919662 -type f -name best_miou.pth | sort
```

For each plausible 12-class candidate, inspect metadata:

```bash
python - <<'PY'
from pathlib import Path
import torch

for path in sorted(Path('/data/pub1/z00919662').rglob('best_miou.pth')):
    try:
        checkpoint = torch.load(path, map_location='cpu')
        print(path)
        print('  classes:', checkpoint.get('class_names'))
        print('  epoch:', checkpoint.get('epoch'), 'best_miou:', checkpoint.get('best_miou'))
    except Exception as error:
        print(path, 'ERROR', error)
PY
```

Choose the checkpoint whose class order is exactly the earlier 12-class order `background..ball`. Set:

```bash
export INIT_CHECKPOINT=/absolute/path/to/12class/best_miou.pth
test -f "${INIT_CHECKPOINT}"
```

The 13-class loader must copy matching head rows 0–11 by class name and newly initialize only `mountain=12`. Save the loader report in the training log.

If no valid 12-class semantic checkpoint exists, use the official RVM MobileNetV3 checkpoint:

```bash
mkdir -p "${REPO}/checkpoints"
wget -c \
  https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth \
  -O "${REPO}/checkpoints/rvm_mobilenetv3.pth"
export INIT_CHECKPOINT=${REPO}/checkpoints/rvm_mobilenetv3.pth
```

With the matting checkpoint, reuse compatible encoder/decoder tensors and initialize the entire 13-class semantic projection. Report which initialization route was used.

## Stage 11 — Run a short end-to-end training smoke test

```bash
cd "${REPO}"
source .venv/bin/activate
rm -f "${REPO}/output/smoke_video13/last.pth" "${REPO}/output/smoke_video13/best_miou.pth"

CUDA_VISIBLE_DEVICES=0 python train_video_semantic.py \
  --data-root "${VIPSEG_OUTPUT}" \
  --output-dir "${REPO}/output/smoke_video13" \
  --init-checkpoint "${INIT_CHECKPOINT}" \
  --input-size 512 \
  --clip-length 3 \
  --frame-stride 1 \
  --train-clip-step 10 \
  --epochs 1 \
  --batch-size 1 \
  --workers 2 \
  --max-train-clips 20 \
  --max-val-clips 10 \
  --amp
```

The explicit removal above is limited to disposable smoke-test checkpoints only. Do not remove production checkpoints.

Acceptance criteria:

- printed class IDs/names are exactly 0–12 in the required order;
- loss and metrics are finite;
- outputs have 13 channels;
- image/mask alignment and label IDs produce no error;
- `last.pth`, `best_miou.pth`, and `metrics.csv` are created under `output/smoke_video13`.

## Stage 12 — Launch full two-GPU training

Production defaults in `scripts/train_2gpu.sh`:

```text
input size              512×512
clip length             5 consecutive frames
frame stride            1
train clip step         5
clips per GPU           2
GPU count               2
gradient accumulation   2
effective batch         8 clips = 40 frames/optimizer step
loss                    Cross-Entropy + Dice
precision               AMP
normalization           SyncBatchNorm
temporal gradient       full-clip BPTT
```

Create the log directory before starting `tee`:

```bash
mkdir -p "${TRAIN_OUTPUT}"
```

If a session already exists, do not start a duplicate:

```bash
tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null && tmux capture-pane -pt "${TRAIN_SESSION}" -S -80
```

If there is no active training session, launch detached. The shell script automatically resumes from `${TRAIN_OUTPUT}/last.pth`; otherwise it uses `INIT_CHECKPOINT`.

```bash
cd "${REPO}"
source .venv/bin/activate

tmux new-session -d -s "${TRAIN_SESSION}" \
  "cd '${REPO}' && source .venv/bin/activate && INIT_CHECKPOINT='${INIT_CHECKPOINT}' DATA_ROOT='${VIPSEG_OUTPUT}' OUTPUT_DIR='${TRAIN_OUTPUT}' CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_2gpu.sh 2>&1 | tee '${TRAIN_OUTPUT}/train.log'"

tmux capture-pane -pt "${TRAIN_SESSION}" -S -80
```

Monitor without starting another process:

```bash
tmux capture-pane -pt "${TRAIN_SESSION}" -S -120
tail -n 100 "${TRAIN_OUTPUT}/train.log"
nvidia-smi
```

Required production outputs:

```text
${TRAIN_OUTPUT}/last.pth
${TRAIN_OUTPUT}/best_miou.pth
${TRAIN_OUTPUT}/metrics.csv
${TRAIN_OUTPUT}/train.log
```

### OOM fallback order

First reduce clips per GPU while preserving the effective optimizer batch:

```bash
INIT_CHECKPOINT="${INIT_CHECKPOINT}" \
DATA_ROOT="${VIPSEG_OUTPUT}" \
OUTPUT_DIR="${TRAIN_OUTPUT}" \
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/train_2gpu.sh --batch-size 1 --gradient-accumulation 4
```

Only if this still fails, shorten temporal gradient range:

```bash
bash scripts/train_2gpu.sh \
  --batch-size 1 \
  --gradient-accumulation 4 \
  --tbptt-chunk 2
```

`--tbptt-chunk 2` carries recurrent states forward but detaches them between chunks. Report its use because it changes temporal training behavior. Do not silently reduce input size or clip length.

### Explicit resume command

Use this after interruption when no training process is active:

```bash
RESUME="${TRAIN_OUTPUT}/last.pth" \
DATA_ROOT="${VIPSEG_OUTPUT}" \
OUTPUT_DIR="${TRAIN_OUTPUT}" \
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/train_2gpu.sh
```

Never pass `INIT_CHECKPOINT` and `RESUME` together. Resume must restore model, optimizer, scheduler, scaler, epoch, and best mIoU.

If the CodeAgent execution window ends while training is active, leave the tmux job running and report the session name, current epoch/iteration, latest metrics, log path, checkpoint status, and exact monitoring/resume commands.

## Stage 13 — Evaluate the best checkpoint

Run after `best_miou.pth` exists:

```bash
cd "${REPO}"
source .venv/bin/activate

CHECKPOINT="${TRAIN_OUTPUT}/best_miou.pth" \
DATA_ROOT="${VIPSEG_OUTPUT}" \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/test_video.sh \
  --output-json "${TRAIN_OUTPUT}/test_metrics.json"
```

Report:

- mIoU, mean Dice, and pixel accuracy;
- per-class IoU, precision, and recall;
- confusion matrix and number of evaluated frames;
- rare-class results for flower, food, desert, ice/snow, text, ball, and mountain.

Interpretation reminder: per-class recall mainly exposes missed target pixels; precision exposes false-positive target pixels; IoU penalizes both false positives and false negatives. Do not use pixel accuracy alone because background and large classes can dominate it.

## Stage 14 — Run recurrent MP4 inference and compare recurrence

Choose at least one representative business MP4 with scene changes and several target categories:

```bash
export INPUT_VIDEO=/absolute/path/to/input.mp4
test -f "${INPUT_VIDEO}"

CHECKPOINT="${TRAIN_OUTPUT}/best_miou.pth" \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/infer_video.sh \
  "${INPUT_VIDEO}" \
  "${TRAIN_OUTPUT}/inference_recurrent" \
  --save-masks
```

Required recurrent behavior:

- retain state across chronological frames in the same video;
- reset state at the start of every video;
- reset state on detected scene cuts, default threshold `0.35`;
- never carry state between files.

Run a non-recurrent comparison:

```bash
CUDA_VISIBLE_DEVICES=0 python inference_video_semantic.py \
  --checkpoint "${TRAIN_OUTPUT}/best_miou.pth" \
  --input "${INPUT_VIDEO}" \
  --output-dir "${TRAIN_OUTPUT}/inference_no_recurrence" \
  --no-recurrent \
  --save-masks
```

Compare overlay videos for boundary stability, flicker, temporal lag/ghosting, and recovery after scene cuts. The inference output must include overlay MP4, per-frame JSONL class ratios, and optional lossless class-index PNG masks.

## Final CodeAgent handoff

Write a concise final report containing all of the following:

1. Repository commit SHA and whether the worktree had pre-existing changes.
2. Python, PyTorch, torchvision, CUDA, GPU, and EasyOCR versions.
3. Actual paths for `VIPSEG_ROOT`, converted data, initialization checkpoint, and training output.
4. Official archive filename, byte size, SHA-256, and license note.
5. Dataset split video/frame counts and per-class pixel/frame occurrence counts.
6. Label unit-test and full validation results.
7. Poster OCR thresholds, total/accepted/rejected instance counts, audit paths, and manual QA findings.
8. Recurrent smoke-test tensor shapes and `gru_gradient_l1`.
9. Initialization loader report: which tensors/classes were copied and which head row was newly initialized.
10. Full training command, GPU allocation, effective batch size, current/best epoch, and checkpoint/log paths.
11. Best validation/test metrics and the complete per-class table.
12. Recurrent and non-recurrent inference output paths plus visual comparison.
13. Any OOM fallback, TBPTT change, interrupted stage, download restriction, zero/rare class, or known label-quality issue.

The task is complete only when the dataset gates and training smoke test pass and full training has either completed with evaluation/inference outputs or is verifiably running in tmux with a valid log and resumable checkpoint path.
