# CodeAgent Task: Train 13-Class Recurrent RVM on Video Clips

## Goal

Train the RVM semantic model on continuous video clips, not independent images. The required tensor flow is:

```text
input  [B,T,3,512,512]
output [B,T,13,512,512]
```

ConvGRU state must propagate between frames inside each clip. State must reset between unrelated clips/videos. Default training uses full-clip BPTT.

## Paths

```bash
REPO=/data/pub1/z00919662/segmentation/rvm-video-semantic-segmentation-13class
DATA_ROOT=/data/pub1/z00919662/dataset/VIPSeg_13cls_video
OUTPUT_DIR=${REPO}/output/rvm_video_semantic_13class
```

## 1. Checkout and environment

```bash
if [ ! -d "${REPO}/.git" ]; then
  git clone https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git "${REPO}"
else
  git -C "${REPO}" pull --ff-only
fi

cd "${REPO}"
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Record:

```bash
python -V
python -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__); print(torch.cuda.is_available(), torch.cuda.device_count())'
nvidia-smi
```

## 2. Dataset gate

Do not train unless this passes:

```bash
python tools/check_video_dataset.py \
  --data-root "${DATA_ROOT}" \
  --splits train,val,test \
  --output-json "${DATA_ROOT}/validation_report_before_training.json"
```

Confirm class 12 has non-zero pixels. Confirm that each training sample directory represents one chronological video and frames sort correctly.

## 3. Model/BPTT smoke test

```bash
cd "${REPO}"
python tools/smoke_video_forward.py --device cuda --time 3 --size 64
```

Acceptance criteria:

- logits are `[1,3,13,64,64]`;
- four recurrent state tensors are returned;
- `gru_gradient_l1 > 0`, proving ConvGRU receives gradients through the clip.

## 4. Choose initialization

Preferred: initialize from the best existing 12-class semantic checkpoint. The loader copies weights by class name and preserves head rows 0–11; only mountain is newly initialized.

Find candidates without changing them:

```bash
find /data/pub1/z00919662 -type f -name best_miou.pth | sort
```

Set the selected path:

```bash
INIT_CHECKPOINT=/absolute/path/to/12class/best_miou.pth
```

If no 12-class checkpoint exists, use the official RVM MobileNetV3 checkpoint:

```bash
mkdir -p "${REPO}/checkpoints"
wget -c \
  https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth \
  -O "${REPO}/checkpoints/rvm_mobilenetv3.pth"
INIT_CHECKPOINT=${REPO}/checkpoints/rvm_mobilenetv3.pth
```

When the official matting checkpoint is used, encoder/decoder tensors are reused but the 13-class semantic projection is initialized for this task.

## 5. Short end-to-end training smoke test

```bash
cd "${REPO}"
CUDA_VISIBLE_DEVICES=0 python train_video_semantic.py \
  --data-root "${DATA_ROOT}" \
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

Acceptance criteria:

- printed classes are exactly IDs 0–12 in the required order;
- loss is finite;
- output logits are 13 channels (the forward smoke test already verifies the exact shape);
- `last.pth`, `best_miou.pth`, and `metrics.csv` are created;
- no mask-ID or image/mask alignment error occurs.

## 6. Full two-GPU training

Default production configuration:

- input: 512×512;
- clip length: 5 consecutive frames;
- frame stride: 1;
- 2 clips/GPU;
- 2 GPUs;
- gradient accumulation: 2;
- effective optimizer batch: 8 clips = 40 frames;
- CE + Dice;
- AMP + SyncBatchNorm;
- full-clip BPTT (`--tbptt-chunk 0`).

Start in tmux:

```bash
tmux new -s rvm_video13
cd "${REPO}"
source .venv/bin/activate

INIT_CHECKPOINT="${INIT_CHECKPOINT}" \
DATA_ROOT="${DATA_ROOT}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/train_2gpu.sh 2>&1 | tee "${OUTPUT_DIR}/train.log"
```

Detach with `Ctrl-b d`. Do not launch a second training process against the same output directory.

### OOM fallback

First reduce clips per GPU and preserve effective batch:

```bash
INIT_CHECKPOINT="${INIT_CHECKPOINT}" \
DATA_ROOT="${DATA_ROOT}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
bash scripts/train_2gpu.sh --batch-size 1 --gradient-accumulation 4
```

If still OOM, use truncated BPTT:

```bash
bash scripts/train_2gpu.sh --batch-size 1 --gradient-accumulation 4 --tbptt-chunk 2
```

`--tbptt-chunk 2` carries state forward but detaches it between chunks; it lowers memory at the cost of shorter temporal gradient range. Do not enable it unless needed.

## 7. Resume safely

```bash
RESUME="${OUTPUT_DIR}/last.pth" \
DATA_ROOT="${DATA_ROOT}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/train_2gpu.sh
```

Resume restores model, optimizer, scheduler, scaler, epoch, and best mIoU. Do not use `--init-checkpoint` together with `--resume`.

## 8. Evaluate test videos

```bash
cd "${REPO}"
CHECKPOINT="${OUTPUT_DIR}/best_miou.pth" \
DATA_ROOT="${DATA_ROOT}" \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/test_video.sh
```

Report:

- mIoU, mean Dice, pixel accuracy;
- per-class IoU, precision, and recall;
- especially `flower`, `food`, `desert`, `ice_or_snow`, `text`, `ball`, and `mountain`;
- number of valid frames and confusion matrix.

## 9. Run direct MP4 inference

```bash
CHECKPOINT="${OUTPUT_DIR}/best_miou.pth" \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/infer_video.sh \
  /absolute/path/input.mp4 \
  "${REPO}/output/inference_mp4" \
  --save-masks
```

Required behavior:

- recurrence is on across frames of the same video;
- state resets at the start of every video;
- state resets on scene cuts (default threshold 0.35);
- state never carries from one file to the next;
- output includes overlay MP4, JSONL class ratios, and optional lossless masks.

Compare one video with recurrence disabled:

```bash
python inference_video_semantic.py \
  --checkpoint "${OUTPUT_DIR}/best_miou.pth" \
  --input /absolute/path/input.mp4 \
  --output-dir "${REPO}/output/inference_no_recurrence" \
  --no-recurrent
```

Visually compare boundary stability, flicker, lag/ghosting, and scene-cut recovery. Do not claim temporal improvement from mIoU alone.

## 10. Final report

Provide:

1. Git commit SHA and environment versions.
2. Dataset validation summary.
3. Initialization checkpoint path and loader report, including copied head classes.
4. Full training command, GPU configuration, and effective batch size.
5. Best epoch and checkpoint path.
6. Validation/test metrics and per-class metrics.
7. Recurrent vs non-recurrent inference videos.
8. Any OOM fallback or TBPTT change.
9. Remaining dataset issues (especially sand/desert and carrier-level text labels; `painting_or_poster` is excluded by default).
