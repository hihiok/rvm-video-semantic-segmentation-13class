# RVM Video Semantic Segmentation — 13 Classes

This project adapts [Robust Video Matting (RVM)](https://github.com/PeterL1n/RobustVideoMatting) into a recurrent **video semantic segmentation** model.

Unlike the earlier single-frame version, training samples are continuous clips shaped `[B,T,3,H,W]`. RVM's four ConvGRU states propagate in chronological order inside each clip and gradients flow through time (BPTT). State is reset between unrelated clips/videos. Video inference keeps state across frames and resets it at detected scene cuts.

## Fixed class mapping

| ID | Class |
|---:|---|
| 0 | background |
| 1 | sky |
| 2 | person |
| 3 | plant |
| 4 | building |
| 5 | flower |
| 6 | food |
| 7 | water |
| 8 | desert |
| 9 | ice_or_snow |
| 10 | text |
| 11 | ball |
| 12 | mountain |

`255` is reserved for VOID/padding and is ignored by CE, Dice, and metrics. It is not a semantic class.

## Network

`MobileNetV3/ResNet50 encoder → LR-ASPP → four-stage recurrent decoder (ConvGRU) → 13-channel 1×1 projection`

- Input: `[B,T,3,H,W]`
- Logits: `[B,T,13,H,W]`
- Recurrent outputs: four hidden states after the last frame
- Loss: multi-class Cross-Entropy + Dice
- Default: full-clip BPTT; optional truncated BPTT via `--tbptt-chunk`

## Dataset layout

```text
DATA_ROOT/
├── images/
│   ├── train/<video_id>/<frame>.jpg
│   ├── val/<video_id>/<frame>.jpg
│   └── test/<video_id>/<frame>.jpg
└── annotations/
    ├── train/<video_id>/<frame>.png
    ├── val/<video_id>/<frame>.png
    └── test/<video_id>/<frame>.png
```

Frames are sorted lexicographically inside each video, so filenames must be zero-padded or naturally sortable in chronological order.

## VIPSeg conversion

The official VIPSeg masks are panoptic IDs, not ordinary semantic IDs. The converter correctly handles:

- `0`: VOID → output `255`
- stuff: raw category ID
- thing: `category_id × 100 + instance_id`
- valid non-target categories → output `background=0`

Run:

```bash
python tools/prepare_vipseg_13class.py \
  --vipseg-root /path/to/VIPSeg_720P \
  --metadata-root /path/to/VIPSeg-Dataset \
  --output-root /data/pub1/z00919662/dataset/VIPSeg_13cls_video \
  --copy-mode hardlink \
  --workers 16

python tools/check_video_dataset.py \
  --data-root /data/pub1/z00919662/dataset/VIPSeg_13cls_video
```

Mapping details are editable in `configs/vipseg_to_13class.json`. Important caveats:

- VIPSeg has `sand`, not a true desert label. The default maps `sand → desert`, so beaches are included.
- Strict text mapping uses only billboard/bulletin-board carriers. `painting_or_poster` is excluded by default because the source class also contains text-free paintings. A broad optional mapping is provided in `configs/vipseg_to_13class_broad_text.json`.

After strict conversion, OCR-filter `painting_or_poster` and add only accepted carrier instances:

```bash
python -m pip install easyocr
python tools/add_text_posters_with_easyocr.py \
  --vipseg-root /path/to/VIPSeg_720P \
  --metadata-root /path/to/VIPSeg-Dataset \
  --converted-root /data/pub1/z00919662/dataset/VIPSeg_13cls_video \
  --languages ch_sim,en \
  --gpu
```

Each EasyOCR polygon must pass confidence, effective-character-count, and carrier-overlap gates. The script then assigns the entire accepted poster instance to `text=10` and writes a resumable JSONL audit plus accepted/rejected QA crops.
- Ball uses only the ball itself; nets and backboards are excluded.

## VSPW supplement

VSPW is the most useful additional dataset for these broad scene classes. A converter is included:

```bash
python tools/prepare_vspw_13class.py \
  --vspw-root /path/to/VSPW_480p \
  --categories-json /path/to/panoVIPSeg_categories.json \
  --output-root /data/pub1/z00919662/dataset/VSPW_13cls_video
```

Cityscapes-VPS and KITTI-STEP are real video segmentation datasets, but are driving-scene focused and have poor coverage of food, flower, desert, ice/snow, text, ball, and mountain. They are not recommended as the main training source for this taxonomy. Static COCO-Stuff/ADE20K data remains useful for initialization, but cannot train temporal recurrence by itself.

## Training

### COCO+ADE13 initialization and VSPW mixed video fine-tuning

`train_vspw_mixed.py` starts from an already-trained **13-class** image model and
jointly trains on chronological VSPW clips and independent COCO+ADE13 images.
Static images are represented as one-frame clips, which resets recurrent state
between unrelated photographs. Real VSPW clips propagate all four ConvGRU
states and use full-clip BPTT by default.

The default curriculum runs as one resumable training job:

| Stage | Epochs | VSPW clip length | Video/static batch ratio |
|---|---:|---:|---:|
| Mixed domain adaptation | 20 | 5 | 1:1 |
| Temporal fine-tuning | 60 | 8 | 2:1 |

Each epoch reports separate VSPW and static validation mIoU. The selected
`best_balanced.pth` / `best_miou.pth` maximizes the weighted mean of both mIoUs
while requiring static mIoU to remain within 0.03 of its initial baseline.
`best_video_miou.pth` and `best_static_miou.pth` are also saved independently.
`stable_gt_flip_rate` is an optical-flow-free diagnostic: prediction changes are
counted only at pixels whose valid ground-truth class remains unchanged between
adjacent frames. It is not used as a training loss.

The launcher validates converted VSPW masks, checks train/validation video
leakage, audits both sources' 13-class coverage, selects available GPUs, and
uses the confirmed original static dataset location:

```bash
STATIC_ROOT=/data/pub1/z00919662/segmentation/RobustVideoMatting-semantic-13class-512/data \
VSPW_ROOT=/data/pub1/z00919662/segmentation/datasets/VSPW_13cls \
INIT_CHECKPOINT=/data/pub1/z00919662/segmentation/RobustVideoMatting-semantic-13class-512/output/rvm_semantic_13class_512_2gpu/best_miou.pth \
bash scripts/train_vspw_mixed.sh
```

`STATIC_ROOT` defaults to the confirmed path shown above. Explicitly set
`AUTO_DISCOVER_STATIC_ROOT=1` only if that dataset was moved and must be located
from checkpoint/project metadata.
Set `CUDA_VISIBLE_DEVICES=3` for one GPU, or `CUDA_VISIBLE_DEVICES=3,4` for two.
Resume with `RESUME=/path/to/last.pth bash scripts/train_vspw_mixed.sh`.
Both `train` and `val` static image/mask splits are required; the script will not
silently substitute VSPW validation or train without replay data.

Quick model check:

```bash
python tools/smoke_video_forward.py --device cuda
```

Two GPUs:

```bash
INIT_CHECKPOINT=/path/to/old_12class_best_miou.pth \
DATA_ROOT=/data/pub1/z00919662/dataset/VIPSeg_13cls_video \
bash scripts/train_2gpu.sh
```

When a 12-class checkpoint is supplied, weights for matching class names are copied into the 13-class head; only `mountain` is newly initialized.

Resume:

```bash
RESUME=/path/to/output/last.pth bash scripts/train_2gpu.sh
```

Evaluate and infer:

```bash
CHECKPOINT=/path/to/best_miou.pth bash scripts/test_video.sh

CHECKPOINT=/path/to/best_miou.pth \
bash scripts/infer_video.sh input.mp4 output_video
```

Inference writes an overlay MP4 plus JSONL per-frame class ratios. `--save-masks` optionally saves lossless class-index PNGs.

## CodeAgent instructions

- `CODEAGENT_DATA_PREPARATION.md`: download/convert/validate VIPSeg (and optional VSPW)
- `CODEAGENT_TRAINING.md`: environment, smoke tests, two-GPU training, resume, test, and video inference

## Licensing

The RVM-derived code remains under GPL-3.0 (see `LICENSE`). VIPSeg data is released by its authors for non-commercial research only; obtain it from the official source and follow its terms.
