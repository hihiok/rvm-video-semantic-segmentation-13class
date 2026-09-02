# NAS 9-label multi-label scene classifier

Labels: `indoor, outdoor, landscape, sports, food, animal, building, sky, office`.

Product display names: `室内、户外、风景、运动、美食、动物、建筑、蓝天、办公`. `sky` means visible sky of any color; the product name may still display as “蓝天”.

## Model

The model uses the public UltraFace `Mb_Tiny` / slim convolutional stack only. RFB, SSD extra layers, bounding-box heads and face confidence heads are removed. The scene model is:

`Mb_Tiny slim backbone -> AdaptiveAvgPool2d(1) -> Dropout -> Linear(9)`

Default `base_channel=16`, input `224x224`:

- ~0.173M trainable parameters
- ~48.3M MACs / image
- only Conv2d / depthwise Conv2d / BatchNorm / ReLU / GAP / Linear at inference
- deployment output is 9 independent logits; apply sigmoid and per-label thresholds

The topology can be widened later with `base_channel=24` without changing the basic operator set if accuracy is insufficient.

## Training data

The training set uses partial labels. Unknown labels are `-1` and are masked out of BCE loss.

- Places365-Standard 256px: indoor/outdoor from the official IO taxonomy; curated scene categories supervise landscape/sports/office.
- COCO 2017: food and animal.
- Existing project 13-class COCO+ADE semantic masks: building, sky, food.

No missing label is silently converted to negative.

Official validation data are deterministically divided into separate calibration-validation and final-test subsets. Thresholds are calibrated on validation only and final metrics are reported on test.

## Default paths

```text
Raw Places/COCO:
/data/pub1/z00919662/scene_multilabel/datasets_raw

Existing segmentation data:
/data/pub1/z00919662/segmentation/datasets/COCO_ADE_13cls_16x9_640x360

Prepared manifests:
/data/pub1/z00919662/scene_multilabel/nas_9label_partial_gt

Training outputs:
/data/pub1/z00919662/scene_multilabel/ultraface_slim_9label/output
```

## Download data

Places365 256px train is about 24 GB and val about 0.5 GB. COCO train/val plus annotations are about 19 GB. The downloader is resumable.

```bash
bash scripts/download_datasets.sh /data/pub1/z00919662/scene_multilabel/datasets_raw
```

## Prepare + train

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_train.sh
```

CPU use is intentionally limited: default DataLoader workers=4 and OMP/MKL/OpenBLAS threads=4.

Outputs include:

```text
last.pth
best_macro_f1.pth
best_deploy.pth
thresholds.json
test_per_class_calibrated.csv
test_summary.json
ultraface_slim_9label_224.onnx
ultraface_slim_9label_224.json
```

For the product requirement, do not rely on overall accuracy alone. Inspect per-class precision/recall/F1, balanced accuracy and AP, plus macro-F1/macro-balanced-accuracy.
