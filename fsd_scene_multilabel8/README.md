# FSD 8-label multi-label scene classification

Labels (fixed order):

1. `night` 夜景
2. `indoor` 室内
3. `rain_snow` 雨/雪
4. `office` 办公场景
5. `outdoor` 户外
6. `landscape` 风景
7. `sports` 运动
8. `objective_image` 客观图（电脑合成 pattern、解析度卡、test chart 等）

This package is a staging/runner package for an existing FSD/UltraFace repository. It does **not** replace the FSD model implementation. Training imports the existing:

```python
create_Mb_Tiny_RFB_fd_3_scene_noRFB(2, 8)
YUV444TrainAugmentation_scene
YUV444TestTransform_scene
```

and trains its scene logits using masked `BCEWithLogitsLoss` for multi-label output.

Labels in manifests are `1/0/-1` = positive/negative/unknown. Unknown labels are excluded from the loss and metrics.

Read-only source datasets:

- `/data/pub1/z00919662/segmentation/datasets/places365`
- `/data/pub1/z00919662/segmentation/datasets/coco`
- `/data/pub1/z00919662/segmentation/datasets/COCO_ADE_13cls_16x9_640x360`
- `/data/pub1/z00919662/dataset/10_scenes`
- key objective-image source: `/data/pub1/z00919662/dataset/10_scenes/train/Computer_synthesized`

No source image or existing label file is modified. `prepare_manifest.py` creates new JSONL manifests containing absolute paths to the original images.

Primary source supervision:

- Places365: indoor/outdoor/office/landscape/sports, plus conservative snow scene labels.
- 10_scenes: night/rain-snow/office/indoor/outdoor/landscape/sports/objective according to explicit folder aliases.
- COCO+ADE 13-class masks: snow and conservative natural-landscape evidence.
- COCO photos: safe negatives for `objective_image` only.

`Computer_synthesized` is mapped to `objective_image=1` and all seven photographic scene labels `=0`.

Training defaults intentionally follow the supplied FSD scene launcher: input size 240 (FSD config resolves the actual H/W), batch 24, SGD lr 1e-2, milestones 95/150, 200 epochs, workers 4.
