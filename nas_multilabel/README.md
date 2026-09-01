# NAS 9-label MobileCLIP2-S0 zero-shot benchmark

This folder evaluates `MobileCLIP2-S0` as a lightweight zero-shot multi-label photo tagger for:

`indoor, outdoor, landscape, sports, food, animal, building, sky, office`

Product display names are `室内、户外、风景、运动、美食、动物、建筑、蓝天、办公`.  Internally `blue sky` is intentionally represented as `sky`: visible sky is positive regardless of color.

## Why partial labels

The probe set is assembled from existing datasets with different annotation taxonomies.  Unknown labels are stored as `-1` and excluded from metrics; they are never silently converted to negatives.

- Places365 contributes scene GT: indoor/outdoor/landscape/sports/office.
- COCO instance annotations contribute food/animal, with sports-object positives as supplemental evidence.
- Existing project 13-class semantic masks contribute sky/building, plus food when available.

The builder requires both known positives and known negatives for all nine labels before inference is allowed to start.

## Build

```bash
cd nas_multilabel
python build_probe_dataset.py \
  --places-root /path/to/Places365 \
  --coco-root /path/to/COCO \
  --seg-root /data/pub1/z00919662/segmentation/datasets/COCO_ADE_13cls_16x9_640x360 \
  --output-dir /data/pub1/z00919662/segmentation/nas_scene_tagging/mobileclip2_s0_probe
```

Default target is 150 positives + 150 negatives per label; the benchmark refuses to proceed if any label has fewer than 50 positives or 50 negatives.

## Zero-shot evaluation

```bash
python run_mobileclip2_zeroshot.py \
  --manifest /data/pub1/z00919662/segmentation/nas_scene_tagging/mobileclip2_s0_probe/manifest.jsonl \
  --output-dir /data/pub1/z00919662/segmentation/nas_scene_tagging/mobileclip2_s0_results \
  --device cuda \
  --amp fp16
```

Primary evaluation is deliberately strict zero-shot:

- official `MobileCLIP2-S0`, pretrained `dfndr2b`
- 9 independent label scores, never a 9-way softmax
- fixed positive/contrast prompt ensembles
- fixed threshold `0.5`
- unknown GT masked

`oracle_threshold_diagnostic.csv` sweeps thresholds on the test GT only to diagnose calibration.  It must not be reported as zero-shot performance.

## Outputs

Dataset build:

- `candidate_summary.json`
- `dataset_summary.json`
- `manifest.jsonl`
- `labels.csv`
- `images/` symlinks by default

Inference/evaluation:

- `REPORT.md`
- `summary.json`
- `per_class_metrics.csv`
- `predictions.csv`
- `prompts.json`
- `oracle_threshold_diagnostic.csv`
- `latency.json`
- `error_visualizations/*_false_positive.jpg`
- `error_visualizations/*_false_negative.jpg`

GPU latency is an image-encoder-only diagnostic on the server and must not be treated as V516 NPU latency.
