# UltraFace slim 8-label scene classification (640x360)

Standalone multi-label scene classifier using the original UltraFace slim `Mb_Tiny` backbone topology.

Upstream reference: `Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB`, commit `dffdddda9794a50607cba8f318507a28c1c27cab`.

## Architecture

`RGB 3x360x640 -> original Mb_Tiny backbone -> AdaptiveAvgPool2d(1) -> Dropout -> Linear(256,8)`

No RFB, no SSD extras, no bbox head, no face-detection classification head, no FSD dependency.

Labels: `night, indoor, rain_snow, office, outdoor, landscape, sports, objective_image`.

Training labels are partial multi-label `{1,0,-1}` and use masked BCEWithLogitsLoss.

## Environment

Use the existing conda environment named `Ultraface`. Do not create another environment.

## Data

Manifest construction tools remain in `../fsd_scene_multilabel8/`; they only read existing datasets and write JSONL manifests. Source images/labels/masks are never modified.

## Input preprocessing

OpenCV read -> BGR-to-RGB -> resize exactly to 640x360 -> `(pixel - 127) / 128` -> CHW float32.

## Training

Run `run_smoke_ultraface_env.sh`, then `run_train.sh` after the dataset audit passes.
