# RVM13 temporal preservation and guided-upsample experiment

## Non-negotiable execution rules

- Use this branch exactly as supplied. Do not edit, generate, delete, commit, or push source code.
- Preserve any pre-existing local changes and stop if the target checkout is dirty.
- Never place proxy credentials in Git, reports, shell history, or logs. Receive the proxy URL through a secure local environment variable.
- Before Git or package downloads, export both lowercase and uppercase HTTP/HTTPS proxy variables from that secure value.
- The internal network performs HTTPS inspection. Before Git operations, set `git config --global http.sslVerify false`.
- Do not enable class weights for this experiment. The objective remains unweighted cross-entropy plus multiclass Dice.

## Fixed experiment inputs

```text
VSPW_ROOT=/data/pub1/z00919662/segmentation/datasets/VSPW_13cls
STATIC_ROOT=/data/pub1/z00919662/segmentation/datasets/COCO_ADE_13cls_16x9_640x360
INIT_CHECKPOINT=/data/pub1/z00919662/segmentation/RobustVideoMatting-semantic-13class-512/output/rvm_semantic_13class_512_2gpu/best_miou.pth
INPUT=640x360
```

Use a new project/output directory. Do not overwrite the completed 80-epoch experiment.

## Required preflight

1. Confirm the Git branch and commit supplied in the handoff.
2. Confirm the worktree is clean.
3. Activate the same PyTorch environment used by the completed RVM13 VSPW training.
4. Run:

```bash
python tests/test_vspw_tools_standalone.py -v
python -m pytest tests/test_vspw_mixed.py tests/test_label_mapping.py -q
python -m compileall -q dataset model tools tests semantic_utils.py train_video_semantic.py train_vspw_mixed.py inference_video_semantic.py
bash -n scripts/train_vspw_mixed.sh
bash -n scripts/train_vspw_temporal_preserve.sh
bash -n scripts/benchmark_guided_upsample.sh
```

Stop on any failure. Do not fix code locally.

## Required CUDA smoke test

Run a reduced one-epoch job with both stages using recurrent-only trainable scope. Confirm:

- only parameters whose names contain `.gru.` are trainable;
- spatial weights and BatchNorm running statistics remain unchanged;
- semantic loss, temporal loss, and temporal stable-pixel count are finite;
- forward and backward pass at 640x360 succeeds;
- `prediction_flip_rate_on_stable_gt` is reported;
- `baseline_stage1.pth` and `last.pth` are written.

Do not proceed to formal training unless the smoke test passes.

## Formal temporal-preservation training

Run `scripts/train_vspw_temporal_preserve.sh` with:

```text
Stage 2: 10 epochs, T=5, video:static optimizer steps=1:1
Stage 3: 15 epochs, T=8, video:static optimizer steps=1:1
Trainable parameters: ConvGRUs only
Stage 2 temporal weight: 0.05
Stage 3 temporal weight: 0.10
Static retention tolerance: 0.0
Class weights: disabled
```

The required deployment candidate is `best_spatial_preserved.pth`. It is eligible only when:

```text
static_mIoU >= baseline static_mIoU
video_mIoU >= baseline video_mIoU
```

Among eligible epochs, selection maximizes:

```text
video_mIoU - 0.1 * prediction_flip_rate_on_stable_gt
```

If no eligible checkpoint exists, report the experiment as unsuccessful. Do not silently substitute `best_balanced.pth` or relax the spatial floor.

## Guided-upsample benchmark

After training, run `scripts/benchmark_guided_upsample.sh`. It compares the same low-resolution logits with:

- bilinear logit upsampling;
- multi-class fast guided-filter logit upsampling.

Report on both VSPW validation and static validation:

```text
mIoU
pixel accuracy
Boundary Precision / Recall / F1
upsample milliseconds per frame
extra peak CUDA memory
per-class IoU
```

The benchmark is an ablation; guided upsampling is accepted only if Boundary F1 improves without an unacceptable mIoU or latency regression.

## Business-video inference

Use `best_spatial_preserved.pth` and produce separate MP4 files for:

```text
--upsample-mode bilinear
--upsample-mode guided
```

Keep recurrent inference enabled and scene-cut reset enabled. Do not compare two upsampling modes in the same recurrent pass; each run must start with empty ConvGRU states.

## Final report

Report branch, commit, environment, GPU IDs, smoke status, trainable parameter count, baseline metrics, best eligible epoch, final metrics, prediction flip-rate change, output checkpoints, guided benchmark JSON files, business MP4 files, warnings, and any human action required.

