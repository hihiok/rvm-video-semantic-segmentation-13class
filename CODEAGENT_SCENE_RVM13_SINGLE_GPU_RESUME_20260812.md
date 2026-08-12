# CODEAGENT — SCENE-RVM13-SINGLE-GPU-RESUME-20260812

Use the downloadable full task file provided by ChatGPT for the complete execution checklist. This repository already contains the required launcher at `scripts/train_1gpu_resume.sh`.

Required launcher commit:

```text
106c9d28723e2cd3ecb1d48f5e4a9b425ef97c5e
```

Core requirements:

- Abandon DDP / NCCL / torchrun.
- Use exactly one V100 GPU.
- Resume from the latest valid 13-class `output/rvm_video_semantic_13class/last.pth` whenever available.
- `train_video_semantic.py` already restores model, optimizer, scheduler, AMP scaler, epoch and best mIoU through `--resume`.
- Always clear `LD_LIBRARY_PATH` and `LD_PRELOAD` so PyTorch does not load the incompatible system cuDNN 8.7.0.
- Continue the verified configuration: T=5, stride=1, input 512, batch size 2, gradient accumulation 2, workers 8, AMP enabled, 100 target epochs.
- Do not pass `--sync-bn` in single-GPU mode.
- Back up current checkpoints and metrics before launching.
- Launch persistently in `tmux` session `rvm_video13_single`.
- Verify the first resumed epoch equals `checkpoint["epoch"] + 1`; stop immediately if it starts at epoch 0.
- Leave healthy Stage 12 training running.
- Do not execute Stage 13 or Stage 14 until training completes.

Canonical launch after choosing an idle physical GPU:

```bash
cd /data/pub1/z00919662/segmentation/rvm-video-semantic-segmentation-13class
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=<GPU_ID>
unset LD_LIBRARY_PATH
unset LD_PRELOAD
bash scripts/train_1gpu_resume.sh
```
