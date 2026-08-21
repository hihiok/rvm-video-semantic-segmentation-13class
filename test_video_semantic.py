#!/usr/bin/env python3
"""Evaluate a 13-class recurrent RVM checkpoint on complete video clips."""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VideoClipDataset, VideoValidTransform
from model import RVMForVideoSemanticSegmentation
from semantic_utils import DEFAULT_CLASS_NAMES, ConfusionMatrix, format_metrics, semantic_loss, torch_load
from train_video_semantic import forward_clip


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", default="images/test")
    parser.add_argument("--annotations", default="annotations/test")
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--input-width", type=int, default=None)
    parser.add_argument("--input-height", type=int, default=None)
    parser.add_argument("--clip-length", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--resize-mode", choices=["letterbox", "stretch"], default="letterbox")
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tbptt-chunk", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def amp_context(enabled):
    return torch.cuda.amp.autocast(enabled=enabled) if torch.cuda.is_available() else nullcontext()


@torch.inference_mode()
def main():
    args = parse_args()
    checkpoint = torch_load(args.checkpoint, "cpu")
    class_names = list(checkpoint.get("class_names", DEFAULT_CLASS_NAMES))
    if class_names != DEFAULT_CLASS_NAMES:
        raise ValueError(f"Expected the fixed 13-class mapping, got {class_names}")
    if args.input_width is not None or args.input_height is not None:
        if args.input_width is None or args.input_height is None:
            raise ValueError("--input-width and --input-height must be supplied together")
        input_size = (args.input_height, args.input_width)
    elif args.input_size is not None:
        input_size = args.input_size
    elif checkpoint.get("input_width") and checkpoint.get("input_height"):
        input_size = (int(checkpoint["input_height"]), int(checkpoint["input_width"]))
    else:
        input_size = int(checkpoint.get("input_size", 512))
    clip_length = args.clip_length or int(checkpoint.get("clip_length", 5))
    dataset = VideoClipDataset(
        args.data_root / args.images,
        args.data_root / args.annotations,
        num_classes=len(class_names),
        clip_length=clip_length,
        frame_stride=args.frame_stride,
        clip_step=clip_length * args.frame_stride,
        transform=VideoValidTransform(input_size, args.resize_mode, args.ignore_index),
        ignore_index=args.ignore_index,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=str(args.device).startswith("cuda"),
        persistent_workers=args.workers > 0,
    )
    model = RVMForVideoSemanticSegmentation(
        checkpoint.get("variant", "mobilenetv3"), len(class_names)
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(args.device)
    matrix = ConfusionMatrix(len(class_names), args.device)
    loss_sum, clip_count, frame_count = 0.0, 0, 0
    for images, masks in tqdm(loader, desc="Test", dynamic_ncols=True):
        images, masks = images.to(args.device, non_blocking=True), masks.to(args.device, non_blocking=True)
        with amp_context(args.amp and str(args.device).startswith("cuda")):
            logits = forward_clip(model, images, args.tbptt_chunk)
            losses = semantic_loss(logits, masks, len(class_names), ignore_index=args.ignore_index)
        predictions = logits.argmax(dim=2)
        matrix.update(masks, predictions, args.ignore_index)
        loss_sum += losses["total"].item() * images.shape[0]
        clip_count += images.shape[0]
        frame_count += int(masks.ne(args.ignore_index).flatten(2).any(2).sum())
    metrics = matrix.compute(class_names)
    metrics.update(
        loss=loss_sum / max(clip_count, 1),
        num_clips=clip_count,
        num_valid_frames=frame_count,
        confusion_matrix=matrix.matrix.to(torch.int64).cpu().tolist(),
    )
    output = args.output_json or args.checkpoint.with_name(f"{args.checkpoint.stem}_video_test.json")
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(format_metrics(metrics))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
