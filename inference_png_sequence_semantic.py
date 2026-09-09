#!/usr/bin/env python3
"""Run RVM 13-class semantic segmentation on one ordered PNG frame sequence."""

import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from inference_video_semantic import (
    add_legend,
    amp_context,
    colorize,
    prepare_frame,
    resolve_input_shape,
    restore_logits,
    restore_mask,
    scene_cut_score,
)
from model import MultiClassFastGuidedFilterRefiner, RVMForVideoSemanticSegmentation
from semantic_utils import DEFAULT_CLASS_NAMES, torch_load


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0, help="Preview MP4 frame rate")
    parser.add_argument("--max-frames", type=int, default=0, help="0 processes every frame")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--input-width", type=int, default=None)
    parser.add_argument("--input-height", type=int, default=None)
    parser.add_argument("--resize-mode", choices=("letterbox", "stretch"), default="letterbox")
    parser.add_argument("--overlay-alpha", type=float, default=0.5)
    parser.add_argument(
        "--upsample-mode",
        choices=("mask_nearest", "bilinear", "guided"),
        default="bilinear",
    )
    parser.add_argument("--guided-radius", type=int, default=1)
    parser.add_argument("--guided-eps", type=float, default=1e-4)
    parser.add_argument("--recurrent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scene-cut-threshold", type=float, default=0.35)
    parser.add_argument("--reset-interval", type=int, default=0)
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--save-color-masks", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def natural_sort_key(path, root):
    relative = str(path.relative_to(root))
    return tuple(
        (0, int(token)) if token.isdigit() else (1, token.lower())
        for token in re.split(r"(\d+)", relative)
    )


def find_png_frames(input_dir):
    if not input_dir.is_dir():
        raise ValueError(f"PNG input must be a directory: {input_dir}")
    frames = [
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".png"
    ]
    if not frames:
        raise RuntimeError(f"No PNG frames found under: {input_dir}")
    parents = {path.parent.resolve() for path in frames}
    if len(parents) != 1:
        examples = "\n".join(str(path) for path in sorted(parents)[:20])
        raise RuntimeError(
            "Multiple PNG frame directories were found; refusing to mix sequences:\n"
            + examples
        )
    return sorted(frames, key=lambda path: natural_sort_key(path, input_dir))


def read_frame(path):
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Cannot decode PNG frame: {path}")
    return frame


@torch.inference_mode()
def process_sequence(args, model, class_names, input_size, guided_refiner=None):
    frame_paths = find_png_frames(args.input_dir)
    if args.max_frames > 0:
        frame_paths = frame_paths[: args.max_frames]

    first_frame = read_frame(frame_paths[0])
    height, width = first_frame.shape[:2]
    output_path = args.output_dir / f"{args.input_dir.name}.mp4"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {output_path}")

    mask_dir = args.output_dir / f"{args.input_dir.name}_masks"
    if args.save_masks or args.save_color_masks:
        mask_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_path.with_suffix(".jsonl")

    recurrence = [None] * 4
    previous_gray = None
    reset_count = 0
    network_seconds = 0.0
    upsample_seconds = 0.0
    processed = 0

    try:
        with stats_path.open("w", encoding="utf-8") as stats_file:
            for frame_index, frame_path in enumerate(
                tqdm(frame_paths, desc=args.input_dir.name, dynamic_ncols=True)
            ):
                frame = first_frame if frame_index == 0 else read_frame(frame_path)
                if frame.shape[:2] != (height, width):
                    raise RuntimeError(
                        f"Frame size changed at {frame_path}: "
                        f"{frame.shape[1]}x{frame.shape[0]} versus {width}x{height}"
                    )

                cut_score, current_gray = scene_cut_score(previous_gray, frame)
                reset_reason = None
                if (
                    args.recurrent
                    and args.scene_cut_threshold > 0
                    and cut_score is not None
                    and cut_score > args.scene_cut_threshold
                ):
                    reset_reason = "scene_cut"
                if (
                    args.recurrent
                    and args.reset_interval > 0
                    and frame_index > 0
                    and frame_index % args.reset_interval == 0
                ):
                    reset_reason = "interval"
                if reset_reason:
                    recurrence = [None] * 4
                    reset_count += 1
                previous_gray = current_gray

                tensor, geometry = prepare_frame(frame, input_size, args.resize_mode)
                tensor = tensor.to(args.device, non_blocking=True)
                if tensor.is_cuda:
                    torch.cuda.synchronize(tensor.device)
                network_started = time.perf_counter()
                with amp_context(args.amp and str(args.device).startswith("cuda")):
                    logits, *new_recurrence = model(tensor, *recurrence)
                if tensor.is_cuda:
                    torch.cuda.synchronize(tensor.device)
                network_seconds += time.perf_counter() - network_started
                recurrence = new_recurrence if args.recurrent else [None] * 4

                upsample_started = time.perf_counter()
                if args.upsample_mode == "mask_nearest":
                    mask = logits.argmax(dim=1)[0].byte().cpu().numpy()
                    mask = restore_mask(mask, width, height, geometry)
                else:
                    restored_logits = restore_logits(
                        logits, tensor, frame, geometry, args.upsample_mode, guided_refiner
                    )
                    mask = restored_logits.argmax(dim=1)[0].byte().cpu().numpy()
                if tensor.is_cuda:
                    torch.cuda.synchronize(tensor.device)
                upsample_seconds += time.perf_counter() - upsample_started

                color_rgb = colorize(mask)
                color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
                overlay = cv2.addWeighted(
                    frame, 1 - args.overlay_alpha, color_bgr, args.overlay_alpha, 0
                )
                writer.write(add_legend(overlay, mask, class_names))

                if args.save_masks:
                    cv2.imwrite(str(mask_dir / f"{frame_index:06d}.png"), mask)
                if args.save_color_masks:
                    cv2.imwrite(
                        str(mask_dir / f"{frame_index:06d}_color.png"), color_bgr
                    )
                counts = np.bincount(mask.reshape(-1), minlength=len(class_names))
                stats_file.write(json.dumps({
                    "frame": frame_index,
                    "source": str(frame_path),
                    "time_seconds": frame_index / args.fps,
                    "scene_cut_score": cut_score,
                    "state_reset": reset_reason,
                    "class_pixel_ratio": {
                        name: float(counts[index] / mask.size)
                        for index, name in enumerate(class_names)
                    },
                }) + "\n")
                processed += 1
    finally:
        writer.release()

    summary = {
        "source": str(args.input_dir),
        "first_frame": str(frame_paths[0]),
        "last_frame": str(frame_paths[-1]),
        "frames": processed,
        "width": width,
        "height": height,
        "preview_fps": args.fps,
        "recurrent": args.recurrent,
        "state_resets": reset_count,
        "upsample_mode": args.upsample_mode,
        "network_ms_per_frame": 1000.0 * network_seconds / max(processed, 1),
        "upsample_ms_per_frame": 1000.0 * upsample_seconds / max(processed, 1),
        "output": str(output_path),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def main():
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.max_frames < 0:
        raise ValueError("--max-frames cannot be negative")
    if not 0 <= args.overlay_alpha <= 1:
        raise ValueError("--overlay-alpha must be in [0,1]")
    if args.guided_radius < 1 or args.guided_eps <= 0:
        raise ValueError("Guided radius must be >=1 and epsilon must be positive")

    checkpoint = torch_load(args.checkpoint, "cpu")
    class_names = list(checkpoint.get("class_names", DEFAULT_CLASS_NAMES))
    if class_names != DEFAULT_CLASS_NAMES:
        raise ValueError(
            f"Checkpoint class mapping is not the fixed 13-class mapping: {class_names}"
        )
    input_size = resolve_input_shape(args, checkpoint)
    print(f"Inference input resolution: {input_size[1]}x{input_size[0]}")
    model = RVMForVideoSemanticSegmentation(
        checkpoint.get("variant", "mobilenetv3"),
        len(class_names),
        temporal_residual=checkpoint.get("temporal_residual_adapter", False),
        temporal_hidden_channels=checkpoint.get("temporal_hidden_channels", 16),
        temporal_scale=checkpoint.get("temporal_adapter_scale", 0.25),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(args.device)

    guided_refiner = None
    if args.upsample_mode == "guided":
        guided_refiner = MultiClassFastGuidedFilterRefiner(
            args.guided_radius, args.guided_eps
        ).eval().to(args.device)
    process_sequence(args, model, class_names, input_size, guided_refiner)


if __name__ == "__main__":
    main()
