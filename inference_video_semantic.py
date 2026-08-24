#!/usr/bin/env python3
"""Run recurrent 13-class RVM semantic segmentation directly on video files."""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from model import RVMForVideoSemanticSegmentation
from semantic_utils import DEFAULT_CLASS_NAMES, DEFAULT_PALETTE, torch_load


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--input-width", type=int, default=None)
    parser.add_argument("--input-height", type=int, default=None)
    parser.add_argument("--resize-mode", choices=["letterbox", "stretch"], default="letterbox")
    parser.add_argument("--overlay-alpha", type=float, default=0.5)
    parser.add_argument("--recurrent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--scene-cut-threshold",
        type=float,
        default=0.35,
        help="Reset ConvGRU when normalized low-resolution frame difference exceeds this value; <=0 disables.",
    )
    parser.add_argument("--reset-interval", type=int, default=0, help="Also reset state every N frames; 0 disables")
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--save-color-masks", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def find_videos(path: Path):
    if path.is_file():
        return [path]
    return sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS)


def spatial_dimensions(size):
    if isinstance(size, int):
        return size, size
    if isinstance(size, (tuple, list)) and len(size) == 2:
        return int(size[0]), int(size[1])
    raise ValueError("Input size must be an integer or a (height, width) pair")


def letterbox_geometry(width, height, size):
    target_h, target_w = spatial_dimensions(size)
    scale = min(target_w / width, target_h / height)
    resized_w = max(1, min(target_w, int(round(width * scale))))
    resized_h = max(1, min(target_h, int(round(height * scale))))
    left, top = (target_w - resized_w) // 2, (target_h - resized_h) // 2
    return resized_w, resized_h, (
        left, top, target_w - resized_w - left, target_h - resized_h - top
    )


def prepare_frame(frame_bgr, size, resize_mode):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    target_h, target_w = spatial_dimensions(size)
    if resize_mode == "stretch":
        resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        geometry = None
    else:
        height, width = rgb.shape[:2]
        resized_w, resized_h, padding = letterbox_geometry(width, height, size)
        resized = cv2.resize(rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        left, top, right, bottom = padding
        resized = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        geometry = (resized_w, resized_h, padding)
    tensor = torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1).float().div_(255).unsqueeze(0)
    return tensor, geometry


def restore_mask(mask, original_width, original_height, geometry):
    if geometry is not None:
        resized_w, resized_h, padding = geometry
        left, top, _, _ = padding
        mask = mask[top : top + resized_h, left : left + resized_w]
    return cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)


def colorize(mask):
    palette = np.asarray(DEFAULT_PALETTE, dtype=np.uint8)
    return palette[mask]


def add_legend(frame, mask, class_names):
    present = [int(value) for value in np.unique(mask) if int(value) < len(class_names)]
    line_height = 24
    box_width = min(frame.shape[1] - 8, 240)
    box_height = 8 + line_height * len(present)
    overlay = frame.copy()
    cv2.rectangle(overlay, (4, 4), (4 + box_width, 4 + box_height), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)
    for row, class_id in enumerate(present):
        y = 22 + row * line_height
        rgb = DEFAULT_PALETTE[class_id]
        cv2.rectangle(frame, (12, y - 13), (28, y + 3), tuple(reversed(rgb)), -1)
        cv2.putText(frame, f"{class_id}: {class_names[class_id]}", (36, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def scene_cut_score(previous_small_gray, frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    score = None if previous_small_gray is None else float(np.mean(np.abs(gray - previous_small_gray)))
    return score, gray


def amp_context(enabled):
    return torch.cuda.amp.autocast(enabled=enabled) if torch.cuda.is_available() else nullcontext()


@torch.inference_mode()
def process_video(video_path, relative_path, args, model, class_names, input_size):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_path = (args.output_dir / relative_path).with_suffix(".mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {output_path}")
    mask_dir = args.output_dir / f"{Path(relative_path).with_suffix('')}_masks"
    if args.save_masks or args.save_color_masks:
        mask_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_path.with_suffix(".jsonl")
    stats_file = stats_path.open("w", encoding="utf-8")

    recurrence = [None] * 4
    previous_gray = None
    frame_index = 0
    reset_count = 0
    progress = tqdm(total=frame_total if frame_total > 0 else None, desc=video_path.name, dynamic_ncols=True)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            cut_score, current_gray = scene_cut_score(previous_gray, frame)
            reset_reason = None
            if args.recurrent and args.scene_cut_threshold > 0 and cut_score is not None and cut_score > args.scene_cut_threshold:
                reset_reason = "scene_cut"
            if args.recurrent and args.reset_interval > 0 and frame_index > 0 and frame_index % args.reset_interval == 0:
                reset_reason = "interval"
            if reset_reason:
                recurrence = [None] * 4
                reset_count += 1
            previous_gray = current_gray

            tensor, geometry = prepare_frame(frame, input_size, args.resize_mode)
            tensor = tensor.to(args.device, non_blocking=True)
            with amp_context(args.amp and str(args.device).startswith("cuda")):
                logits, *new_recurrence = model(tensor, *recurrence)
            recurrence = new_recurrence if args.recurrent else [None] * 4
            mask = logits.argmax(dim=1)[0].byte().cpu().numpy()
            mask = restore_mask(mask, width, height, geometry)
            color_rgb = colorize(mask)
            color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
            overlay = cv2.addWeighted(frame, 1 - args.overlay_alpha, color_bgr, args.overlay_alpha, 0)
            overlay = add_legend(overlay, mask, class_names)
            writer.write(overlay)

            if args.save_masks:
                cv2.imwrite(str(mask_dir / f"{frame_index:06d}.png"), mask)
            if args.save_color_masks:
                cv2.imwrite(str(mask_dir / f"{frame_index:06d}_color.png"), color_bgr)
            counts = np.bincount(mask.reshape(-1), minlength=len(class_names))
            stats_file.write(json.dumps({
                "frame": frame_index,
                "time_seconds": frame_index / fps,
                "scene_cut_score": cut_score,
                "state_reset": reset_reason,
                "class_pixel_ratio": {name: float(counts[i] / mask.size) for i, name in enumerate(class_names)},
            }) + "\n")
            frame_index += 1
            progress.update(1)
    finally:
        progress.close()
        stats_file.close()
        writer.release()
        capture.release()
    print(f"Saved {frame_index} frames, {reset_count} state resets: {output_path}")


def main():
    args = parse_args()
    if not 0 <= args.overlay_alpha <= 1:
        raise ValueError("--overlay-alpha must be in [0,1]")
    checkpoint = torch_load(args.checkpoint, "cpu")
    class_names = list(checkpoint.get("class_names", DEFAULT_CLASS_NAMES))
    if class_names != DEFAULT_CLASS_NAMES:
        raise ValueError(f"Checkpoint class mapping is not the fixed 13-class mapping: {class_names}")
    if args.input_size is not None and (
        args.input_width is not None or args.input_height is not None
    ):
        raise ValueError("Use either --input-size or --input-width/--input-height, not both")
    if (args.input_width is None) != (args.input_height is None):
        raise ValueError("--input-width and --input-height must be supplied together")
    if args.input_size is not None:
        input_size = (args.input_size, args.input_size)
    elif args.input_width is not None:
        input_size = (args.input_height, args.input_width)
    else:
        legacy_size = int(checkpoint.get("input_size", 512))
        input_size = (
            int(checkpoint.get("input_height", legacy_size)),
            int(checkpoint.get("input_width", legacy_size)),
        )
    if min(input_size) < 1:
        raise ValueError("Input width and height must be positive")
    print(f"Network input: {input_size[1]}x{input_size[0]} (width x height)")
    model = RVMForVideoSemanticSegmentation(checkpoint.get("variant", "mobilenetv3"), len(class_names))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(args.device)
    videos = find_videos(args.input)
    if not videos:
        raise RuntimeError(f"No video files found: {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for video in videos:
        relative = Path(video.name) if args.input.is_file() else video.relative_to(args.input)
        process_video(video, relative, args, model, class_names, input_size)


if __name__ == "__main__":
    main()
