#!/usr/bin/env python3
"""Run recurrent 13-class RVM semantic segmentation directly on video files."""

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from model import MultiClassFastGuidedFilterRefiner, RVMForVideoSemanticSegmentation
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
    parser.add_argument(
        "--upsample-mode",
        choices=("mask_nearest", "bilinear", "guided"),
        default="mask_nearest",
        help="Restore model output to source resolution; mask_nearest preserves legacy behavior.",
    )
    parser.add_argument("--guided-radius", type=int, default=1)
    parser.add_argument("--guided-eps", type=float, default=1e-4)
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


def normalize_spatial_size(size):
    if isinstance(size, int):
        return size, size
    if isinstance(size, (tuple, list)) and len(size) == 2:
        return int(size[0]), int(size[1])
    raise ValueError(f"Expected an integer or (height, width), received {size!r}")


def resolve_input_shape(args, checkpoint):
    if args.input_width is not None or args.input_height is not None:
        if args.input_width is None or args.input_height is None:
            raise ValueError("--input-width and --input-height must be supplied together")
        return args.input_height, args.input_width
    if args.input_size is not None:
        return args.input_size, args.input_size
    if checkpoint.get("input_width") and checkpoint.get("input_height"):
        return int(checkpoint["input_height"]), int(checkpoint["input_width"])
    legacy_size = int(checkpoint.get("input_size", 512))
    return legacy_size, legacy_size


def letterbox_geometry(width, height, size):
    target_h, target_w = normalize_spatial_size(size)
    scale = min(target_w / width, target_h / height)
    resized_w = max(1, min(target_w, int(round(width * scale))))
    resized_h = max(1, min(target_h, int(round(height * scale))))
    left, top = (target_w - resized_w) // 2, (target_h - resized_h) // 2
    return resized_w, resized_h, (
        left, top, target_w - resized_w - left, target_h - resized_h - top
    )


def prepare_frame(frame_bgr, size, resize_mode):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    target_h, target_w = normalize_spatial_size(size)
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


def crop_model_padding(tensor, geometry):
    if geometry is None:
        return tensor
    resized_w, resized_h, padding = geometry
    left, top, _, _ = padding
    return tensor[..., top : top + resized_h, left : left + resized_w]


def frame_rgb_tensor(frame_bgr, device, dtype=torch.float32):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return (
        torch.from_numpy(np.ascontiguousarray(rgb))
        .permute(2, 0, 1)
        .to(device=device, dtype=dtype)
        .div_(255)
        .unsqueeze(0)
    )


def restore_logits(
    logits,
    base_rgb,
    frame_bgr,
    geometry,
    mode,
    guided_refiner=None,
):
    """Restore class logits before argmax so competing classes remain available."""
    logits = crop_model_padding(logits, geometry)
    base_rgb = crop_model_padding(base_rgb, geometry)
    output_size = frame_bgr.shape[:2]
    if mode == "bilinear":
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
    if mode != "guided" or guided_refiner is None:
        raise ValueError(f"Unsupported logit restoration mode: {mode}")
    fine_rgb = frame_rgb_tensor(frame_bgr, logits.device, logits.dtype)
    # The guided filter is numerically safer in FP32 even when the network uses AMP.
    return guided_refiner(
        base_rgb.float(), logits.float(), fine_rgb.float()
    ).to(logits.dtype)


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
def process_video(video_path, relative_path, args, model, class_names, input_size, guided_refiner=None):
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
    network_seconds = 0.0
    upsample_seconds = 0.0
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
    summary = {
        "frames": frame_index,
        "state_resets": reset_count,
        "upsample_mode": args.upsample_mode,
        "network_ms_per_frame": 1000.0 * network_seconds / max(frame_index, 1),
        "upsample_ms_per_frame": 1000.0 * upsample_seconds / max(frame_index, 1),
        "output": str(output_path),
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    args = parse_args()
    if not 0 <= args.overlay_alpha <= 1:
        raise ValueError("--overlay-alpha must be in [0,1]")
    if args.guided_radius < 1 or args.guided_eps <= 0:
        raise ValueError("Guided radius must be >=1 and epsilon must be positive")
    checkpoint = torch_load(args.checkpoint, "cpu")
    class_names = list(checkpoint.get("class_names", DEFAULT_CLASS_NAMES))
    if class_names != DEFAULT_CLASS_NAMES:
        raise ValueError(f"Checkpoint class mapping is not the fixed 13-class mapping: {class_names}")
    input_size = resolve_input_shape(args, checkpoint)
    print(f"Inference input resolution: {input_size[1]}x{input_size[0]}")
    model = RVMForVideoSemanticSegmentation(checkpoint.get("variant", "mobilenetv3"), len(class_names))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(args.device)
    guided_refiner = None
    if args.upsample_mode == "guided":
        guided_refiner = MultiClassFastGuidedFilterRefiner(
            args.guided_radius, args.guided_eps
        ).eval().to(args.device)
    videos = find_videos(args.input)
    if not videos:
        raise RuntimeError(f"No video files found: {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for video in videos:
        relative = Path(video.name) if args.input.is_file() else video.relative_to(args.input)
        process_video(video, relative, args, model, class_names, input_size, guided_refiner)


if __name__ == "__main__":
    main()
