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


OTHER_CLASS_NAME = "other"


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
    parser.add_argument(
        "--output-classes",
        nargs="+",
        default=None,
        metavar="CLASS",
        help=(
            "Limit the output to named checkpoint classes. Add 'other' to keep a "
            "catch-all label; for example: sky water mountain other."
        ),
    )
    parser.add_argument(
        "--restricted-min-confidence",
        type=float,
        default=0.20,
        help=(
            "With 'other' enabled, assign a pixel to its best requested class only "
            "when that class reaches this probability; otherwise emit other."
        ),
    )
    parser.add_argument(
        "--temporal-ema-alpha",
        type=float,
        default=1.0,
        help=(
            "Current-frame weight for probability EMA. 1 disables EMA; values such "
            "as 0.20-0.30 give strong, inexpensive temporal smoothing."
        ),
    )
    parser.add_argument(
        "--temporal-hysteresis-margin",
        type=float,
        default=0.0,
        help=(
            "Extra probability advantage required before a pixel changes label. "
            "0 disables hysteresis; 0.05-0.10 is a useful stable range."
        ),
    )
    parser.add_argument(
        "--scene-cut-method",
        choices=("histogram", "gray"),
        default="histogram",
        help="Histogram is robust to object/camera motion; gray preserves legacy behavior.",
    )
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


def resolve_output_classes(requested, class_names):
    """Resolve a user-facing class list while retaining checkpoint class IDs."""
    if requested is None:
        return None

    normalized = [name.strip().lower() for name in requested]
    if not normalized or any(not name for name in normalized):
        raise ValueError("--output-classes must contain at least one class name")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Duplicate --output-classes are not allowed: {requested}")

    include_other = OTHER_CLASS_NAME in normalized
    selected_names = [name for name in normalized if name != OTHER_CLASS_NAME]
    if not selected_names:
        raise ValueError("--output-classes cannot contain only 'other'")

    name_to_index = {name.lower(): index for index, name in enumerate(class_names)}
    unknown = [name for name in selected_names if name not in name_to_index]
    if unknown:
        raise ValueError(
            f"Unknown output classes {unknown}; available classes: {class_names}, other"
        )

    selected_indices = [name_to_index[name] for name in selected_names]
    if include_other and 0 in selected_indices:
        raise ValueError(
            "'background' and 'other' cannot be requested together because both use "
            "output mask ID 0"
        )
    return {
        "requested": normalized,
        "selected_names": selected_names,
        "selected_indices": selected_indices,
        "include_other": include_other,
    }


def histogram_scene_cut_score(previous_histogram, frame):
    """Return a [0, 1] HSV histogram distance and the current frame histogram."""
    thumbnail = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv], [0, 1, 2], None, [24, 16, 16], [0, 180, 0, 256, 0, 256]
    )
    histogram = cv2.normalize(
        histogram, None, alpha=1.0, norm_type=cv2.NORM_L1
    ).astype(np.float32)
    if previous_histogram is None:
        return None, histogram
    score = cv2.compareHist(
        previous_histogram, histogram, cv2.HISTCMP_BHATTACHARYYA
    )
    return float(np.clip(score, 0.0, 1.0)), histogram


def update_probability_ema(current_probabilities, previous_probabilities, alpha):
    if previous_probabilities is None or alpha >= 1.0:
        return current_probabilities
    return (
        alpha * current_probabilities
        + (1.0 - alpha) * previous_probabilities
    )


def unrestricted_mask_from_probabilities(probabilities, previous_mask=None, margin=0.0):
    confidence, candidate = probabilities.max(dim=1)
    if previous_mask is None or margin <= 0:
        return candidate
    previous_confidence = probabilities.gather(
        1, previous_mask.unsqueeze(1).long()
    ).squeeze(1)
    keep_previous = (
        candidate.ne(previous_mask)
        & confidence.lt(previous_confidence + margin)
    )
    return torch.where(keep_previous, previous_mask, candidate)


def restricted_mask_from_probabilities(
    probabilities,
    output_spec,
    min_confidence,
    previous_mask=None,
    margin=0.0,
):
    """Map probabilities to retained source IDs plus optional ID-0 other."""
    selected_indices = output_spec["selected_indices"]
    index_tensor = torch.as_tensor(
        selected_indices, device=probabilities.device, dtype=torch.long
    )
    selected = probabilities.index_select(1, index_tensor)
    best_confidence, best_position = selected.max(dim=1)
    candidate = index_tensor[best_position]

    if output_spec["include_other"]:
        candidate = torch.where(
            best_confidence.ge(min_confidence), candidate, torch.zeros_like(candidate)
        )
    if previous_mask is None or margin <= 0:
        return candidate

    same = candidate.eq(previous_mask)
    switch = same.clone()
    previous_is_other = previous_mask.eq(0) & output_spec["include_other"]
    candidate_is_other = candidate.eq(0) & output_spec["include_other"]

    # Enter/leave other with two different thresholds to avoid rapid toggling.
    switch |= (
        previous_is_other
        & ~candidate_is_other
        & best_confidence.ge(min(1.0, min_confidence + margin))
    )
    switch |= (
        ~previous_is_other
        & candidate_is_other
        & best_confidence.lt(max(0.0, min_confidence - margin))
    )

    # Between two retained semantic classes, require a real confidence advantage.
    both_selected = ~previous_is_other & ~candidate_is_other & ~same
    safe_previous = previous_mask.clamp(0, probabilities.shape[1] - 1).long()
    previous_confidence = probabilities.gather(
        1, safe_previous.unsqueeze(1)
    ).squeeze(1)
    switch |= (
        both_selected
        & best_confidence.ge(previous_confidence + margin)
    )
    return torch.where(switch, candidate, previous_mask)


def output_mask_from_probabilities(
    probabilities,
    output_spec,
    min_confidence,
    previous_mask=None,
    margin=0.0,
):
    if output_spec is None:
        return unrestricted_mask_from_probabilities(
            probabilities, previous_mask, margin
        )
    return restricted_mask_from_probabilities(
        probabilities,
        output_spec,
        min_confidence,
        previous_mask,
        margin,
    )


def output_class_ratios(mask, class_names, output_spec):
    counts = np.bincount(mask.reshape(-1), minlength=len(class_names))
    if output_spec is None:
        return {
            name: float(counts[index] / mask.size)
            for index, name in enumerate(class_names)
        }
    ratios = {}
    if output_spec["include_other"]:
        ratios[OTHER_CLASS_NAME] = float(counts[0] / mask.size)
    for name, index in zip(
        output_spec["selected_names"], output_spec["selected_indices"]
    ):
        ratios[name] = float(counts[index] / mask.size)
    return ratios


@torch.inference_mode()
def process_sequence(
    args, model, class_names, input_size, output_spec, guided_refiner=None
):
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
    previous_scene_signature = None
    previous_ema_probabilities = None
    previous_raw_mask_lowres = None
    previous_stable_mask_lowres = None
    previous_output_mask = None
    reset_count = 0
    scene_cut_reset_count = 0
    network_seconds = 0.0
    upsample_seconds = 0.0
    raw_flip_sum = 0.0
    stable_flip_sum = 0.0
    flip_transition_count = 0
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

                if args.scene_cut_method == "histogram":
                    cut_score, current_scene_signature = histogram_scene_cut_score(
                        previous_scene_signature, frame
                    )
                else:
                    cut_score, current_scene_signature = scene_cut_score(
                        previous_scene_signature, frame
                    )
                reset_reason = None
                if (
                    args.scene_cut_threshold > 0
                    and cut_score is not None
                    and cut_score > args.scene_cut_threshold
                ):
                    reset_reason = "scene_cut"
                if (
                    reset_reason is None
                    and args.reset_interval > 0
                    and frame_index > 0
                    and frame_index % args.reset_interval == 0
                ):
                    reset_reason = "interval"
                if reset_reason:
                    recurrence = [None] * 4
                    previous_ema_probabilities = None
                    previous_raw_mask_lowres = None
                    previous_stable_mask_lowres = None
                    previous_output_mask = None
                    reset_count += 1
                    if reset_reason == "scene_cut":
                        scene_cut_reset_count += 1
                previous_scene_signature = current_scene_signature

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
                current_probabilities = torch.softmax(logits.float(), dim=1)
                ema_probabilities = update_probability_ema(
                    current_probabilities,
                    previous_ema_probabilities,
                    args.temporal_ema_alpha,
                )
                raw_mask_lowres = output_mask_from_probabilities(
                    current_probabilities,
                    output_spec,
                    args.restricted_min_confidence,
                )
                stable_mask_lowres = output_mask_from_probabilities(
                    ema_probabilities,
                    output_spec,
                    args.restricted_min_confidence,
                    previous_stable_mask_lowres,
                    args.temporal_hysteresis_margin,
                )
                raw_flip_ratio = None
                stable_flip_ratio = None
                if previous_raw_mask_lowres is not None:
                    raw_flip_ratio = float(
                        raw_mask_lowres.ne(previous_raw_mask_lowres).float().mean().item()
                    )
                    stable_flip_ratio = float(
                        stable_mask_lowres.ne(previous_stable_mask_lowres)
                        .float()
                        .mean()
                        .item()
                    )
                    raw_flip_sum += raw_flip_ratio
                    stable_flip_sum += stable_flip_ratio
                    flip_transition_count += 1

                if args.upsample_mode == "mask_nearest":
                    mask = stable_mask_lowres[0].byte().cpu().numpy()
                    mask = restore_mask(mask, width, height, geometry)
                else:
                    restored_probabilities = restore_logits(
                        ema_probabilities,
                        tensor,
                        frame,
                        geometry,
                        args.upsample_mode,
                        guided_refiner,
                    )
                    output_mask = output_mask_from_probabilities(
                        restored_probabilities,
                        output_spec,
                        args.restricted_min_confidence,
                        previous_output_mask,
                        args.temporal_hysteresis_margin,
                    )
                    previous_output_mask = output_mask
                    mask = output_mask[0].byte().cpu().numpy()
                if tensor.is_cuda:
                    torch.cuda.synchronize(tensor.device)
                upsample_seconds += time.perf_counter() - upsample_started
                previous_ema_probabilities = ema_probabilities
                previous_raw_mask_lowres = raw_mask_lowres
                previous_stable_mask_lowres = stable_mask_lowres

                color_rgb = colorize(mask)
                color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
                overlay = cv2.addWeighted(
                    frame, 1 - args.overlay_alpha, color_bgr, args.overlay_alpha, 0
                )
                display_class_names = list(class_names)
                if output_spec is not None and output_spec["include_other"]:
                    display_class_names[0] = OTHER_CLASS_NAME
                writer.write(add_legend(overlay, mask, display_class_names))

                if args.save_masks:
                    cv2.imwrite(str(mask_dir / f"{frame_index:06d}.png"), mask)
                if args.save_color_masks:
                    cv2.imwrite(
                        str(mask_dir / f"{frame_index:06d}_color.png"), color_bgr
                    )
                stats_file.write(json.dumps({
                    "frame": frame_index,
                    "source": str(frame_path),
                    "time_seconds": frame_index / args.fps,
                    "scene_cut_score": cut_score,
                    "scene_cut_method": args.scene_cut_method,
                    "state_reset": reset_reason,
                    "raw_flip_ratio_lowres": raw_flip_ratio,
                    "stabilized_flip_ratio_lowres": stable_flip_ratio,
                    "output_class_pixel_ratio": output_class_ratios(
                        mask, class_names, output_spec
                    ),
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
        "scene_cut_resets": scene_cut_reset_count,
        "scene_cut_method": args.scene_cut_method,
        "scene_cut_threshold": args.scene_cut_threshold,
        "output_classes": (
            list(class_names) if output_spec is None else output_spec["requested"]
        ),
        "restricted_min_confidence": args.restricted_min_confidence,
        "temporal_ema_alpha": args.temporal_ema_alpha,
        "temporal_hysteresis_margin": args.temporal_hysteresis_margin,
        "raw_flip_rate_lowres": raw_flip_sum / max(flip_transition_count, 1),
        "stabilized_flip_rate_lowres": (
            stable_flip_sum / max(flip_transition_count, 1)
        ),
        "flip_transitions_measured": flip_transition_count,
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
    if not 0 <= args.restricted_min_confidence <= 1:
        raise ValueError("--restricted-min-confidence must be in [0,1]")
    if not 0 < args.temporal_ema_alpha <= 1:
        raise ValueError("--temporal-ema-alpha must be in (0,1]")
    if not 0 <= args.temporal_hysteresis_margin <= 1:
        raise ValueError("--temporal-hysteresis-margin must be in [0,1]")

    checkpoint = torch_load(args.checkpoint, "cpu")
    class_names = list(checkpoint.get("class_names", DEFAULT_CLASS_NAMES))
    if class_names != DEFAULT_CLASS_NAMES:
        raise ValueError(
            f"Checkpoint class mapping is not the fixed 13-class mapping: {class_names}"
        )
    output_spec = resolve_output_classes(args.output_classes, class_names)
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
    process_sequence(
        args, model, class_names, input_size, output_spec, guided_refiner
    )


if __name__ == "__main__":
    main()
