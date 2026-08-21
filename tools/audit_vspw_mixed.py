#!/usr/bin/env python3
"""Audit source pairing, masks, split leakage and 13-class coverage."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from resolve_static_dataset import find_split


CLASS_NAMES = [
    "background", "sky", "person", "plant", "building", "flower", "food",
    "water", "desert", "ice_or_snow", "text", "ball", "mountain",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_SUFFIXES = (".png", ".bmp", ".tif", ".tiff")


def discover_pairs(image_root, mask_root):
    images = sorted(
        path for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise RuntimeError(f"No images found in {image_root}")
    pairs, errors = [], []
    for image in images:
        relative = image.relative_to(image_root)
        matches = [mask_root / relative.with_suffix(ext) for ext in MASK_SUFFIXES]
        matches = [path for path in matches if path.is_file()]
        if len(matches) != 1:
            errors.append(f"{relative}: expected exactly one mask, found {len(matches)}")
        else:
            pairs.append((image, matches[0], relative))
    if errors:
        raise RuntimeError(f"Found {len(errors)} unmatched images; examples: {errors[:10]}")
    return pairs


def evenly_sample(items, maximum):
    if maximum <= 0 or len(items) <= maximum:
        return items
    indices = np.linspace(0, len(items) - 1, num=maximum, dtype=np.int64)
    return [items[index] for index in indices.tolist()]


def audit_split(image_root, mask_root, max_frames, video):
    pairs = discover_pairs(image_root, mask_root)
    selected = evenly_sample(pairs, max_frames)
    pixels = Counter()
    frames = Counter()
    class_videos = defaultdict(set)
    videos = set()
    for image_path, mask_path, relative in selected:
        with Image.open(image_path) as image, Image.open(mask_path) as mask_file:
            if image.size != mask_file.size:
                raise ValueError(f"Image/mask dimensions differ: {image_path}")
            mask = np.asarray(mask_file)
        if mask.ndim != 2:
            raise ValueError(f"Mask is not single-channel: {mask_path} {mask.shape}")
        values, counts = np.unique(mask, return_counts=True)
        invalid = [int(value) for value in values if value != 255 and not 0 <= value <= 12]
        if invalid:
            raise ValueError(f"Invalid IDs {invalid} in {mask_path}; expected 0..12 or 255")
        video_name = relative.parts[0] if video else None
        if video_name is not None:
            videos.add(video_name)
        for value, count in zip(values.tolist(), counts.tolist()):
            pixels[value] += count
            if value != 255:
                frames[value] += 1
                if video_name is not None:
                    class_videos[value].add(video_name)

    all_videos = {relative.parts[0] for _, _, relative in pairs} if video else set()
    result = {
        "images": len(pairs),
        "checked_masks": len(selected),
        "fully_scanned_masks": len(selected) == len(pairs),
        "pixel_counts": {name: int(pixels[index]) for index, name in enumerate(CLASS_NAMES)},
        "ignore_pixels": int(pixels[255]),
        "frames_containing_class": {name: int(frames[index]) for index, name in enumerate(CLASS_NAMES)},
        "missing_classes_in_checked_masks": [
            name for index, name in enumerate(CLASS_NAMES) if pixels[index] == 0
        ],
    }
    if video:
        result["videos"] = len(all_videos)
        result["sampled_videos"] = len(videos)
        result["sampled_videos_containing_class"] = {
            name: len(class_videos[index]) for index, name in enumerate(CLASS_NAMES)
        }
        result["video_names"] = sorted(all_videos)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vspw-root", type=Path, required=True)
    parser.add_argument("--static-root", type=Path, required=True)
    parser.add_argument("--max-video-frames", type=int, default=20000)
    parser.add_argument("--max-static-images", type=int, default=10000)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)

    result = {"class_names": CLASS_NAMES, "vspw": {}, "static": {}}
    for split in ("train", "val"):
        video_images = args.vspw_root / "images" / split
        video_masks = args.vspw_root / "annotations" / split
        if not video_images.is_dir() or not video_masks.is_dir():
            raise FileNotFoundError(f"Missing converted VSPW split: {video_images}, {video_masks}")
        static = find_split(args.static_root, split)
        if static is None:
            raise FileNotFoundError(f"Missing static {split} split below {args.static_root}")
        result["vspw"][split] = audit_split(video_images, video_masks, args.max_video_frames, True)
        result["static"][split] = audit_split(
            Path(static["images"]), Path(static["annotations"]), args.max_static_images, False
        )

    train_videos = set(result["vspw"]["train"].pop("video_names"))
    val_videos = set(result["vspw"]["val"].pop("video_names"))
    overlap = sorted(train_videos & val_videos)
    result["train_val_video_overlap"] = overlap[:20]
    result["train_val_video_overlap_count"] = len(overlap)
    if overlap:
        raise RuntimeError(f"VSPW train/val video leakage detected: {overlap[:10]}")

    result["warnings"] = []
    for index, name in enumerate(CLASS_NAMES):
        video_count = result["vspw"]["train"]["pixel_counts"][name]
        static_count = result["static"]["train"]["pixel_counts"][name]
        if video_count == 0 and static_count == 0:
            result["warnings"].append(f"{name}: absent from BOTH sampled training sources")
        elif video_count == 0:
            result["warnings"].append(f"{name}: absent from sampled VSPW training frames; static replay preserves it")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
