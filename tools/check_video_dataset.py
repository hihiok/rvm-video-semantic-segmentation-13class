#!/usr/bin/env python3
"""Validate converted 13-class video data before training."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable


CLASS_NAMES = [
    "background", "sky", "person", "plant", "building", "flower", "food",
    "water", "desert", "ice_or_snow", "text", "ball", "mountain",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTENSIONS = (".png", ".bmp", ".tif", ".tiff")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--max-frames", type=int, default=0, help="0 checks every frame")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--require-target", action="store_true",
        help="Reject frames whose mask contains no foreground target class ID 1..12",
    )
    return parser.parse_args()


def find_mask(mask_root, relative_image):
    matches = [mask_root / relative_image.with_suffix(extension) for extension in MASK_EXTENSIONS]
    return [path for path in matches if path.is_file()]


def main():
    args = parse_args()
    result = {"classes": dict(enumerate(CLASS_NAMES)), "splits": {}}
    for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
        image_root = args.data_root / "images" / split
        mask_root = args.data_root / "annotations" / split
        if not image_root.exists() or not mask_root.exists():
            print(f"Skip missing split: {split}")
            continue
        images = sorted(
            path for path in image_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        videos = {path.relative_to(image_root).parts[0] for path in images}
        sequences = {str(path.relative_to(image_root).parent) for path in images}
        pixel_counts = Counter()
        frame_class_counts = Counter()
        checked, errors = 0, []
        for image_path in tqdm(images, desc=f"check {split}"):
            if args.max_frames and checked >= args.max_frames:
                break
            relative = image_path.relative_to(image_root)
            candidates = find_mask(mask_root, relative)
            if len(candidates) != 1:
                errors.append(f"{image_path}: expected one mask, got {candidates}")
                continue
            try:
                with Image.open(image_path) as image_file, Image.open(candidates[0]) as mask_file:
                    if image_file.size != mask_file.size:
                        raise ValueError(f"size mismatch {image_file.size} vs {mask_file.size}")
                    mask = np.asarray(mask_file)
                if mask.ndim != 2:
                    raise ValueError(f"mask shape {mask.shape}")
                invalid = ~np.isin(mask, list(range(13)) + [255])
                if invalid.any():
                    raise ValueError(f"invalid IDs {np.unique(mask[invalid]).tolist()}")
                if args.require_target and not np.any((mask >= 1) & (mask <= 12)):
                    raise ValueError("no foreground target class ID 1..12")
                values, counts = np.unique(mask, return_counts=True)
                for value, count in zip(values.tolist(), counts.tolist()):
                    pixel_counts[value] += count
                    if value != 255:
                        frame_class_counts[value] += 1
                checked += 1
            except Exception as error:
                errors.append(f"{image_path}: {error}")
        result["splits"][split] = {
            "videos": len(videos),
            "sequences": len(sequences),
            "discovered_frames": len(images),
            "checked_frames": checked,
            "require_target": args.require_target,
            "pixel_counts": {str(i): int(pixel_counts[i]) for i in list(range(13)) + [255]},
            "frames_containing_class": {str(i): int(frame_class_counts[i]) for i in range(13)},
            "errors": errors[:100],
            "error_count": len(errors),
        }
        if errors:
            raise RuntimeError(f"{split}: found {len(errors)} errors; first: {errors[0]}")
    output = args.output_json or args.data_root / "validation_report.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["splits"], indent=2))


if __name__ == "__main__":
    main()
