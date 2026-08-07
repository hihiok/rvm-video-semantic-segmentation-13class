#!/usr/bin/env python3
"""Validate converted 13-class video data before training."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


CLASS_NAMES = ["background", "sky", "person", "plant", "building", "flower", "food", "water", "desert", "ice_or_snow", "text", "ball", "mountain"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--max-frames", type=int, default=0, help="0 checks every frame")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    result = {"classes": dict(enumerate(CLASS_NAMES)), "splits": {}}
    for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
        image_root = args.data_root / "images" / split
        mask_root = args.data_root / "annotations" / split
        if not image_root.exists() or not mask_root.exists():
            print(f"Skip missing split: {split}")
            continue
        videos = sorted(path for path in image_root.iterdir() if path.is_dir())
        pixel_counts = Counter()
        frame_class_counts = Counter()
        checked, errors = 0, []
        for video_dir in tqdm(videos, desc=f"check {split}"):
            images = sorted(path for path in video_dir.iterdir() if path.is_file())
            for image_path in images:
                if args.max_frames and checked >= args.max_frames:
                    break
                candidates = list((mask_root / video_dir.name).glob(f"{image_path.stem}.*"))
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
                    values, counts = np.unique(mask, return_counts=True)
                    for value, count in zip(values.tolist(), counts.tolist()):
                        pixel_counts[value] += count
                        if value != 255:
                            frame_class_counts[value] += 1
                    checked += 1
                except Exception as error:
                    errors.append(f"{image_path}: {error}")
            if args.max_frames and checked >= args.max_frames:
                break
        result["splits"][split] = {
            "videos": len(videos),
            "checked_frames": checked,
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
