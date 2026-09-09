#!/usr/bin/env python3
"""Fail closed when an image tree and its OneFormer probability cache diverge."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.video_semantic import IMAGE_EXTENSIONS
from distillation.cache import cache_path_for_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    root = args.image_root.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    images = sorted(item for item in root.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)
    if args.max_images > 0:
        images = images[: args.max_images]
    missing, invalid, sums, class_mass = [], [], [], np.zeros(13, dtype=np.float64)
    for image in images:
        path = cache_path_for_image(image, root, cache_root)
        if not path.is_file():
            missing.append(str(image.relative_to(root)))
            continue
        try:
            with np.load(path, allow_pickle=False) as payload:
                if set(payload.files) != {"probabilities"}:
                    raise ValueError(f"unexpected keys={payload.files}")
                value = payload["probabilities"]
            if value.dtype != np.uint8 or value.ndim != 3 or value.shape[0] != 13:
                raise ValueError(f"dtype={value.dtype}, shape={value.shape}")
            pixel_sum = value.sum(0, keepdims=True)
            if (pixel_sum == 0).any():
                raise ValueError("one or more cached pixels have zero probability mass")
            normalized = value.astype(np.float32) / pixel_sum
            sums.append(float(normalized.sum(0).mean()))
            class_mass += normalized.sum((1, 2))
        except Exception as error:
            invalid.append(f"{image.relative_to(root)}: {error}")
    active_classes = int((class_mass > 0).sum())
    report = {
        "format": "oneformer_rvm13_cache_audit_v1",
        "image_root": str(root),
        "cache_root": str(cache_root),
        "images": len(images),
        "valid_entries": len(images) - len(missing) - len(invalid),
        "coverage": (len(images) - len(missing) - len(invalid)) / max(len(images), 1),
        "missing_examples": missing[:20],
        "invalid_examples": invalid[:20],
        "mean_probability_sum": float(np.mean(sums)) if sums else None,
        "active_classes": active_classes,
        "mean_class_distribution": (class_mass / max(class_mass.sum(), 1)).tolist(),
    }
    output = json.dumps(report, indent=2)
    print(output)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(output + "\n", encoding="utf-8")
    if report["coverage"] != 1.0 or active_classes < 2:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
