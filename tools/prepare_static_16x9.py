#!/usr/bin/env python3
"""Prepare COCO+ADE13 once as fixed 16:9 image/mask pairs for fast replay."""

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from resolve_static_dataset import find_split


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_SUFFIXES = (".png", ".bmp", ".tif", ".tiff")
CLASS_NAMES = [
    "background", "sky", "person", "plant", "building", "flower", "food",
    "water", "desert", "ice_or_snow", "text", "ball", "mountain",
]
MANIFEST_NAME = "PREPARED_16X9_MANIFEST.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--crop-probability", type=float, default=0.5)
    parser.add_argument("--min-foreground-retention", type=float, default=0.45)
    parser.add_argument("--crop-attempts", type=int, default=12)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-images-per-split", type=int, default=0)
    args = parser.parse_args(argv)
    if args.width < 1 or args.height < 1 or args.width * 9 != args.height * 16:
        parser.error("Output dimensions must be a positive exact 16:9 ratio")
    if not 0 <= args.crop_probability <= 1 or not 0 <= args.min_foreground_retention <= 1:
        parser.error("Probabilities and foreground retention must be between 0 and 1")
    if args.crop_attempts < 1 or args.workers < 1:
        parser.error("--crop-attempts and --workers must be positive")
    if args.source_root.resolve() == args.output_root.resolve():
        parser.error("Output root must differ from the original immutable dataset root")
    return args


def discover_pairs(source_root, split, maximum=0):
    paths = find_split(source_root, split)
    if paths is None:
        raise FileNotFoundError(f"Missing source {split} images/annotations under {source_root}")
    image_root, mask_root = Path(paths["images"]), Path(paths["annotations"])
    images = sorted(
        path for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if maximum > 0:
        images = images[:maximum]
    if not images:
        raise RuntimeError(f"No source {split} images found in {image_root}")
    pairs = []
    for image in images:
        relative = image.relative_to(image_root)
        matches = [mask_root / relative.with_suffix(ext) for ext in MASK_SUFFIXES]
        matches = [candidate for candidate in matches if candidate.is_file()]
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected exactly one mask for {relative}; found {matches}")
        pairs.append((image, matches[0], relative))
    return pairs


def deterministic_random(seed, split, relative):
    digest = hashlib.sha256(f"{seed}:{split}:{relative}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def validate_mask(mask, source):
    values = np.unique(mask)
    invalid = [int(value) for value in values if value != 255 and not 0 <= value <= 12]
    if mask.ndim != 2 or invalid:
        raise ValueError(f"Invalid mask {source}: shape={mask.shape}, invalid_ids={invalid}")
    return {int(value) for value in values if 1 <= value <= 12}


def crop_box(width, height, target_width, target_height, rng):
    target_ratio = target_width / target_height
    if width / height >= target_ratio:
        crop_width = min(width, max(1, int(round(height * target_ratio))))
        crop_height = height
        left = rng.randint(0, width - crop_width)
        top = 0
    else:
        crop_width = width
        crop_height = min(height, max(1, int(round(width / target_ratio))))
        left = 0
        top = rng.randint(0, height - crop_height)
    return left, top, left + crop_width, top + crop_height


def choose_safe_crop(mask, target_width, target_height, rng, attempts, min_retention):
    height, width = mask.shape
    foreground = (mask > 0) & (mask != 255)
    foreground_total = int(foreground.sum())
    source_classes = {int(value) for value in np.unique(mask) if 1 <= value <= 12}
    for _ in range(attempts):
        box = crop_box(width, height, target_width, target_height, rng)
        left, top, right, bottom = box
        cropped = mask[top:bottom, left:right]
        classes = {int(value) for value in np.unique(cropped) if 1 <= value <= 12}
        if classes != source_classes:
            continue
        retained = int(((cropped > 0) & (cropped != 255)).sum())
        retention = retained / foreground_total if foreground_total else 1.0
        if retention >= min_retention:
            return box, retention
    return None, -1.0


def resize_and_pad(image, mask, target_width, target_height):
    scale = min(target_width / image.width, target_height / image.height)
    width = max(1, min(target_width, int(round(image.width * scale))))
    height = max(1, min(target_height, int(round(image.height * scale))))
    resized_image = image.resize((width, height), Image.Resampling.BILINEAR)
    resized_mask = mask.resize((width, height), Image.Resampling.NEAREST)
    left, top = (target_width - width) // 2, (target_height - height) // 2
    output_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    output_mask = Image.new("L", (target_width, target_height), 255)
    output_image.paste(resized_image, (left, top))
    output_mask.paste(resized_mask, (left, top))
    return output_image, output_mask


def valid_existing(image_path, mask_path, width, height):
    if not image_path.is_file() or not mask_path.is_file():
        return False
    try:
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            return image.size == (width, height) and mask.size == (width, height)
    except (OSError, ValueError):
        return False


def process_one(job):
    (
        source_image, source_mask, relative, split, output_root, target_width,
        target_height, crop_probability, min_retention, crop_attempts, seed,
        quality, resume,
    ) = job
    source_image, source_mask = Path(source_image), Path(source_mask)
    relative, output_root = Path(relative), Path(output_root)
    output_image = output_root / "images" / split / relative.with_suffix(".jpg")
    output_mask = output_root / "annotations" / split / relative.with_suffix(".png")
    if resume and valid_existing(output_image, output_mask, target_width, target_height):
        with Image.open(source_image) as image, Image.open(output_mask) as mask:
            original_size = image.size
            values, counts = np.unique(np.asarray(mask), return_counts=True)
        return {
            "split": split, "relative": str(relative), "mode": "existing",
            "source_width": original_size[0], "source_height": original_size[1],
            "class_pixels": {str(int(value)): int(count) for value, count in zip(values, counts)},
        }

    with Image.open(source_image) as handle:
        image = handle.convert("RGB")
    with Image.open(source_mask) as handle:
        if handle.mode not in ("L", "P", "I", "I;16"):
            raise ValueError(f"Mask must be an indexed single-channel image: {source_mask}")
        original_mask = np.asarray(handle.copy())
    source_classes = validate_mask(original_mask, source_mask)
    # Palette-mode semantic PNGs store class IDs as palette indices. convert("L")
    # would translate their colors into luminance and silently corrupt labels.
    mask = Image.fromarray(original_mask.astype(np.uint8, copy=False), mode="L")
    if image.size != mask.size:
        raise ValueError(f"Image/mask size mismatch: {source_image} {image.size} vs {mask.size}")
    rng = deterministic_random(seed, split, relative)

    if image.width * target_height == image.height * target_width:
        mode = "resize"
        output_image_object = image.resize((target_width, target_height), Image.Resampling.BILINEAR)
        output_mask_object = mask.resize((target_width, target_height), Image.Resampling.NEAREST)
    else:
        use_crop = split == "train" and rng.random() < crop_probability
        box = None
        if use_crop:
            box, _ = choose_safe_crop(
                original_mask, target_width, target_height, rng, crop_attempts, min_retention
            )
        if box is not None:
            mode = "crop"
            output_image_object = image.crop(box).resize(
                (target_width, target_height), Image.Resampling.BILINEAR
            )
            output_mask_object = mask.crop(box).resize(
                (target_width, target_height), Image.Resampling.NEAREST
            )
        else:
            mode = "crop_fallback_pad" if use_crop else "pad"
            output_image_object, output_mask_object = resize_and_pad(
                image, mask, target_width, target_height
            )

    processed_mask = np.asarray(output_mask_object)
    output_classes = validate_mask(processed_mask, output_mask)
    lost_classes = sorted(source_classes - output_classes)
    if lost_classes and mode == "crop":
        mode = "crop_fallback_pad"
        output_image_object, output_mask_object = resize_and_pad(
            image, mask, target_width, target_height
        )
        processed_mask = np.asarray(output_mask_object)
        output_classes = validate_mask(processed_mask, output_mask)
        lost_classes = sorted(source_classes - output_classes)

    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_mask.parent.mkdir(parents=True, exist_ok=True)
    temporary_image = output_image.with_name(f".{output_image.stem}.{os.getpid()}.tmp.jpg")
    temporary_mask = output_mask.with_name(f".{output_mask.stem}.{os.getpid()}.tmp.png")
    try:
        output_image_object.save(temporary_image, "JPEG", quality=quality, optimize=False)
        output_mask_object.save(temporary_mask, "PNG", optimize=False)
        os.replace(temporary_image, output_image)
        os.replace(temporary_mask, output_mask)
    finally:
        temporary_image.unlink(missing_ok=True)
        temporary_mask.unlink(missing_ok=True)

    values, counts = np.unique(processed_mask, return_counts=True)
    return {
        "split": split, "relative": str(relative), "mode": mode,
        "source_width": image.width, "source_height": image.height,
        "class_pixels": {str(int(value)): int(count) for value, count in zip(values, counts)},
        "lost_classes_after_resize": lost_classes,
    }


def main(argv=None):
    args = parse_args(argv)
    source_root, output_root = args.source_root.resolve(), args.output_root.resolve()
    jobs = []
    split_counts = {}
    for split in ("train", "val"):
        pairs = discover_pairs(source_root, split, args.max_images_per_split)
        split_counts[split] = len(pairs)
        jobs.extend(
            (
                str(image), str(mask), str(relative), split, str(output_root),
                args.width, args.height, args.crop_probability,
                args.min_foreground_retention, args.crop_attempts, args.seed,
                args.jpeg_quality, args.resume,
            )
            for image, mask, relative in pairs
        )
    output_root.mkdir(parents=True, exist_ok=True)
    mode_counts = {"train": Counter(), "val": Counter()}
    pixels = {"train": Counter(), "val": Counter()}
    source_dimensions = Counter()
    lost_class_examples = []
    log_path = output_root / "preparation_records.jsonl"

    with log_path.open("w", encoding="utf-8") as log:
        if args.workers == 1:
            results = map(process_one, jobs)
            executor = None
        else:
            executor = ProcessPoolExecutor(max_workers=args.workers)
            results = executor.map(process_one, jobs, chunksize=16)
        try:
            for index, record in enumerate(results, 1):
                split = record["split"]
                mode_counts[split][record["mode"]] += 1
                pixels[split].update({int(key): value for key, value in record["class_pixels"].items()})
                source_dimensions[(record["source_width"], record["source_height"])] += 1
                if record.get("lost_classes_after_resize") and len(lost_class_examples) < 50:
                    lost_class_examples.append({
                        "relative": record["relative"], "classes": record["lost_classes_after_resize"]
                    })
                log.write(json.dumps(record) + "\n")
                if index == len(jobs) or index % 500 == 0:
                    print(f"Prepared {index}/{len(jobs)} static 16:9 image/mask pairs", flush=True)
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)

    manifest = {
        "format": "prepared_static_semantic_16x9_v1",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "input_width": args.width,
        "input_height": args.height,
        "aspect_ratio": "16:9",
        "ignore_index": 255,
        "class_names": CLASS_NAMES,
        "crop_probability": args.crop_probability,
        "validation_policy": "preserve_full_image_with_letterbox_padding",
        "seed": args.seed,
        "splits": {
            split: {
                "images": split_counts[split],
                "mode_counts": dict(mode_counts[split]),
                "pixel_counts": {name: int(pixels[split][index]) for index, name in enumerate(CLASS_NAMES)},
                "ignore_pixels": int(pixels[split][255]),
            }
            for split in ("train", "val")
        },
        "top_source_dimensions": [
            {"width": width, "height": height, "count": count}
            for (width, height), count in source_dimensions.most_common(20)
        ],
        "lost_classes_after_resize_examples": lost_class_examples,
    }
    manifest_path = output_root / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Saved prepared static dataset manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
