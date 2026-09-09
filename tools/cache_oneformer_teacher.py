#!/usr/bin/env python3
"""Cache compact OneFormer soft targets for one image tree."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.video_semantic import IMAGE_EXTENSIONS
from distillation.cache import cache_path_for_image, save_cached_probabilities
from distillation.oneformer_teacher import OneFormer13ClassTeacher, load_mapping


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, default=Path("configs/oneformer_ade20k_to_13class.json"))
    parser.add_argument("--model", default="shi-labs/oneformer_ade20k_swin_large")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cache-height", type=int, default=45)
    parser.add_argument("--cache-width", type=int, default=80)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main():
    args = parse_args()
    if args.batch_size < 1 or args.cache_height < 1 or args.cache_width < 1:
        raise ValueError("Batch and cache dimensions must be positive")
    image_root = args.image_root.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    images = sorted(
        item for item in image_root.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )
    if args.max_images > 0:
        images = images[: args.max_images]
    if not images:
        raise RuntimeError(f"No images found under {image_root}")
    mapping_bytes = args.mapping.read_bytes()
    mapping = load_mapping(args.mapping)
    mapping_hash = hashlib.sha256(mapping_bytes).hexdigest()
    manifest_path = cache_root / "MANIFEST.json"
    identity = {
        "format": "oneformer_rvm13_uint8_probability_cache_v1",
        "teacher_model": args.model,
        "mapping_sha256": mapping_hash,
        "class_names": mapping["target_class_names"],
        "unsupported_target_ids": mapping["unsupported_target_ids"],
        "image_root": str(image_root),
        "cache_shape": [13, args.cache_height, args.cache_width],
    }
    existing_entries = any(cache_root.rglob("*.npz")) if cache_root.is_dir() else False
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = {
            key: {"expected": value, "actual": previous.get(key)}
            for key, value in identity.items() if previous.get(key) != value
        }
        if mismatches and not args.overwrite:
            raise RuntimeError(
                f"Existing cache identity differs from this run: {mismatches}. "
                "Use a new cache directory or deliberately pass --overwrite."
            )
        if mismatches and args.max_images > 0:
            raise RuntimeError("A mapping/model change cannot be overwritten with --max-images")
    elif existing_entries and not args.overwrite:
        raise RuntimeError(
            f"Cache entries exist without a manifest under {cache_root}; pass --overwrite "
            "after confirming that regeneration is intended"
        )

    pending = [
        item for item in images
        if args.overwrite or not cache_path_for_image(item, image_root, cache_root).is_file()
    ]
    cache_root.mkdir(parents=True, exist_ok=True)
    building_manifest = {
        **identity,
        "status": "building",
        "images_discovered": len(images),
        "entries_pending_at_start": len(pending),
    }
    manifest_path.write_text(json.dumps(building_manifest, indent=2) + "\n", encoding="utf-8")
    teacher = None
    written = 0
    for batch_paths in tqdm(list(chunks(pending, args.batch_size)), desc="OneFormer cache", dynamic_ncols=True):
        if teacher is None:
            teacher = OneFormer13ClassTeacher(args.model, args.mapping, args.device, args.amp)
        opened = []
        try:
            for path in batch_paths:
                with Image.open(path) as handle:
                    opened.append(handle.convert("RGB"))
            probabilities = teacher(opened, (args.cache_height, args.cache_width))
            for path, value in zip(batch_paths, probabilities):
                save_cached_probabilities(cache_path_for_image(path, image_root, cache_root), value)
                written += 1
        finally:
            for image in opened:
                image.close()
    manifest = {
        **identity,
        "status": "complete",
        "images_discovered": len(images),
        "entries_written_this_run": written,
        "entries_existing_or_written": len(images),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
