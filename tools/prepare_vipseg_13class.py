#!/usr/bin/env python3
"""Convert original VIPSeg panoptic video annotations into 13-class clips."""

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from video_label_mapping import convert_mask, load_mapping


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vipseg-root", type=Path, required=True, help="Contains images/ and panomasks/")
    parser.add_argument("--metadata-root", type=Path, required=True, help="Official repo containing train.txt/val.txt/test.txt and category JSON")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, default=Path(__file__).parents[1] / "configs/vipseg_to_13class.json")
    parser.add_argument("--copy-mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    parser.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def link_or_copy(source, destination, mode, overwrite):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if not overwrite:
            return
        destination.unlink()
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source.resolve())
    else:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)


def frame_pairs(image_dir, mask_dir):
    images = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    mask_by_stem = {path.stem: path for path in mask_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png"}
    pairs = []
    for image in images:
        mask = mask_by_stem.get(image.stem)
        if mask is None:
            raise FileNotFoundError(f"No panomask for {image}")
        pairs.append((image, mask))
    if not pairs:
        raise RuntimeError(f"Empty video directory: {image_dir}")
    return pairs


def process_task(task):
    image_source, mask_source, image_destination, mask_destination, raw_to_target, args_dict = task
    link_or_copy(Path(image_source), Path(image_destination), args_dict["copy_mode"], args_dict["overwrite"])
    mask_destination = Path(mask_destination)
    if args_dict["overwrite"] or not mask_destination.exists():
        with Image.open(mask_source) as mask_file:
            raw_mask = np.asarray(mask_file)
        converted = convert_mask(raw_mask, raw_to_target, args_dict["ignore_index"], panoptic=True)
        mask_destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(converted, mode="L").save(mask_destination)
    else:
        with Image.open(mask_destination) as mask_file:
            converted = np.asarray(mask_file)
    return np.bincount(converted.reshape(-1), minlength=256)[:256].tolist()


def read_split(path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    args = parse_args()
    from concurrent.futures import ProcessPoolExecutor
    images_root = args.vipseg_root / "images"
    masks_root = args.vipseg_root / "panomasks"
    categories_json = args.metadata_root / "panoVIPSeg_categories.json"
    if not images_root.is_dir() or not masks_root.is_dir() or not categories_json.is_file():
        raise FileNotFoundError("Expected VIPSeg images/, panomasks/, and official categories JSON")
    target_classes, raw_to_target, config = load_mapping(categories_json, args.mapping)
    args_dict = dict(copy_mode=args.copy_mode, overwrite=args.overwrite, ignore_index=args.ignore_index)
    all_stats = Counter()
    summary = {"classes": target_classes, "mapping_notes": config.get("mapping_notes", {}), "splits": {}}
    for split in ("train", "val", "test"):
        videos = read_split(args.metadata_root / f"{split}.txt")
        tasks = []
        for video in videos:
            pairs = frame_pairs(images_root / video, masks_root / video)
            for image_source, mask_source in pairs:
                tasks.append((
                    str(image_source),
                    str(mask_source),
                    str(args.output_root / "images" / split / video / image_source.name),
                    str(args.output_root / "annotations" / split / video / f"{image_source.stem}.png"),
                    raw_to_target,
                    args_dict,
                ))
        pixel_counts = Counter()
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for counts in tqdm(executor.map(process_task, tasks, chunksize=16), total=len(tasks), desc=split):
                pixel_counts.update({index: value for index, value in enumerate(counts) if value})
        all_stats.update(pixel_counts)
        summary["splits"][split] = {
            "videos": len(videos),
            "frames": len(tasks),
            "pixels": {str(index): int(pixel_counts[index]) for index in list(range(len(target_classes))) + [args.ignore_index]},
        }
    summary["all_pixels"] = {str(index): int(all_stats[index]) for index in list(range(len(target_classes))) + [args.ignore_index]}
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_root / "class_mapping.json").write_text(args.mapping.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps(summary["splits"], indent=2))


if __name__ == "__main__":
    main()
