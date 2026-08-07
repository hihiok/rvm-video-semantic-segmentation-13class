#!/usr/bin/env python3
"""Convert VSPW semantic videos (data/<video>/origin+mask) into 13 classes."""

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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vspw-root", type=Path, required=True, help="Contains data/ and train.txt/val.txt/test.txt")
    parser.add_argument("--categories-json", type=Path, required=True, help="panoVIPSeg_categories.json (same 124 semantic names)")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, default=Path(__file__).parents[1] / "configs/vipseg_to_13class.json")
    parser.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def process_task(task):
    image_source, mask_source, image_destination, mask_destination, raw_to_target, ignore_index, overwrite = task
    image_destination, mask_destination = Path(image_destination), Path(mask_destination)
    image_destination.parent.mkdir(parents=True, exist_ok=True)
    mask_destination.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not image_destination.exists():
        shutil.copy2(image_source, image_destination)
    if overwrite or not mask_destination.exists():
        with Image.open(mask_source) as mask_file:
            converted = convert_mask(np.asarray(mask_file), raw_to_target, ignore_index, panoptic=False)
        Image.fromarray(converted, mode="L").save(mask_destination)
    else:
        with Image.open(mask_destination) as mask_file:
            converted = np.asarray(mask_file)
    return np.bincount(converted.reshape(-1), minlength=256).tolist()


def main():
    args = parse_args()
    from concurrent.futures import ProcessPoolExecutor
    target_classes, raw_to_target, config = load_mapping(args.categories_json, args.mapping)
    summary = {"classes": target_classes, "mapping_notes": config.get("mapping_notes", {}), "splits": {}}
    for split in ("train", "val", "test"):
        split_file = args.vspw_root / f"{split}.txt"
        if not split_file.exists():
            if split == "test":
                continue
            raise FileNotFoundError(split_file)
        videos = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        tasks = []
        for video in videos:
            origin = args.vspw_root / "data" / video / "origin"
            masks = args.vspw_root / "data" / video / "mask"
            for image in sorted(origin.iterdir()):
                if not image.is_file():
                    continue
                mask = masks / f"{image.stem}.png"
                if not mask.exists():
                    continue
                tasks.append((
                    str(image), str(mask),
                    str(args.output_root / "images" / split / video / image.name),
                    str(args.output_root / "annotations" / split / video / f"{image.stem}.png"),
                    raw_to_target, args.ignore_index, args.overwrite,
                ))
        pixel_counts = Counter()
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for counts in tqdm(executor.map(process_task, tasks, chunksize=16), total=len(tasks), desc=split):
                pixel_counts.update({i: value for i, value in enumerate(counts) if value})
        summary["splits"][split] = {
            "videos": len(videos), "frames": len(tasks),
            "pixels": {str(i): int(pixel_counts[i]) for i in list(range(len(target_classes))) + [args.ignore_index]},
        }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["splits"], indent=2))


if __name__ == "__main__":
    main()
