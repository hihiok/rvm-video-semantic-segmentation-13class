#!/usr/bin/env python3
"""Convert official VSPW origin/mask pairs into the filtered 13-class dataset."""

import argparse
import errno
import json
import os
import re
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
try:
    from tqdm import tqdm
except ImportError:  # Keep conversion usable in minimal data-preparation environments.
    def tqdm(iterable, **_kwargs):
        return iterable

from video_label_mapping import convert_mask, load_mapping


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_OUTPUT_ROOT = Path("/data/pub1/z00919662/segmentation/datasets/VSPW_13cls")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vspw-root", type=Path, required=True,
        help="Official root containing data/<video>/origin, data/<video>/mask and split txt files",
    )
    parser.add_argument(
        "--categories-json", type=Path, required=True,
        help="panoVIPSeg_categories.json; VSPW uses the same ordered 124 semantic names",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--mapping", type=Path,
        default=Path(__file__).parents[1] / "configs/vipseg_to_13class.json",
    )
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument(
        "--copy-mode", choices=("hardlink", "copy", "symlink"), default="hardlink",
        help="How retained source images are placed in the output; masks are always newly encoded PNGs",
    )
    parser.add_argument(
        "--minimum-target-pixels", type=int, default=1,
        help="Keep a frame only when at least this many pixels map to foreground target IDs 1..12",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Remove only this converter's images/annotations/metadata outputs and rebuild them",
    )
    return parser.parse_args(argv)


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def discover_pairs(vspw_root, video):
    video_root = Path(vspw_root) / "data" / video
    origin_root, mask_root = video_root / "origin", video_root / "mask"
    if not origin_root.is_dir() or not mask_root.is_dir():
        raise FileNotFoundError(f"Missing official origin/mask directories for {video}: {video_root}")

    images = {
        path.stem: path for path in origin_root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    masks = {
        path.stem: path for path in mask_root.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    }
    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        raise RuntimeError(
            f"{video}: unpaired official frames; missing masks={missing_masks[:5]}, "
            f"missing images={missing_images[:5]}"
        )
    if not images:
        raise RuntimeError(f"{video}: no image/mask pairs found")
    return [(images[stem], masks[stem]) for stem in sorted(images, key=lambda item: natural_key(images[item]))]


def place_image(source, destination, copy_mode):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode == "copy":
        shutil.copy2(source, destination)
    elif copy_mode == "symlink":
        destination.symlink_to(source.resolve())
    else:
        try:
            os.link(source, destination)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            shutil.copy2(source, destination)


def save_mask(mask, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    Image.fromarray(mask, mode="L").save(temporary, format="PNG")
    temporary.replace(destination)


def process_video(task):
    (
        split, video, vspw_root, output_root, raw_to_target, foreground_ids,
        ignore_index, minimum_target_pixels, copy_mode,
    ) = task
    output_root = Path(output_root)
    pixel_counts = Counter()
    records, segment_lengths = [], Counter()
    previous_kept = False
    segment_index = -1
    source_frames = kept_frames = dropped_frames = 0
    source_ignore_alias_253_frames = source_ignore_alias_253_pixels = 0

    for image_source, mask_source in discover_pairs(vspw_root, video):
        source_frames += 1
        with Image.open(mask_source) as mask_file:
            if mask_file.mode not in ("L", "P", "I", "I;16"):
                raise ValueError(f"Mask must be one channel: {mask_source} ({mask_file.mode})")
            raw_mask = np.asarray(mask_file)
        alias_pixels = int(np.count_nonzero(raw_mask == 253))
        source_ignore_alias_253_frames += int(alias_pixels > 0)
        source_ignore_alias_253_pixels += alias_pixels
        with Image.open(image_source) as image_file:
            image_size = image_file.size
        if raw_mask.shape != (image_size[1], image_size[0]):
            raise ValueError(
                f"Official image/mask size mismatch: {image_source} {image_size} vs "
                f"{mask_source} {raw_mask.shape[::-1]}"
            )

        converted = convert_mask(raw_mask, raw_to_target, ignore_index, panoptic=False)
        target_pixels = int(np.isin(converted, foreground_ids).sum())
        keep = target_pixels >= minimum_target_pixels
        if not keep:
            dropped_frames += 1
            previous_kept = False
            records.append((video, image_source.name, "drop", "no_target_1_12", "", "", ""))
            continue

        if not previous_kept:
            segment_index += 1
        previous_kept = True
        kept_frames += 1
        segment_name = f"segment_{segment_index:04d}"
        relative_image = Path(video) / segment_name / image_source.name
        relative_mask = Path(video) / segment_name / f"{image_source.stem}.png"
        image_destination = output_root / "images" / split / relative_image
        mask_destination = output_root / "annotations" / split / relative_mask
        place_image(image_source, image_destination, copy_mode)
        save_mask(converted, mask_destination)
        values, counts = np.unique(converted, return_counts=True)
        pixel_counts.update(dict(zip(values.tolist(), counts.tolist())))
        segment_lengths[segment_name] += 1
        records.append((
            video, image_source.name, "keep", "contains_target_1_12",
            str(Path("images") / split / relative_image),
            str(Path("annotations") / split / relative_mask), segment_name,
        ))

    return {
        "video": video,
        "source_frames": source_frames,
        "kept_frames": kept_frames,
        "dropped_no_target_frames": dropped_frames,
        "source_ignore_alias_253_frames": source_ignore_alias_253_frames,
        "source_ignore_alias_253_pixels": source_ignore_alias_253_pixels,
        "segments": len(segment_lengths),
        "segment_lengths": dict(segment_lengths),
        "pixels": dict(pixel_counts),
        "records": records,
    }


def reset_output(output_root, overwrite):
    output_root = Path(output_root)
    generated = [
        output_root / "images", output_root / "annotations", output_root / "metadata",
        output_root / "dataset_summary.json", output_root / "_SUCCESS",
    ]
    existing = [path for path in generated if path.exists() or path.is_symlink()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output already contains generated data: {existing}. "
            "Audit it or rerun explicitly with --overwrite."
        )
    if overwrite:
        for path in existing:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metadata").mkdir(parents=True, exist_ok=True)


def read_split(vspw_root, split):
    split_file = Path(vspw_root) / f"{split}.txt"
    if not split_file.is_file():
        raise FileNotFoundError(split_file)
    videos = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(videos) != len(set(videos)):
        raise ValueError(f"Duplicate video IDs in {split_file}")
    return videos


def build_dataset(args):
    if not 1 <= args.minimum_target_pixels:
        raise ValueError("--minimum-target-pixels must be at least 1")
    if not 0 <= args.ignore_index <= 255:
        raise ValueError("--ignore-index must fit uint8")
    reset_output(args.output_root, args.overwrite)
    target_classes, raw_to_target, config = load_mapping(args.categories_json, args.mapping)
    if [item["id"] for item in target_classes] != list(range(13)):
        raise ValueError("This converter requires exactly the fixed target IDs 0..12")
    foreground_ids = tuple(item["id"] for item in target_classes if item["id"] != 0)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    if not splits:
        raise ValueError("No splits selected")

    summary = {
        "classes": target_classes,
        "mapping_notes": config.get("mapping_notes", {}),
        "filter": {
            "keep": "mapped mask contains at least one target class ID in 1..12",
            "drop": "mapped mask contains only background=0 and/or ignore=255",
            "minimum_target_pixels": args.minimum_target_pixels,
        },
        "layout": "images|annotations/<split>/<video>/segment_NNNN/<frame>",
        "copy_mode": args.copy_mode,
        "source_ignore_aliases": {"253": args.ignore_index},
        "splits": {},
    }
    all_split_videos = {}
    for split in splits:
        videos = read_split(args.vspw_root, split)
        overlap = set(videos) & set().union(*(set(value) for value in all_split_videos.values()))
        if overlap:
            raise RuntimeError(f"Official split leakage for {split}: {sorted(overlap)[:10]}")
        all_split_videos[split] = videos
        tasks = [(
            split, video, str(args.vspw_root), str(args.output_root), raw_to_target,
            foreground_ids, args.ignore_index, args.minimum_target_pixels, args.copy_mode,
        ) for video in videos]
        split_counts = Counter()
        videos_with_data = segments = 0
        metadata_path = args.output_root / "metadata" / f"frame_filter_{split}.tsv"
        with metadata_path.open("w", encoding="utf-8") as metadata:
            metadata.write(
                "video\tsource_frame\tdecision\treason\toutput_image\toutput_mask\tsegment\n"
            )
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                results = executor.map(process_video, tasks, chunksize=1)
                for result in tqdm(results, total=len(tasks), desc=f"convert {split}"):
                    split_counts["source_frames"] += result["source_frames"]
                    split_counts["kept_frames"] += result["kept_frames"]
                    split_counts["dropped_no_target_frames"] += result["dropped_no_target_frames"]
                    split_counts["source_ignore_alias_253_frames"] += result[
                        "source_ignore_alias_253_frames"
                    ]
                    split_counts["source_ignore_alias_253_pixels"] += result[
                        "source_ignore_alias_253_pixels"
                    ]
                    segments += result["segments"]
                    videos_with_data += int(result["kept_frames"] > 0)
                    for class_id, count in result["pixels"].items():
                        split_counts[f"pixels_{class_id}"] += count
                    for record in result["records"]:
                        metadata.write("\t".join(map(str, record)) + "\n")
        summary["splits"][split] = {
            "source_videos": len(videos),
            "videos_with_kept_frames": videos_with_data,
            "segments": segments,
            "source_frames": split_counts["source_frames"],
            "kept_frames": split_counts["kept_frames"],
            "dropped_no_target_frames": split_counts["dropped_no_target_frames"],
            "source_ignore_alias_253_frames": split_counts[
                "source_ignore_alias_253_frames"
            ],
            "source_ignore_alias_253_pixels": split_counts[
                "source_ignore_alias_253_pixels"
            ],
            "keep_rate": (
                split_counts["kept_frames"] / split_counts["source_frames"]
                if split_counts["source_frames"] else 0.0
            ),
            "pixels": {
                str(class_id): int(split_counts[f"pixels_{class_id}"])
                for class_id in list(range(13)) + [args.ignore_index]
            },
        }

    (args.output_root / "metadata" / "class_mapping.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_root / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_root / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    return summary


def main(argv=None):
    args = parse_args(argv)
    summary = build_dataset(args)
    print(json.dumps(summary["splits"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
