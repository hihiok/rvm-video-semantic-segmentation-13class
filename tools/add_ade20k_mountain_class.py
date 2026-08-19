#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Incrementally upgrade ADE20K_custom12 to 13 classes by adding mountain=12.

This tool does NOT rebuild the dataset. It:
1) validates/reconstructs the original deterministic custom train/val split;
2) patches mountain pixels (ADE source ID 17) into existing masks as target ID 12;
3) optionally recovers ADE-training images that were previously filtered out only
   because mountain was not among the old target classes;
4) updates manifests/metadata/stats and writes an audit report.

Only custom train/val are written. ADE official validation is audited but never mixed
into custom val.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

ADE_MOUNTAIN_ID = 17
TARGET_MOUNTAIN_ID = 12
TARGET_BACKGROUND_ID = 0
SPLITS = ("train", "val")

FALLBACK_MAPPING = {
    3: 1,
    13: 2,
    5: 3, 10: 3, 18: 3, 73: 3,
    2: 4, 26: 4, 49: 4, 80: 4, 85: 4,
    67: 5,
    121: 6,
    22: 7, 27: 7, 61: 7, 110: 7, 114: 7, 129: 7,
    47: 8,
    # 9 ice_or_snow remains empty in ADE20K-150.
    # Refined text mapping: signboard, poster, bulletin board only.
    44: 10, 101: 10, 145: 10,
    120: 11,
}

TARGET_CLASSES = {
    0: "background",
    1: "sky",
    2: "person",
    3: "plant",
    4: "building",
    5: "flower",
    6: "food",
    7: "water",
    8: "desert",
    9: "ice_or_snow",
    10: "text",
    11: "ball",
    12: "mountain",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--ade-root", type=Path, required=True)
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--split-seed", type=int, default=20260730)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--recover-missing-mountain", action="store_true")
    p.add_argument("--backup-dir", type=Path, default=None)
    return p.parse_args()


def detect_mask_root(dataset_root: Path) -> Path:
    for name in ("masks", "annotations"):
        root = dataset_root / name
        if (root / "train").is_dir() and (root / "val").is_dir():
            return root
    raise FileNotFoundError("Expected masks/train+val or annotations/train+val")


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.asarray(im)
    if arr.ndim != 2:
        raise ValueError(f"Expected single-channel mask: {path} shape={arr.shape}")
    return arr.astype(np.uint8, copy=False)


def load_active_mapping(dataset_root: Path) -> Dict[int, int]:
    path = dataset_root / "class_mapping.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload.get("active_ade_id_to_target_id", {})
        if raw:
            mapping = {int(k): int(v) for k, v in raw.items()}
            # Safety: mountain must not already point elsewhere.
            if ADE_MOUNTAIN_ID in mapping and mapping[ADE_MOUNTAIN_ID] != TARGET_MOUNTAIN_ID:
                raise ValueError(f"ADE 17 already mapped to {mapping[ADE_MOUNTAIN_ID]}")
            return mapping
    return dict(FALLBACK_MAPPING)


def build_lut(mapping: Dict[int, int], include_mountain: bool) -> np.ndarray:
    lut = np.zeros(256, dtype=np.uint8)
    for src, dst in mapping.items():
        if src == ADE_MOUNTAIN_ID:
            continue
        lut[src] = dst
    if include_mountain:
        lut[ADE_MOUNTAIN_ID] = TARGET_MOUNTAIN_ID
    return lut


def reconstruct_split(ade_root: Path, train_ratio: float, seed: int) -> Dict[str, str]:
    image_dir = ade_root / "images" / "training"
    files = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not files:
        raise RuntimeError(f"No ADE training images in {image_dir}")

    indices = list(range(len(files)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_train = int(round(len(indices) * train_ratio))
    n_train = max(1, min(n_train, len(indices) - 1))
    train_indices = set(indices[:n_train])
    return {
        p.stem: ("train" if i in train_indices else "val")
        for i, p in enumerate(files)
    }


def backup_file(src: Path, backup_root: Path, dataset_root: Path) -> None:
    if not src.exists() and not src.is_symlink():
        return
    dst = backup_root / src.relative_to(dataset_root)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_symlink():
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def find_source_image(ade_root: Path, stem: str) -> Path:
    candidates = list((ade_root / "images" / "training").glob(stem + ".*"))
    candidates = [p for p in candidates if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one ADE image for {stem}, got {candidates}")
    return candidates[0]


def validate_existing_split(mask_root: Path, split_map: Dict[str, str]) -> None:
    mismatches = []
    for split in SPLITS:
        for p in (mask_root / split).glob("*.png"):
            expected = split_map.get(p.stem)
            if expected is None:
                mismatches.append((p.name, split, "not_in_ADE_training"))
            elif expected != split:
                mismatches.append((p.name, split, expected))
            if len(mismatches) >= 20:
                break
    if mismatches:
        raise RuntimeError(
            "Current custom train/val does not match the original deterministic split. "
            f"Examples: {mismatches}"
        )


def rewrite_manifest(dataset_root: Path, mask_root: Path, split: str, dry_run: bool, backup_dir: Path | None) -> None:
    manifest_path = dataset_root / f"{split}_manifest.json"
    if not manifest_path.is_file():
        return

    old_records = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_stem = {Path(r["mask"]).stem: r for r in old_records}

    for mask_path in sorted((mask_root / split).glob("*.png")):
        mask = load_mask(mask_path)
        unique, counts = np.unique(mask, return_counts=True)
        count_map = {int(k): int(v) for k, v in zip(unique, counts)}
        selected_pixels = int(mask.size - count_map.get(0, 0))
        record = by_stem.get(mask_path.stem, {})
        record.update({
            "image": record.get("image", mask_path.stem + ".jpg"),
            "mask": mask_path.name,
            "width": int(mask.shape[1]),
            "height": int(mask.shape[0]),
            "selected_pixels": selected_pixels,
            "selected_ratio": selected_pixels / float(mask.size),
            "present_classes": [int(x) for x in sorted(unique.tolist()) if int(x) != 0],
        })
        by_stem[mask_path.stem] = record

    records = [by_stem[k] for k in sorted(by_stem)]
    if not dry_run:
        if backup_dir is not None:
            backup_file(manifest_path, backup_dir, dataset_root)
        manifest_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def update_metadata(dataset_root: Path, mapping: Dict[int, int], dry_run: bool, backup_dir: Path | None) -> None:
    classes_path = dataset_root / "classes.txt"
    mapping_path = dataset_root / "class_mapping.json"

    if not dry_run:
        if backup_dir is not None:
            backup_file(classes_path, backup_dir, dataset_root)
            backup_file(mapping_path, backup_dir, dataset_root)
        classes_path.write_text(
            "".join(f"{i}\t{name}\n" for i, name in TARGET_CLASSES.items()),
            encoding="utf-8",
        )

    if mapping_path.is_file():
        payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    else:
        payload = {}

    payload["num_classes_including_background"] = 13
    payload["target_classes"] = {str(k): v for k, v in TARGET_CLASSES.items()}
    active = {str(k): int(v) for k, v in mapping.items() if k != ADE_MOUNTAIN_ID}
    active[str(ADE_MOUNTAIN_ID)] = TARGET_MOUNTAIN_ID
    payload["active_ade_id_to_target_id"] = dict(sorted(active.items(), key=lambda x: int(x[0])))
    source_groups = payload.setdefault("source_groups", {})
    source_groups["12"] = {
        "target_name": "mountain",
        "source_ids": [ADE_MOUNTAIN_ID],
        "source_names": ["mountain/mount"],
    }
    payload["mountain_refinement"] = {
        "ade_source_id": ADE_MOUNTAIN_ID,
        "target_id": TARGET_MOUNTAIN_ID,
        "method": "incremental patch plus recovery of previously filtered ADE-training mountain samples",
    }
    if not dry_run:
        mapping_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def recompute_stats(dataset_root: Path, mask_root: Path) -> dict:
    result = {"splits": {}}
    for split in SPLITS:
        pixels = Counter()
        images = Counter()
        n = 0
        for p in tqdm(sorted((mask_root / split).glob("*.png")), desc=f"stats {split}"):
            arr = load_mask(p)
            unique, counts = np.unique(arr, return_counts=True)
            for cid, count in zip(unique, counts):
                pixels[int(cid)] += int(count)
                images[int(cid)] += 1
            n += 1
        total_pixels = sum(pixels.values())
        result["splits"][split] = {
            "kept_images": n,
            "classes": {
                str(cid): {
                    "name": TARGET_CLASSES[cid],
                    "pixels": int(pixels[cid]),
                    "pixel_ratio": pixels[cid] / total_pixels if total_pixels else 0.0,
                    "images": int(images[cid]),
                    "image_ratio": images[cid] / n if n else 0.0,
                }
                for cid in TARGET_CLASSES
            },
        }
    return result


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    ade_root = args.ade_root.resolve()
    mask_root = detect_mask_root(dataset_root)
    backup_dir = args.backup_dir.resolve() if args.backup_dir else None
    if backup_dir and not args.dry_run:
        backup_dir.mkdir(parents=True, exist_ok=True)

    split_map = reconstruct_split(ade_root, args.train_ratio, args.split_seed)
    validate_existing_split(mask_root, split_map)

    mapping = load_active_mapping(dataset_root)
    old_lut = build_lut(mapping, include_mountain=False)
    new_lut = build_lut(mapping, include_mountain=True)

    report = {
        "dry_run": args.dry_run,
        "recover_missing_mountain": args.recover_missing_mountain,
        "ade_mountain_id": ADE_MOUNTAIN_ID,
        "target_mountain_id": TARGET_MOUNTAIN_ID,
        "existing_mountain_samples_patched": {"train": 0, "val": 0},
        "missing_mountain_samples": {"train": 0, "val": 0},
        "missing_with_no_old_foreground": {"train": 0, "val": 0},
        "missing_below_old_0p001_threshold": {"train": 0, "val": 0},
        "recovered_samples": {"train": 0, "val": 0},
        "mountain_pixels_added": {"train": 0, "val": 0},
        "official_validation_mountain_images_unused": 0,
    }

    existing = {
        split: {p.stem for p in (mask_root / split).glob("*.png")}
        for split in SPLITS
    }

    source_ann_dir = ade_root / "annotations" / "training"
    mountain_source_masks = []
    for p in tqdm(sorted(source_ann_dir.glob("*.png")), desc="audit ADE mountain"):
        source = load_mask(p)
        if np.any(source == ADE_MOUNTAIN_ID):
            mountain_source_masks.append((p, source))

    # Audit official validation, but never insert it into custom train/val.
    val_ann_dir = ade_root / "annotations" / "validation"
    for p in tqdm(sorted(val_ann_dir.glob("*.png")), desc="audit ADE official validation"):
        if np.any(load_mask(p) == ADE_MOUNTAIN_ID):
            report["official_validation_mountain_images_unused"] += 1

    for source_path, source in tqdm(mountain_source_masks, desc="patch/recover mountain"):
        stem = source_path.stem
        split = split_map[stem]
        target_path = mask_root / split / source_path.name
        mountain = source == ADE_MOUNTAIN_ID
        mountain_pixels = int(np.count_nonzero(mountain))

        if stem in existing[split]:
            current = load_mask(target_path)
            if current.shape != source.shape:
                raise ValueError(f"Shape mismatch: {target_path} vs {source_path}")
            if current.max() > 11:
                raise ValueError(f"Pre-patch mask already has ID >11: {target_path}")
            # Since ADE mountain was previously unmapped, every source mountain pixel must be bg=0.
            conflicts = mountain & (current != TARGET_BACKGROUND_ID)
            if np.any(conflicts):
                vals, counts = np.unique(current[conflicts], return_counts=True)
                raise RuntimeError(
                    f"Mountain overlaps existing target classes in {target_path}: "
                    f"{dict(zip(vals.tolist(), counts.tolist()))}"
                )
            patched = current.copy()
            patched[mountain] = TARGET_MOUNTAIN_ID
            # Non-mountain pixels must be bit-identical.
            assert np.array_equal(current[~mountain], patched[~mountain])
            report["existing_mountain_samples_patched"][split] += 1
            report["mountain_pixels_added"][split] += mountain_pixels
            if not args.dry_run:
                if backup_dir is not None:
                    backup_file(target_path, backup_dir, dataset_root)
                Image.fromarray(patched, mode="L").save(target_path)
            continue

        report["missing_mountain_samples"][split] += 1
        old_target = old_lut[source]
        old_fg = int(np.count_nonzero(old_target))
        old_ratio = old_fg / float(old_target.size)
        if old_fg == 0:
            report["missing_with_no_old_foreground"][split] += 1
        if old_ratio < 0.001:
            report["missing_below_old_0p001_threshold"][split] += 1

        if not args.recover_missing_mountain:
            continue

        recovered = new_lut[source]
        assert np.any(recovered == TARGET_MOUNTAIN_ID)
        image_src = find_source_image(ade_root, stem)
        image_dst_dir = dataset_root / "images" / split
        image_dst = image_dst_dir / image_src.name
        report["recovered_samples"][split] += 1
        report["mountain_pixels_added"][split] += mountain_pixels

        if not args.dry_run:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            image_dst_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(recovered, mode="L").save(target_path)
            if image_dst.exists() or image_dst.is_symlink():
                raise FileExistsError(image_dst)
            image_dst.symlink_to(image_src.resolve())

    if not args.dry_run:
        update_metadata(dataset_root, mapping, False, backup_dir)
        for split in SPLITS:
            rewrite_manifest(dataset_root, mask_root, split, False, backup_dir)
        stats = recompute_stats(dataset_root, mask_root)
        stats_path = dataset_root / "split_stats_13class_mountain.json"
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        mountain_lines = ["split,image_path,mask_path,mountain_pixel_count,mountain_pixel_ratio"]
        for split in SPLITS:
            for p in sorted((mask_root / split).glob("*.png")):
                arr = load_mask(p)
                count = int(np.count_nonzero(arr == TARGET_MOUNTAIN_ID))
                if not count:
                    continue
                image_candidates = list((dataset_root / "images" / split).glob(p.stem + ".*"))
                image_path = image_candidates[0] if image_candidates else ""
                mountain_lines.append(
                    f"{split},{image_path},{p},{count},{count / float(arr.size):.8f}"
                )
        (dataset_root / "mountain_candidates.txt").write_text("\n".join(mountain_lines) + "\n", encoding="utf-8")

    report_path = dataset_root / (
        "mountain_addition_dry_run.json" if args.dry_run else "mountain_addition_report.json"
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
