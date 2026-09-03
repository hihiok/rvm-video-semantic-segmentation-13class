#!/usr/bin/env python3
"""Build read-only partial-label manifests for 8-label FSD scene tagging.

Sources are never modified or copied. Output is JSONL manifests referencing the
original absolute image paths. Labels use {1 positive, 0 negative, -1 unknown}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from config import (
    DISPLAY_NAMES, LABELS, LANDSCAPE_NEG, LANDSCAPE_POS, OFFICE_NEG, OFFICE_POS,
    RAIN_SNOW_PLACES_NEG, RAIN_SNOW_PLACES_POS, SEG_CLASS_IDS, SPORTS_NEG,
    SPORTS_POS, TEN_SCENES_ALIASES,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--places-root", type=Path, required=True)
    p.add_argument("--coco-root", type=Path, required=True)
    p.add_argument("--seg-root", type=Path, required=True)
    p.add_argument("--ten-scenes-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--places-train-cap-per-class", type=int, default=500)
    p.add_argument("--coco-objective-neg-train-cap", type=int, default=5000)
    p.add_argument("--coco-objective-neg-eval-cap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260903)
    p.add_argument("--snow-min-area", type=float, default=0.01)
    p.add_argument("--landscape-min-area", type=float, default=0.30)
    return p.parse_args()


def empty_labels():
    return {k: -1 for k in LABELS}


def stable_bucket(key: str, train_pct=80, val_pct=10):
    v = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    if v < train_pct:
        return "train"
    if v < train_pct + val_pct:
        return "val"
    return "test"


def stable_eval_half(key: str):
    v = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) & 1
    return "val" if v == 0 else "test"


def norm_folder(name: str):
    s = name.strip().lower().replace("&", "and")
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def canonical_flat_key(name: str):
    return norm_folder(name)


def load_places_io_map(repo_root: Path):
    path = repo_root / "nas_scene_multilabel" / "metadata" / "places365_io_map.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Missing repository Places365 IO metadata: {path}")
    canonical_to_io = {}
    flat_to_canonical = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        raw, value = line.rsplit("\t", 1)
        raw = raw.strip().lstrip("/")
        parts = raw.split("/")
        if len(parts) >= 2 and len(parts[0]) == 1:
            canonical = "/".join(parts[1:])
        else:
            canonical = raw
        canonical_to_io[canonical] = int(value)
        flat = canonical.replace("/", "_").replace("_", "-")
        flat_to_canonical[canonical_flat_key(flat)] = canonical
        flat_to_canonical[canonical_flat_key(canonical.replace("/", "-"))] = canonical
        flat_to_canonical[canonical_flat_key(canonical)] = canonical
    return canonical_to_io, flat_to_canonical


def locate_places(root: Path):
    candidates = [root / "versions" / "1", root]
    for base in candidates:
        train = base / "train"
        val = base / "val"
        if train.is_dir() and val.is_dir():
            return train, val
    raise FileNotFoundError(
        f"Places365 layout not found under {root}; expected versions/1/train and versions/1/val")


def places_labels(cat: str, io_map):
    y = empty_labels()
    io = io_map.get(cat)
    if io == 1:
        y["indoor"], y["outdoor"] = 1, 0
    elif io == 2:
        y["indoor"], y["outdoor"] = 0, 1
    if cat in LANDSCAPE_POS:
        y["landscape"] = 1
    elif cat in LANDSCAPE_NEG:
        y["landscape"] = 0
    if cat in SPORTS_POS:
        y["sports"] = 1
    elif cat in SPORTS_NEG:
        y["sports"] = 0
    if cat in OFFICE_POS:
        y["office"] = 1
    elif cat in OFFICE_NEG:
        y["office"] = 0
    if cat in RAIN_SNOW_PLACES_POS:
        y["rain_snow"] = 1
    elif cat in RAIN_SNOW_PLACES_NEG:
        y["rain_snow"] = 0
    # Places365 consists of photographic scenes, so it is a safe negative for
    # computer-synthesized objective/test-pattern imagery.
    y["objective_image"] = 0
    return y


def add_places(records, root: Path, repo_root: Path, cap: int, rng: random.Random, audit):
    train_root, val_root = locate_places(root)
    io_map, flat_map = load_places_io_map(repo_root)
    unknown = []
    for split_root, source_split in [(train_root, "train"), (val_root, "eval")]:
        for folder in sorted(p for p in split_root.iterdir() if p.is_dir()):
            key = canonical_flat_key(folder.name)
            cat = flat_map.get(key)
            if cat is None:
                unknown.append(folder.name)
                continue
            imgs = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
            if source_split == "train" and len(imgs) > cap:
                imgs = rng.sample(imgs, cap)
            y = places_labels(cat, io_map)
            for p in imgs:
                out_split = "train" if source_split == "train" else stable_eval_half(p.name)
                records[out_split].append({
                    "image": str(p.resolve()), "labels": y, "source": "places365",
                    "detail": cat,
                })
    audit["places_unknown_folders"] = sorted(set(unknown))
    if unknown:
        raise RuntimeError("Unrecognized Places365 category folders: " + ", ".join(sorted(set(unknown))))


def ten_scene_folder_labels(folder_name: str):
    key = norm_folder(folder_name)
    matched = [label for label, aliases in TEN_SCENES_ALIASES.items() if key in aliases]
    y = empty_labels()
    if len(matched) > 1:
        raise RuntimeError(f"10_scenes folder maps to multiple labels: {folder_name}: {matched}")
    if matched:
        label = matched[0]
        y[label] = 1
        # Safe hierarchy/contrast labels.
        if label == "office":
            y["indoor"], y["outdoor"] = 1, 0
        elif label == "indoor":
            y["outdoor"] = 0
        elif label == "outdoor":
            y["indoor"] = 0
        elif label == "landscape":
            y["outdoor"] = 1
        elif label == "objective_image":
            for other in LABELS:
                if other != "objective_image":
                    y[other] = 0
        if label != "objective_image":
            y["objective_image"] = 0
    return matched[0] if matched else None, y


def iter_ten_scene_category_dirs(root: Path):
    # Prefer explicit split dirs when present, but collect all category dirs and
    # perform our own stable 80/10/10 split so classes that exist only in train
    # (notably Computer_synthesized) still get validation/test coverage.
    bases = [p for p in [root / "train", root / "val", root / "test"] if p.is_dir()]
    if not bases:
        bases = [root]
    out = []
    for base in bases:
        for d in sorted(p for p in base.iterdir() if p.is_dir()):
            out.append(d)
    return out


def add_ten_scenes(records, root: Path, audit):
    folder_map = {}
    unmapped = []
    for folder in iter_ten_scene_category_dirs(root):
        label, y = ten_scene_folder_labels(folder.name)
        folder_map[str(folder)] = label
        if label is None:
            unmapped.append(folder.name)
            continue
        for p in folder.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
                continue
            split = stable_bucket(str(p.resolve()))
            records[split].append({
                "image": str(p.resolve()), "labels": y, "source": "10_scenes",
                "detail": f"{folder.name}->{label}",
            })
    audit["ten_scenes_folder_mapping"] = folder_map
    audit["ten_scenes_unmapped_folders"] = sorted(set(unmapped))
    # Unmapped folders are not automatically negatives. They are reported so a
    # human can decide whether they correspond to one of the 8 product labels.


def find_coco_images(root: Path, split: str):
    candidates = [root / f"{split}2017", root / "images" / f"{split}2017"]
    for d in candidates:
        if d.is_dir():
            return d
    hits = list(root.glob(f"**/{split}2017"))
    for h in hits:
        if h.is_dir() and "annotation" not in str(h).lower():
            return h
    raise FileNotFoundError(f"Cannot locate COCO {split}2017 image dir under {root}")


def add_coco_objective_negatives(records, root: Path, train_cap: int, eval_cap: int, rng: random.Random):
    for src_split in ("train", "val"):
        d = find_coco_images(root, src_split)
        imgs = [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
        if src_split == "train":
            if len(imgs) > train_cap:
                imgs = rng.sample(imgs, train_cap)
            for p in imgs:
                y = empty_labels(); y["objective_image"] = 0
                records["train"].append({"image": str(p.resolve()), "labels": y, "source": "coco2017", "detail": "photo_negative_for_objective"})
        else:
            by_split = {"val": [], "test": []}
            for p in imgs:
                by_split[stable_eval_half(p.name)].append(p)
            for out_split in ("val", "test"):
                selected = by_split[out_split]
                if len(selected) > eval_cap:
                    selected = rng.sample(selected, eval_cap)
                for p in selected:
                    y = empty_labels(); y["objective_image"] = 0
                    records[out_split].append({"image": str(p.resolve()), "labels": y, "source": "coco2017", "detail": "photo_negative_for_objective"})


def resolve_seg_dirs(root: Path, split: str):
    for a, b in [
        (root / "images" / split, root / "annotations" / split),
        (root / split / "images", root / split / "annotations"),
    ]:
        if a.is_dir() and b.is_dir():
            return a, b
    raise FileNotFoundError(f"Cannot resolve segmentation split={split} under {root}")


def index_images(root: Path):
    idx = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            rel = str(p.relative_to(root).with_suffix("")).replace("\\", "/")
            idx[rel] = p
            idx.setdefault(p.stem, p)
    return idx


def labels_from_seg(mask_path: Path, snow_min_area: float, landscape_min_area: float):
    arr = np.asarray(Image.open(mask_path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    valid = arr != 255
    denom = max(int(valid.sum()), 1)
    y = empty_labels()
    snow_id = SEG_CLASS_IDS["ice_or_snow"]
    snow_count = int(((arr == snow_id) & valid).sum())
    snow_ratio = snow_count / denom
    if snow_ratio >= snow_min_area:
        y["rain_snow"] = 1
    elif snow_count == 0:
        y["rain_snow"] = 0

    natural_ids = [SEG_CLASS_IDS[k] for k in ("plant", "water", "desert", "ice_or_snow", "mountain")]
    natural_count = int(np.isin(arr, natural_ids).astype(np.uint8)[valid].sum())
    natural_ratio = natural_count / denom
    if natural_ratio >= landscape_min_area:
        y["landscape"] = 1
    # Keep ambiguous/negative landscape as unknown; Places365 supplies clean negatives.
    y["objective_image"] = 0
    return y, snow_ratio, natural_ratio


def add_seg(records, root: Path, snow_min_area: float, landscape_min_area: float):
    for src_split in ("train", "val"):
        image_root, mask_root = resolve_seg_dirs(root, src_split)
        idx = index_images(image_root)
        for m in mask_root.rglob("*"):
            if not m.is_file() or m.suffix.lower() not in IMG_EXTS:
                continue
            rel = str(m.relative_to(mask_root).with_suffix("")).replace("\\", "/")
            p = idx.get(rel) or idx.get(m.stem)
            if p is None:
                continue
            try:
                y, sr, lr = labels_from_seg(m, snow_min_area, landscape_min_area)
            except Exception:
                continue
            out_split = "train" if src_split == "train" else stable_eval_half(p.name)
            records[out_split].append({
                "image": str(p.resolve()), "labels": y, "source": "seg13",
                "detail": f"mask={m.resolve()};snow_ratio={sr:.6f};natural_ratio={lr:.6f}",
            })


def summarize(items):
    counts = {l: {"pos": 0, "neg": 0, "unknown": 0} for l in LABELS}
    by_source = defaultdict(int)
    for r in items:
        by_source[r["source"]] += 1
        for l, v in r["labels"].items():
            counts[l]["pos" if v == 1 else "neg" if v == 0 else "unknown"] += 1
    return {"num_records": len(items), "by_source": dict(by_source), "labels": counts}


def main():
    args = parse_args()
    for p in [args.places_root, args.coco_root, args.seg_root, args.ten_scenes_root]:
        if not p.is_dir():
            raise FileNotFoundError(p)
    rng = random.Random(args.seed)
    records = {"train": [], "val": [], "test": []}
    audit = {}
    repo_root = Path(__file__).resolve().parents[1]

    add_places(records, args.places_root, repo_root, args.places_train_cap_per_class, rng, audit)
    add_ten_scenes(records, args.ten_scenes_root, audit)
    add_coco_objective_negatives(records, args.coco_root, args.coco_objective_neg_train_cap, args.coco_objective_neg_eval_cap, rng)
    add_seg(records, args.seg_root, args.snow_min_area, args.landscape_min_area)

    args.output_root.mkdir(parents=True, exist_ok=True)
    report = {"labels": LABELS, "display_names": DISPLAY_NAMES, "splits": {}, "audit": audit}
    for split, items in records.items():
        rng.shuffle(items)
        with (args.output_root / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        report["splits"][split] = summarize(items)

    errors = []
    for split in ("train", "val", "test"):
        for label in LABELS:
            c = report["splits"][split]["labels"][label]
            if c["pos"] == 0 or c["neg"] == 0:
                errors.append(f"{split}:{label}:pos={c['pos']},neg={c['neg']}")
    report["coverage_errors"] = errors
    (args.output_root / "dataset_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_root / "source_folder_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit("Incomplete 8-label supervision coverage:\n" + "\n".join(errors))


if __name__ == "__main__":
    main()
