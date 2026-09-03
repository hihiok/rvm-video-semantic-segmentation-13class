#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from config import (
    COCO_ANIMAL, COCO_FOOD, DISPLAY_NAMES, LABELS, LANDSCAPE_NEG, LANDSCAPE_POS,
    OFFICE_NEG, OFFICE_POS, SEG_CLASS_IDS, SPORTS_NEG, SPORTS_POS,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BUNDLED_PLACES_IO = Path(__file__).resolve().parent / "metadata" / "IO_places365.txt"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--places-root", type=Path, required=True)
    p.add_argument("--coco-root", type=Path, required=True)
    p.add_argument("--seg-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--places-train-cap-per-class", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260902)
    p.add_argument("--sky-min-area", type=float, default=0.01)
    p.add_argument("--building-min-area", type=float, default=0.02)
    p.add_argument("--food-min-area", type=float, default=0.01)
    return p.parse_args()


def stable_half(key: str) -> str:
    h = hashlib.sha1(key.encode("utf-8")).digest()[0]
    return "val" if h < 128 else "test"


def empty_labels():
    return {k: -1 for k in LABELS}


def norm_cat(raw: str) -> str:
    raw = raw.strip().replace("\\", "/").strip("/").lower()
    parts = raw.split("/")
    if len(parts) >= 2 and len(parts[0]) == 1:
        return "/".join(parts[1:])
    return raw


def find_one(root: Path, names):
    for name in names:
        p = root / name
        if p.exists():
            return p
    for name in names:
        hits = list(root.glob(f"**/{name}"))
        if hits:
            return hits[0]
    return None


def parse_io(path: Path):
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            a, b = line.strip().rsplit(" ", 1)
            out[norm_cat(a)] = int(b)  # 1 indoor, 2 outdoor
    return out


def flat_places_alias(cat: str) -> str:
    """Map official names such as field/cultivated to folder-style field-cultivated."""
    return norm_cat(cat).replace("/", "-").replace("_", "-")


def build_places_alias_map(io_map):
    aliases = {}
    collisions = defaultdict(set)
    for canonical in io_map:
        candidates = {
            canonical,
            canonical.replace("/", "-"),
            canonical.replace("_", "-"),
            flat_places_alias(canonical),
        }
        for alias in candidates:
            alias = norm_cat(alias)
            if alias in aliases and aliases[alias] != canonical:
                collisions[alias].update({aliases[alias], canonical})
            else:
                aliases[alias] = canonical
    if collisions:
        raise RuntimeError(f"Ambiguous Places365 folder aliases: {dict(collisions)}")
    return aliases


def canonicalize_places_category(raw: str, alias_map):
    raw = norm_cat(raw)
    candidates = [
        raw,
        raw.replace("_", "-"),
        raw.replace("/", "-"),
        raw.replace("/", "-").replace("_", "-"),
    ]
    for x in candidates:
        if x in alias_map:
            return alias_map[x]
    return None


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
    return y


def locate_places_dirs(root: Path):
    """Resolve both standard Places365 and the local versions/1 folder layout.

    Supported examples:
      root/versions/1/train/<flattened-category>/*.jpg
      root/versions/1/val/<flattened-category>/*.jpg
      root/train/<category>/*.jpg
      root/val/<category>/*.jpg
      root/data_256/... and root/val_256/...

    Local categories_places365.txt and IO_places365.txt are not required.  The
    official indoor/outdoor taxonomy is bundled with this repository.
    """
    pairs = [
        (root / "versions" / "1" / "train", root / "versions" / "1" / "val"),
        (root / "train", root / "val"),
        (root / "data_256", root / "val_256"),
    ]
    train = val = None
    for tr, va in pairs:
        if tr.is_dir() and va.is_dir():
            train, val = tr, va
            break
    if train is None:
        train = find_one(root, ["data_256", "train"])
        val = find_one(root, ["val_256", "val"])

    io_file = find_one(root, ["IO_places365.txt"])
    if io_file is None:
        io_file = BUNDLED_PLACES_IO

    missing = []
    if train is None or not train.is_dir():
        missing.append("train")
    if val is None or not val.is_dir():
        missing.append("val")
    if not io_file.is_file():
        missing.append(f"IO taxonomy ({io_file})")
    if missing:
        raise FileNotFoundError(f"Places365 missing: {missing}; root={root}")
    return train, val, io_file


def iter_places_category_dirs(split_root: Path, alias_map):
    """Yield (canonical_category, dir, image_files) for any folder containing images."""
    unknown = []
    seen = set()
    for dirpath, _, filenames in os.walk(split_root):
        image_names = [x for x in filenames if Path(x).suffix.lower() in IMG_EXTS]
        if not image_names:
            continue
        d = Path(dirpath)
        rel = d.relative_to(split_root)
        raw = norm_cat(str(rel))
        cat = canonicalize_places_category(raw, alias_map)
        if cat is None:
            # Flat local layout normally needs only the leaf folder name.
            cat = canonicalize_places_category(d.name, alias_map)
        if cat is None:
            unknown.append(str(rel))
            continue
        key = (cat, str(d.resolve()))
        if key in seen:
            continue
        seen.add(key)
        yield cat, d, [d / x for x in image_names]
    if unknown:
        preview = sorted(set(unknown))[:50]
        raise RuntimeError(
            f"Unrecognized Places365 category folders ({len(set(unknown))} total), "
            f"first entries={preview}. Do not guess category mappings."
        )


def add_places(records, root: Path, cap: int, rng: random.Random):
    train_root, val_root, io_file = locate_places_dirs(root)
    io_map = parse_io(io_file)
    alias_map = build_places_alias_map(io_map)

    print(f"PLACES_TRAIN_ROOT={train_root}")
    print(f"PLACES_VAL_ROOT={val_root}")
    print(f"PLACES_IO_SOURCE={io_file}")

    train_categories = 0
    for cat, _, imgs in iter_places_category_dirs(train_root, alias_map):
        train_categories += 1
        if len(imgs) > cap:
            imgs = rng.sample(imgs, cap)
        y = places_labels(cat, io_map)
        if all(v < 0 for v in y.values()):
            continue
        for p in imgs:
            records["train"].append({
                "image": str(p.resolve()), "labels": dict(y),
                "source": "places365", "detail": cat,
            })

    val_categories = 0
    for cat, _, imgs in iter_places_category_dirs(val_root, alias_map):
        val_categories += 1
        y = places_labels(cat, io_map)
        if all(v < 0 for v in y.values()):
            continue
        for p in imgs:
            rel_key = str(p.relative_to(val_root)).replace("\\", "/")
            split = stable_half(rel_key)
            records[split].append({
                "image": str(p.resolve()), "labels": dict(y),
                "source": "places365", "detail": cat,
            })

    print(f"PLACES_RECOGNIZED_TRAIN_CATEGORY_DIRS={train_categories}")
    print(f"PLACES_RECOGNIZED_VAL_CATEGORY_DIRS={val_categories}")
    if train_categories == 0 or val_categories == 0:
        raise RuntimeError(
            f"No Places365 category directories recognized: train={train_categories}, val={val_categories}"
        )


def coco_split(root: Path, split: str):
    ann = root / "annotations" / f"instances_{split}2017.json"
    img = root / f"{split}2017"
    if not img.is_dir():
        img = root / "images" / f"{split}2017"
    if not ann.exists() or not img.is_dir():
        raise FileNotFoundError(f"COCO {split} missing: ann={ann}, images={img}")
    return ann, img


def add_coco(records, root: Path):
    for split_name in ("train", "val"):
        ann_path, image_root = coco_split(root, split_name)
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in data["categories"]}
        by_image = defaultdict(set)
        for a in data["annotations"]:
            if not a.get("iscrowd", 0):
                by_image[a["image_id"]].add(cats[a["category_id"]])
        for im in data["images"]:
            p = image_root / im["file_name"]
            if not p.exists():
                continue
            names = by_image.get(im["id"], set())
            y = empty_labels()
            y["food"] = int(bool(names & COCO_FOOD))
            y["animal"] = int(bool(names & COCO_ANIMAL))
            out_split = "train" if split_name == "train" else stable_half(im["file_name"])
            records[out_split].append({
                "image": str(p.resolve()), "labels": y, "source": "coco2017",
                "detail": ",".join(sorted(names)),
            })


def resolve_seg_dirs(root: Path, split: str):
    pairs = [
        (root / "images" / split, root / "annotations" / split),
        (root / split / "images", root / split / "annotations"),
    ]
    for a, b in pairs:
        if a.is_dir() and b.is_dir():
            return a, b
    raise FileNotFoundError(f"Cannot resolve segmentation {split} under {root}")


def index_images(root: Path):
    idx = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            rel = str(p.relative_to(root).with_suffix("")).replace("\\", "/")
            idx[rel] = p
            idx.setdefault(p.stem, p)
    return idx


def seg_labels(mask_path: Path, thresholds):
    arr = np.asarray(Image.open(mask_path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    valid = arr != 255
    denom = max(int(valid.sum()), 1)
    y = empty_labels()
    for label, class_id in SEG_CLASS_IDS.items():
        count = int(((arr == class_id) & valid).sum())
        frac = count / denom
        if frac >= thresholds[label]:
            y[label] = 1
        elif count == 0:
            y[label] = 0
        else:
            y[label] = -1
    return y


def add_seg(records, root: Path, thresholds):
    for split_name in ("train", "val"):
        image_root, mask_root = resolve_seg_dirs(root, split_name)
        idx = index_images(image_root)
        for m in mask_root.rglob("*"):
            if not m.is_file() or m.suffix.lower() not in IMG_EXTS:
                continue
            rel = str(m.relative_to(mask_root).with_suffix("")).replace("\\", "/")
            p = idx.get(rel) or idx.get(m.stem)
            if p is None:
                continue
            try:
                y = seg_labels(m, thresholds)
            except Exception:
                continue
            out_split = "train" if split_name == "train" else stable_half(rel)
            records[out_split].append({
                "image": str(p.resolve()), "labels": y, "source": "seg13",
                "detail": str(m.resolve()),
            })


def summarize(items):
    s = {l: {"pos": 0, "neg": 0, "unknown": 0} for l in LABELS}
    by_source = defaultdict(int)
    for r in items:
        by_source[r["source"]] += 1
        for l, v in r["labels"].items():
            s[l]["pos" if v == 1 else "neg" if v == 0 else "unknown"] += 1
    return {"num_images": len(items), "by_source": dict(by_source), "labels": s}


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    records = {"train": [], "val": [], "test": []}

    add_places(records, args.places_root, args.places_train_cap_per_class, rng)
    add_coco(records, args.coco_root)
    add_seg(records, args.seg_root, {
        "sky": args.sky_min_area,
        "building": args.building_min_area,
        "food": args.food_min_area,
    })

    args.output_root.mkdir(parents=True, exist_ok=True)
    report = {"labels": LABELS, "display_names": DISPLAY_NAMES, "splits": {}}
    for split, items in records.items():
        rng.shuffle(items)
        out = args.output_root / f"{split}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        report["splits"][split] = summarize(items)

    # Hard coverage guard: every label needs positive and negative supervision in train/val/test.
    errors = []
    for split in ("train", "val", "test"):
        for l in LABELS:
            c = report["splits"][split]["labels"][l]
            if c["pos"] == 0 or c["neg"] == 0:
                errors.append(f"{split}:{l}:pos={c['pos']},neg={c['neg']}")
    report["coverage_errors"] = errors
    (args.output_root / "dataset_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit("Incomplete supervision coverage:\n" + "\n".join(errors))


if __name__ == "__main__":
    main()
