#!/usr/bin/env python3
"""Build a partial-label NAS 9-tag zero-shot probe set from existing datasets.

The benchmark deliberately does NOT treat unknown labels as negatives.  Each
source contributes only labels for which it has reliable ground truth:
  * Places365: indoor/outdoor/landscape/sports/office scene labels
  * COCO instances: food/animal (and optional sports-object evidence)
  * Existing 13-class semantic masks: building/sky (plus optional food)

Output labels are {-1, 0, 1}; -1 means unknown and is masked from metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from labels import (
    COCO_ANIMAL,
    COCO_FOOD,
    COCO_SPORTS,
    DISPLAY_NAMES,
    LABELS,
    PLACES_GROUPS,
    SEG_CLASS_IDS,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--places-root", type=Path, default=None)
    p.add_argument("--coco-root", type=Path, default=None)
    p.add_argument("--seg-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n-pos", type=int, default=150, help="Target known positives per label")
    p.add_argument("--n-neg", type=int, default=150, help="Target known negatives per label")
    p.add_argument("--min-pos", type=int, default=50, help="Hard minimum positives per label")
    p.add_argument("--min-neg", type=int, default=50, help="Hard minimum negatives per label")
    p.add_argument("--seed", type=int, default=20260828)
    p.add_argument("--places-per-category-cap", type=int, default=250)
    p.add_argument("--seg-candidate-cap", type=int, default=3000)
    p.add_argument("--sky-min-area", type=float, default=0.01)
    p.add_argument("--building-min-area", type=float, default=0.02)
    p.add_argument("--food-min-area", type=float, default=0.01)
    p.add_argument("--materialize", choices=["symlink", "copy", "none"], default="symlink")
    return p.parse_args()


def norm_rel(path: Path) -> str:
    return str(path).replace("\\", "/").strip("/").lower().replace(" ", "_")


def match_places_category(rel_dir: Path) -> Optional[str]:
    """Return the longest curated Places365 alias matching a directory suffix."""
    s = norm_rel(rel_dir)
    aliases = set().union(*PLACES_GROUPS.values())
    matches = [a for a in aliases if s == a or s.endswith("/" + a)]
    return max(matches, key=len) if matches else None


def sample_images_in_dir(files: Sequence[str], base: Path, cap: int, rng: random.Random) -> List[Path]:
    imgs = [base / f for f in files if Path(f).suffix.lower() in IMAGE_EXTS]
    if len(imgs) > cap:
        imgs = rng.sample(imgs, cap)
    return imgs


def collect_places(root: Path, cap: int, rng: random.Random, cand) -> Dict[str, int]:
    counts = defaultdict(int)
    if root is None or not root.exists():
        return counts

    indoor = PLACES_GROUPS["indoor"]
    outdoor = PLACES_GROUPS["outdoor"]
    landscape = PLACES_GROUPS["landscape"]
    sports = PLACES_GROUPS["sports"]
    office = PLACES_GROUPS["office"]

    for dirpath, dirnames, filenames in os.walk(root):
        if not filenames:
            continue
        d = Path(dirpath)
        try:
            rel = d.relative_to(root)
        except ValueError:
            continue
        cat = match_places_category(rel)
        if cat is None:
            continue
        imgs = sample_images_in_dir(filenames, d, cap, rng)
        if not imgs:
            continue

        for img in imgs:
            ev = f"Places365:{cat}"
            if cat in indoor:
                cand["indoor"][1].append((img, ev))
                cand["outdoor"][0].append((img, ev))
                counts["indoor_pos"] += 1
                counts["outdoor_neg"] += 1
            if cat in outdoor:
                cand["outdoor"][1].append((img, ev))
                cand["indoor"][0].append((img, ev))
                counts["outdoor_pos"] += 1
                counts["indoor_neg"] += 1
            if cat in landscape:
                cand["landscape"][1].append((img, ev))
                counts["landscape_pos"] += 1
            elif cat in indoor:
                cand["landscape"][0].append((img, ev))
                counts["landscape_neg"] += 1
            if cat in sports:
                cand["sports"][1].append((img, ev))
                counts["sports_pos"] += 1
            elif cat in indoor or cat in outdoor:
                cand["sports"][0].append((img, ev))
                counts["sports_neg"] += 1
            if cat in office:
                cand["office"][1].append((img, ev))
                counts["office_pos"] += 1
            elif cat in indoor:
                cand["office"][0].append((img, ev))
                counts["office_neg"] += 1
    return counts


def find_coco_annotation(root: Path) -> Optional[Path]:
    candidates = [
        root / "annotations" / "instances_val2017.json",
        root / "annotations" / "instances_train2017.json",
        root / "instances_val2017.json",
        root / "instances_train2017.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    hits = list(root.glob("**/instances_val2017.json"))
    if not hits:
        hits = list(root.glob("**/instances_train2017.json"))
    return hits[0] if hits else None


def find_coco_image_dir(root: Path, ann_path: Path) -> Optional[Path]:
    split = "val2017" if "val2017" in ann_path.name else "train2017"
    candidates = [root / split, root / "images" / split, ann_path.parent.parent / split, ann_path.parent.parent / "images" / split]
    for p in candidates:
        if p.is_dir():
            return p
    return None


def collect_coco(root: Path, cand) -> Dict[str, int]:
    counts = defaultdict(int)
    if root is None or not root.exists():
        return counts
    ann_path = find_coco_annotation(root)
    if ann_path is None:
        return counts
    image_dir = find_coco_image_dir(root, ann_path)
    if image_dir is None:
        return counts

    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    cat_name = {c["id"]: c["name"] for c in data["categories"]}
    by_image = defaultdict(set)
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        by_image[ann["image_id"]].add(cat_name.get(ann["category_id"], ""))

    for im in data["images"]:
        path = image_dir / im["file_name"]
        if not path.exists():
            continue
        cats = by_image.get(im["id"], set())
        ev = "COCO:" + ",".join(sorted(cats))
        food = bool(cats & COCO_FOOD)
        animal = bool(cats & COCO_ANIMAL)
        sports = bool(cats & COCO_SPORTS)
        cand["food"][1 if food else 0].append((path, ev))
        cand["animal"][1 if animal else 0].append((path, ev))
        # Sports from COCO is used as supplemental evidence.  Product semantics
        # are scene/activity-level, so Places365 remains the main source.
        if sports:
            cand["sports"][1].append((path, ev))
        counts["food_pos" if food else "food_neg"] += 1
        counts["animal_pos" if animal else "animal_neg"] += 1
        if sports:
            counts["sports_pos_coco"] += 1
    return counts


def resolve_seg_dirs(root: Path) -> Tuple[Optional[Path], Optional[Path]]:
    pairs = [
        (root / "images" / "val", root / "annotations" / "val"),
        (root / "images" / "validation", root / "annotations" / "validation"),
        (root / "val" / "images", root / "val" / "annotations"),
        (root / "images" / "test", root / "annotations" / "test"),
    ]
    for i, m in pairs:
        if i.is_dir() and m.is_dir():
            return i, m
    return None, None


def build_stem_index(image_dir: Path) -> Dict[str, Path]:
    out = {}
    for p in image_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            # Preserve relative parent to avoid collisions where possible.
            rel = p.relative_to(image_dir)
            key = norm_rel(rel.with_suffix(""))
            out[key] = p
            out.setdefault(p.stem.lower(), p)
    return out


def find_image_for_mask(mask: Path, mask_dir: Path, image_index: Dict[str, Path]) -> Optional[Path]:
    rel = mask.relative_to(mask_dir)
    key = norm_rel(rel.with_suffix(""))
    return image_index.get(key) or image_index.get(mask.stem.lower())


def collect_seg(root: Path, cand, rng: random.Random, cap: int, thresholds: Dict[str, float]) -> Dict[str, int]:
    counts = defaultdict(int)
    if root is None or not root.exists():
        return counts
    image_dir, mask_dir = resolve_seg_dirs(root)
    if image_dir is None:
        return counts
    image_index = build_stem_index(image_dir)
    masks = [p for p in mask_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    rng.shuffle(masks)
    masks = masks[:cap]

    for mask_path in masks:
        image_path = find_image_for_mask(mask_path, mask_dir, image_index)
        if image_path is None:
            continue
        try:
            arr = np.asarray(Image.open(mask_path))
        except Exception:
            continue
        if arr.ndim == 3:
            arr = arr[..., 0]
        valid = arr != 255
        denom = max(int(valid.sum()), 1)
        for label in ("sky", "building", "food"):
            cls_id = SEG_CLASS_IDS[label]
            frac = float(((arr == cls_id) & valid).sum()) / denom
            thr = thresholds[label]
            if frac >= thr:
                cand[label][1].append((image_path, f"SEG13:{label}:area={frac:.6f}"))
                counts[f"{label}_pos"] += 1
            elif frac == 0.0:
                cand[label][0].append((image_path, f"SEG13:no_{label}"))
                counts[f"{label}_neg"] += 1
    return counts


def unique_candidates(items: Iterable[Tuple[Path, str]]) -> List[Tuple[Path, str]]:
    out = []
    seen = set()
    for p, ev in items:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            out.append((p, ev))
    return out


def choose(items, n: int, rng: random.Random):
    items = unique_candidates(items)
    if len(items) <= n:
        return items
    return rng.sample(items, n)


def materialize(src: Path, dst: Path, mode: str) -> Path:
    if mode == "none":
        return src.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        os.symlink(str(src.resolve()), str(dst))
    return dst.resolve() if mode == "copy" else dst.absolute()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    cand = {label: {0: [], 1: []} for label in LABELS}
    source_stats = {}
    if args.places_root:
        source_stats["places"] = dict(collect_places(args.places_root, args.places_per_category_cap, rng, cand))
    if args.coco_root:
        source_stats["coco"] = dict(collect_coco(args.coco_root, cand))
    if args.seg_root:
        source_stats["seg13"] = dict(collect_seg(
            args.seg_root, cand, rng, args.seg_candidate_cap,
            {"sky": args.sky_min_area, "building": args.building_min_area, "food": args.food_min_area},
        ))

    availability = {
        label: {
            "positive_candidates": len(unique_candidates(cand[label][1])),
            "negative_candidates": len(unique_candidates(cand[label][0])),
        }
        for label in LABELS
    }
    pre_summary = {
        "labels": LABELS,
        "display_names": DISPLAY_NAMES,
        "sources": {
            "places_root": str(args.places_root) if args.places_root else None,
            "coco_root": str(args.coco_root) if args.coco_root else None,
            "seg_root": str(args.seg_root) if args.seg_root else None,
        },
        "source_stats": source_stats,
        "candidate_availability": availability,
        "seed": args.seed,
    }
    (out / "candidate_summary.json").write_text(json.dumps(pre_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    missing = []
    for label in LABELS:
        if availability[label]["positive_candidates"] < args.min_pos:
            missing.append(f"{label}: positives {availability[label]['positive_candidates']} < {args.min_pos}")
        if availability[label]["negative_candidates"] < args.min_neg:
            missing.append(f"{label}: negatives {availability[label]['negative_candidates']} < {args.min_neg}")
    if missing:
        raise SystemExit(
            "Nine-label coverage check failed. Do not run inference on an incomplete benchmark.\n"
            + "\n".join(missing)
            + f"\nSee {out / 'candidate_summary.json'}"
        )

    # Merge selected evidence by source image. Unknown labels remain -1.
    entries: Dict[str, Dict] = {}
    for label in LABELS:
        for value, n in ((1, args.n_pos), (0, args.n_neg)):
            for path, evidence in choose(cand[label][value], n, rng):
                key = str(path.resolve())
                e = entries.setdefault(key, {
                    "source_path": key,
                    "labels": {x: -1 for x in LABELS},
                    "evidence": defaultdict(list),
                })
                old = e["labels"][label]
                if old not in (-1, value):
                    raise RuntimeError(f"Conflicting GT for {label}: {path}: {old} vs {value}")
                e["labels"][label] = value
                e["evidence"][label].append(evidence)

    # Deterministic order and optional symlink/copy materialization.
    ordered = sorted(entries.values(), key=lambda x: x["source_path"])
    manifest_path = out / "manifest.jsonl"
    csv_path = out / "labels.csv"
    image_out = out / "images"
    if args.materialize != "none":
        image_out.mkdir(exist_ok=True)

    known_counts = {l: {"positive": 0, "negative": 0, "unknown": 0} for l in LABELS}
    with manifest_path.open("w", encoding="utf-8") as mf, csv_path.open("w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["sample_id", "image_path", "source_path"] + LABELS)
        for idx, e in enumerate(ordered):
            src = Path(e["source_path"])
            sample_id = f"sample_{idx:06d}"
            suffix = src.suffix.lower() if src.suffix.lower() in IMAGE_EXTS else ".jpg"
            dst = image_out / f"{sample_id}{suffix}"
            image_path = materialize(src, dst, args.materialize)
            labels = e["labels"]
            evidence = {k: list(v) for k, v in e["evidence"].items()}
            rec = {
                "sample_id": sample_id,
                "image_path": str(image_path),
                "source_path": e["source_path"],
                "labels": labels,
                "evidence": evidence,
            }
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            writer.writerow([sample_id, str(image_path), e["source_path"]] + [labels[l] for l in LABELS])
            for l, v in labels.items():
                if v == 1:
                    known_counts[l]["positive"] += 1
                elif v == 0:
                    known_counts[l]["negative"] += 1
                else:
                    known_counts[l]["unknown"] += 1

    final_summary = {
        **pre_summary,
        "num_unique_images": len(ordered),
        "materialize": args.materialize,
        "known_counts": known_counts,
        "manifest": str(manifest_path),
        "csv": str(csv_path),
        "note": "-1 is unknown and MUST be excluded from metrics; never convert unknown to negative.",
    }
    (out / "dataset_summary.json").write_text(json.dumps(final_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(final_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
