#!/usr/bin/env python3
"""Reorganize official Places365 validation images into category folders.

The official Places365 validation archive contains 36,500 flat JPEG files and
`places365_val.txt` stores the class id for each image.  The existing NAS probe
builder expects category folders, so this script creates a symlink-only view:

  output_root/<category_name>/<Places365_val_xxxxxxxx.jpg>

No image bytes are copied by default.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--places-root", type=Path, required=True,
                   help="Root containing val_256, places365_val.txt, categories_places365.txt")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--copy", action="store_true", help="Copy instead of symlink")
    return p.parse_args()


def normalize_category(raw: str) -> str:
    raw = raw.strip().lstrip("/")
    parts = raw.split("/")
    if parts and len(parts[0]) == 1:
        parts = parts[1:]
    return "/".join(parts)


def find_val_dir(root: Path) -> Path:
    for p in (root / "val_256", root / "val_large", root / "val"):
        if p.is_dir():
            return p
    raise FileNotFoundError(f"Cannot find Places365 validation image directory under {root}")


def read_categories(path: Path):
    id_to_cat = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw, idx = line.rsplit(maxsplit=1)
            id_to_cat[int(idx)] = normalize_category(raw)
    if len(id_to_cat) != 365:
        raise RuntimeError(f"Expected 365 Places365 categories, got {len(id_to_cat)} from {path}")
    return id_to_cat


def read_val_list(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            filename, idx = line.rsplit(maxsplit=1)
            rows.append((Path(filename).name, int(idx)))
    if len(rows) != 36500:
        raise RuntimeError(f"Expected 36,500 validation rows, got {len(rows)} from {path}")
    return rows


def main():
    args = parse_args()
    root = args.places_root
    val_dir = find_val_dir(root)
    categories = root / "categories_places365.txt"
    val_list = root / "places365_val.txt"
    if not categories.is_file():
        raise FileNotFoundError(categories)
    if not val_list.is_file():
        raise FileNotFoundError(val_list)

    id_to_cat = read_categories(categories)
    rows = read_val_list(val_list)
    args.output_root.mkdir(parents=True, exist_ok=True)

    created = 0
    missing = []
    for filename, cls_id in rows:
        src = val_dir / filename
        if not src.is_file():
            missing.append(str(src))
            continue
        cat = id_to_cat[cls_id]
        dst_dir = args.output_root / cat
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / filename
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if args.copy:
            import shutil
            shutil.copy2(src, dst)
        else:
            os.symlink(str(src.resolve()), str(dst))
        created += 1

    print(f"VAL_DIR={val_dir}")
    print(f"OUTPUT_ROOT={args.output_root}")
    print(f"CREATED={created}")
    print(f"MISSING={len(missing)}")
    if missing:
        for p in missing[:20]:
            print(f"MISSING_FILE={p}")
        raise SystemExit(2)
    if created != 36500:
        raise RuntimeError(f"Expected 36,500 linked images, got {created}")


if __name__ == "__main__":
    main()
