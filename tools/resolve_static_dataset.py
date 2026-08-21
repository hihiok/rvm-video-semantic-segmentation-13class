#!/usr/bin/env python3
"""Find the actual COCO+ADE13 replay dataset; never silently pick tied roots."""

import argparse
import json
import re
import sys
from pathlib import Path


IMAGE_NAMES = ("images", "image", "imgs", "JPEGImages")
MASK_NAMES = ("annotations", "annotation", "masks", "mask", "labels", "label", "SegmentationClass")


def find_split(root, split):
    aliases = ("train", "training") if split == "train" else ("val", "valid", "validation")
    for alias in aliases:
        for image_name in IMAGE_NAMES:
            for mask_name in MASK_NAMES:
                for images, masks in (
                    (root / image_name / alias, root / mask_name / alias),
                    (root / alias / image_name, root / alias / mask_name),
                ):
                    if images.is_dir() and masks.is_dir():
                        return {"images": str(images.resolve()), "annotations": str(masks.resolve())}
    return None


def checkpoint_candidates(checkpoint):
    if checkpoint is None or not checkpoint.is_file():
        return []
    try:
        import torch

        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint, map_location="cpu")
    except Exception as error:
        print(f"WARNING: cannot inspect checkpoint metadata: {error}", file=sys.stderr)
        return []
    if not isinstance(payload, dict):
        return []

    roots = []
    for container in (payload, payload.get("args", {}), payload.get("config", {})):
        if not isinstance(container, dict):
            continue
        for key, value in container.items():
            if not isinstance(value, (str, Path)):
                continue
            if not any(token in key.lower() for token in ("data", "image", "mask", "annot", "root")):
                continue
            path = Path(value).expanduser()
            if not path.exists():
                continue
            for candidate in (path, *list(path.parents)[:3]):
                if candidate.is_dir():
                    roots.append(candidate.resolve())
    return roots


def source_project_candidates(source_project):
    if source_project is None or not source_project.is_dir():
        return []
    patterns = ("train*.py", "*.sh", "configs/*.json", "configs/*.yaml", "configs/*.yml", "scripts/*.sh")
    found = []
    path_pattern = re.compile(r"/(?:data|mnt)/[^\s\"'\\,;]+")
    for pattern in patterns:
        for source in source_project.glob(pattern):
            if not source.is_file() or source.stat().st_size > 1_000_000:
                continue
            for value in path_pattern.findall(source.read_text(encoding="utf-8", errors="ignore")):
                candidate = Path(value.rstrip("/)]}"))
                if candidate.is_dir():
                    found.extend([candidate.resolve(), *list(candidate.resolve().parents)[:2]])
    return found


def score_candidate(path, metadata_roots, project_roots):
    name = path.name.lower()
    score = 0
    if path in metadata_roots:
        score += 100
    if path in project_roots:
        score += 40
    if "coco" in name:
        score += 12
    if "ade" in name:
        score += 12
    if "13" in name:
        score += 15
    if "12" in name and "13" not in name:
        score -= 5
    if any(token in name for token in ("vspw", "vipseg", "video")):
        score -= 200
    return score


def discover_candidates(search_roots, metadata_roots=(), project_roots=()):
    metadata_roots, project_roots = set(metadata_roots), set(project_roots)
    candidates = set(metadata_roots) | set(project_roots)
    for root in search_roots:
        if not root.is_dir():
            continue
        candidates.add(root.resolve())
        for child in root.iterdir():
            if child.is_dir():
                candidates.add(child.resolve())
                if any(token in child.name.lower() for token in ("coco", "ade", "segment")):
                    try:
                        candidates.update(item.resolve() for item in child.iterdir() if item.is_dir())
                    except OSError:
                        pass

    found = []
    for candidate in candidates:
        train = find_split(candidate, "train")
        val = find_split(candidate, "val")
        if train and val:
            score = score_candidate(candidate, metadata_roots, project_roots)
            if score > 0:
                found.append({"root": str(candidate), "score": score, "train": train, "val": val})
    return sorted(found, key=lambda item: (-item["score"], item["root"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source-project", type=Path)
    parser.add_argument("--search-root", type=Path, action="append", default=[])
    parser.add_argument("--root-only", action="store_true")
    args = parser.parse_args(argv)
    defaults = [Path("/data/pub1/z00919662/dataset"), Path("/data/pub1/z00919662/segmentation/datasets")]
    metadata = checkpoint_candidates(args.checkpoint)
    project = source_project_candidates(args.source_project)
    candidates = discover_candidates(args.search_root or defaults, metadata, project)
    if not candidates:
        print(
            "ERROR: no valid COCO+ADE13 root with both train and val image/mask splits was found. "
            "Set STATIC_ROOT=/absolute/path/to/the/original/13-class/dataset.",
            file=sys.stderr,
        )
        return 2
    best = candidates[0]["score"]
    tied = [item for item in candidates if item["score"] == best]
    if len(tied) > 1:
        print(
            "ERROR: multiple equally plausible static replay datasets exist; "
            f"set STATIC_ROOT explicitly. Candidates: {json.dumps(tied, indent=2)}",
            file=sys.stderr,
        )
        return 3
    if args.root_only:
        print(candidates[0]["root"])
    else:
        print(json.dumps({"selected": candidates[0], "candidates": candidates}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
