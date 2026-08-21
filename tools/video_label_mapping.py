"""Common VIPSeg/VSPW label decoding and mapping helpers."""

import json
from pathlib import Path

import numpy as np


def normalize_name(name):
    return str(name).strip().lower()


def load_mapping(categories_json, mapping_json):
    categories = json.loads(Path(categories_json).read_text(encoding="utf-8"))
    config = json.loads(Path(mapping_json).read_text(encoding="utf-8"))
    target_classes = sorted(config["target_classes"], key=lambda item: item["id"])
    expected_ids = list(range(len(target_classes)))
    if [item["id"] for item in target_classes] != expected_ids:
        raise ValueError("Target class IDs must be contiguous and begin at zero")
    source_name_to_raw_id = {
        normalize_name(item["name"]): int(item["id"]) + 1 for item in categories
    }
    raw_to_target = {}
    missing = []
    for target in target_classes:
        for source_name in target["source_names"]:
            raw_id = source_name_to_raw_id.get(normalize_name(source_name))
            if raw_id is None:
                missing.append(source_name)
            else:
                raw_to_target[raw_id] = int(target["id"])
    if missing:
        raise ValueError(f"Source categories missing from categories JSON: {missing}")
    return target_classes, raw_to_target, config


def semantic_category_ids(raw_mask, panoptic):
    raw_mask = np.asarray(raw_mask)
    if raw_mask.ndim != 2:
        raise ValueError(f"Expected a one-channel mask, got {raw_mask.shape}")
    values = raw_mask.astype(np.int64, copy=False)
    if panoptic:
        values = np.where(values > 124, values // 100, values)
    source_ignore = (values == 255) if not panoptic else np.zeros(values.shape, dtype=bool)
    invalid = ((values < 0) | (values > 124)) & ~source_ignore
    if invalid.any():
        raise ValueError(f"Invalid source category IDs: {np.unique(values[invalid]).tolist()}")
    return values


def convert_mask(raw_mask, raw_to_target, ignore_index=255, panoptic=True):
    category_ids = semantic_category_ids(raw_mask, panoptic=panoptic)
    output = np.zeros(category_ids.shape, dtype=np.uint8)
    output[(category_ids == 0) | (category_ids == 255)] = ignore_index
    for raw_id, target_id in raw_to_target.items():
        output[category_ids == raw_id] = target_id
    return output
