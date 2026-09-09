#!/usr/bin/env python3
"""Select and validate the approved residual-v1 checkpoint for distillation."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import RVMForVideoSemanticSegmentation
from model.segmentation import extract_state_dict
from semantic_utils import DEFAULT_CLASS_NAMES, torch_load


PRIORITY = (
    "best_spatial_preserved.pth",
    "best_balanced.pth",
    "best_miou.pth",
    "best_video_miou.pth",
)


def inspect_checkpoint(path):
    checkpoint = torch_load(path, "cpu")
    state = extract_state_dict(checkpoint)
    names = checkpoint.get("class_names") if isinstance(checkpoint, dict) else None
    if names is not None and names != DEFAULT_CLASS_NAMES:
        raise ValueError(f"class_names mismatch: {names}")
    weight = state.get("project_seg.conv.weight")
    bias = state.get("project_seg.conv.bias")
    if weight is None or bias is None or weight.shape[0] != 13 or bias.shape[0] != 13:
        raise ValueError("checkpoint does not contain a compatible 13-class project_seg head")
    variant = checkpoint.get("variant", "mobilenetv3") if isinstance(checkpoint, dict) else "mobilenetv3"
    if variant != "mobilenetv3":
        raise ValueError(f"expected mobilenetv3 student, got {variant}")
    target = RVMForVideoSemanticSegmentation("mobilenetv3", 13).state_dict()
    compatible = sum(key in target and target[key].shape == value.shape for key, value in state.items())
    ratio = compatible / max(len(target), 1)
    if ratio < 0.8:
        raise ValueError(f"only {compatible}/{len(target)} model tensors are compatible ({ratio:.1%})")
    for field, expected in (("input_width", 640), ("input_height", 360)):
        actual = checkpoint.get(field) if isinstance(checkpoint, dict) else None
        if actual is not None and actual != expected:
            raise ValueError(f"{field} mismatch: expected {expected}, got {actual}")
    return {
        "path": str(path.resolve()),
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "variant": variant,
        "input_width": checkpoint.get("input_width") if isinstance(checkpoint, dict) else None,
        "input_height": checkpoint.get("input_height") if isinstance(checkpoint, dict) else None,
        "model_tensors": len(state),
        "compatible_tensors": compatible,
        "target_tensors": len(target),
        "compatibility_ratio": ratio,
    }


def select_checkpoint(directory):
    errors = {}
    for name in PRIORITY:
        path = directory / name
        if not path.is_file():
            continue
        try:
            return inspect_checkpoint(path), errors
        except Exception as error:
            errors[name] = str(error)
    raise RuntimeError(
        f"No compatible approved best checkpoint under {directory}; validation errors={errors}. "
        "last.pth is intentionally not selected automatically."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--print-path-only", action="store_true")
    args = parser.parse_args()
    report, validation_errors = select_checkpoint(args.checkpoint_dir.expanduser().resolve())
    report["selection_priority"] = list(PRIORITY)
    report["rejected_candidates"] = validation_errors
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report["path"] if args.print_path_only else json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
