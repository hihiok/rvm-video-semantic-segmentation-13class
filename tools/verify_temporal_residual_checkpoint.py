#!/usr/bin/env python3
"""Verify frozen-base integrity and strict reset-frame bypass for a trained adapter."""

import argparse
import json
from pathlib import Path

import torch

from model import RVMForVideoSemanticSegmentation
from model.segmentation import extract_state_dict
from semantic_utils import DEFAULT_CLASS_NAMES, torch_load


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--require-trained-adapter", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch_load(args.checkpoint, "cpu")
    if checkpoint.get("class_names") != DEFAULT_CLASS_NAMES:
        raise ValueError("Checkpoint does not use the fixed 13-class mapping")
    if not checkpoint.get("temporal_residual_adapter"):
        raise ValueError("Checkpoint is not a temporal residual adapter model")

    trained = extract_state_dict(checkpoint)
    stage1 = extract_state_dict(torch_load(args.stage1_checkpoint, "cpu"))
    compared, changed = 0, []
    for name, value in trained.items():
        if name.startswith("temporal_residual_adapter.") or name not in stage1:
            continue
        if value.shape != stage1[name].shape:
            changed.append(name)
            continue
        compared += 1
        if not torch.equal(value.cpu(), stage1[name].cpu()):
            changed.append(name)
    if not compared:
        raise RuntimeError("No frozen Stage-1 tensors were comparable")
    if changed:
        raise AssertionError(f"Frozen Stage-1 tensors changed: {changed[:20]}")

    adapter_tensors = {
        name: value for name, value in trained.items()
        if name.startswith("temporal_residual_adapter.")
    }
    if not adapter_tensors:
        raise AssertionError("Checkpoint contains no temporal adapter tensors")
    output_changed = any(
        torch.count_nonzero(value).item() > 0
        for name, value in adapter_tensors.items()
        if ".output_projection." in name
    )
    if args.require_trained_adapter and not output_changed:
        raise AssertionError("Temporal output projection is still entirely zero")

    model = RVMForVideoSemanticSegmentation(
        checkpoint.get("variant", "mobilenetv3"),
        len(DEFAULT_CLASS_NAMES),
        temporal_residual=True,
        temporal_hidden_channels=checkpoint.get("temporal_hidden_channels", 16),
        temporal_scale=checkpoint.get("temporal_adapter_scale", 0.25),
    )
    model.load_state_dict(trained, strict=True)
    model.eval().to(args.device)
    torch.manual_seed(17)
    image = torch.rand(1, 1, 3, args.height, args.width, device=args.device)
    with torch.inference_mode():
        spatial = model.forward_spatial(image)
        reset_output, state, *_ = model(image)
    exact = torch.equal(spatial, reset_output)
    if not exact:
        raise AssertionError(
            f"Reset frame is not exact: max_abs_error={(spatial - reset_output).abs().max().item()}"
        )
    if state is None:
        raise AssertionError("Temporal adapter did not return an updated recurrent state")

    print(json.dumps({
        "status": "PASS",
        "frozen_stage1_tensors_compared": compared,
        "frozen_stage1_tensors_changed": 0,
        "adapter_tensors": len(adapter_tensors),
        "adapter_output_projection_nonzero": output_changed,
        "reset_frame_exact_spatial_bypass": exact,
        "checkpoint": str(args.checkpoint),
    }, indent=2))


if __name__ == "__main__":
    main()
