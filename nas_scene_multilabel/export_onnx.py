#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

from config import LABELS
from model import create_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--opset", type=int, default=13)
    args = p.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu")
    input_size = int(ck.get("input_size", 224))
    base_channel = int(ck.get("base_channel", 16))
    model = create_model(len(LABELS), base_channel=base_channel, dropout=0.0)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    dummy = torch.zeros(1, 3, input_size, input_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(args.output), input_names=["input"], output_names=["logits"],
        dynamic_axes=None, opset_version=args.opset, do_constant_folding=True,
    )
    meta = {
        "labels": LABELS,
        "input_size": input_size,
        "normalization": "RGB float: x/255 -> (x-0.5)/0.5 per channel",
        "output": "9 logits; apply per-label sigmoid then thresholds",
        "thresholds": ck.get("thresholds", {l: 0.5 for l in LABELS}),
    }
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
