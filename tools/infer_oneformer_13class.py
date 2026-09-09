#!/usr/bin/env python3
"""Inspect the exact soft teacher mapping used by distillation on one image."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from distillation.oneformer_teacher import OneFormer13ClassTeacher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, default=Path("configs/oneformer_ade20k_to_13class.json"))
    parser.add_argument("--model", default="shi-labs/oneformer_ade20k_swin_large")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    with Image.open(args.image) as handle:
        image = handle.convert("RGB")
    teacher = OneFormer13ClassTeacher(args.model, args.mapping, args.device)
    probability = teacher([image], (image.height, image.width))[0]
    prediction = probability.argmax(0).byte().numpy()
    args.output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(prediction).save(args.output / "mask_13class.png")
    stats = {name: float(probability[index].mean()) for index, name in enumerate(teacher.mapping["target_class_names"])}
    (args.output / "class_probability_mass.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
