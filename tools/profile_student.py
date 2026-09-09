#!/usr/bin/env python3
"""Report student parameters and 1080p compute with explicit counting conventions."""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import RVMForVideoSemanticSegmentation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-width", type=int, default=640)
    parser.add_argument("--profile-height", type=int, default=360)
    parser.add_argument("--target-width", type=int, default=1920)
    parser.add_argument("--target-height", type=int, default=1080)
    args = parser.parse_args()
    model = RVMForVideoSemanticSegmentation("mobilenetv3", 13).eval()
    macs = 0

    def count_conv(module, inputs, output):
        nonlocal macs
        value = output[0] if isinstance(output, (tuple, list)) else output
        macs += value.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]

    handles = [module.register_forward_hook(count_conv) for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
    with torch.inference_mode():
        model(torch.zeros(1, 1, 3, args.profile_height, args.profile_width))
    for handle in handles:
        handle.remove()
    area_scale = args.target_width * args.target_height / (args.profile_width * args.profile_height)
    target_macs = macs * area_scale
    parameters = sum(item.numel() for item in model.parameters())
    # The original RVM paper/table convention quotes 6.4G at 512x288. Keep it
    # alongside direct operator counting so deployment reports cannot mix units.
    rvm_table_gops = 6.4 * (args.target_width * args.target_height) / (512 * 288)
    print(json.dumps({
        "parameters": parameters,
        "parameters_million": parameters / 1e6,
        "target_resolution": [args.target_width, args.target_height],
        "direct_conv_gmac": target_macs / 1e9,
        "direct_conv_arithmetic_gflop_2_per_mac": 2 * target_macs / 1e9,
        "rvm_table_scaled_gops": rvm_table_gops,
        "note": "90G target uses the RVM table convention; direct Conv MAC/FLOP counting is also shown.",
    }, indent=2))


if __name__ == "__main__":
    main()
