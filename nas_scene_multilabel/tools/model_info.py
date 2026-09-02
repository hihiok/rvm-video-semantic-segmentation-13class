#!/usr/bin/env python3
import argparse
import torch
from torch import nn

from model import create_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--base-channel", type=int, default=16)
    args = p.parse_args()
    model = create_model(base_channel=args.base_channel).eval()
    params = sum(x.numel() for x in model.parameters())
    macs = 0

    def hook(m, inp, out):
        nonlocal macs
        if isinstance(m, nn.Conv2d):
            _, cout, h, w = out.shape
            kh, kw = m.kernel_size
            macs += cout * h * w * (m.in_channels // m.groups) * kh * kw
        elif isinstance(m, nn.Linear):
            macs += m.in_features * m.out_features

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        model(torch.zeros(1, 3, args.input_size, args.input_size))
    for h in hs:
        h.remove()
    print(f"params={params} ({params/1e6:.6f}M)")
    print(f"MACs={macs} ({macs/1e6:.3f}M MACs)")
    print(f"approx_FLOPs_if_2_per_MAC={2*macs/1e6:.3f}M FLOPs")


if __name__ == "__main__":
    main()
