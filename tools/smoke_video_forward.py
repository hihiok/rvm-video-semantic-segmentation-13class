#!/usr/bin/env python3
"""Small CPU/GPU check that verifies 5D output and ConvGRU gradients."""

import argparse

import torch

from model import RVMForVideoSemanticSegmentation
from semantic_utils import semantic_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--size", type=int, default=64, help="Legacy square smoke-test size")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--time", type=int, default=3)
    args = parser.parse_args()
    if (args.width is None) != (args.height is None):
        parser.error("--width and --height must be supplied together")
    width = args.size if args.width is None else args.width
    height = args.size if args.height is None else args.height
    if min(width, height) < 1:
        parser.error("Input width and height must be positive")
    model = RVMForVideoSemanticSegmentation("mobilenetv3", 13).to(args.device).train()
    images = torch.rand(1, args.time, 3, height, width, device=args.device)
    targets = torch.randint(0, 13, (1, args.time, height, width), device=args.device)
    logits, *states = model(images)
    assert logits.shape == (1, args.time, 13, height, width), logits.shape
    assert len(states) == 4 and all(state is not None for state in states)
    loss = semantic_loss(logits, targets, 13)["total"]
    loss.backward()
    gru_gradient = sum(
        float(parameter.grad.abs().sum())
        for name, parameter in model.named_parameters()
        if ".gru." in name and parameter.grad is not None
    )
    assert gru_gradient > 0, "ConvGRU did not receive gradients"
    print({"logits": list(logits.shape), "state_shapes": [list(state.shape) for state in states], "loss": float(loss), "gru_gradient_l1": gru_gradient})


if __name__ == "__main__":
    main()
