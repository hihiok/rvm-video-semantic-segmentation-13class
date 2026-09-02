#!/usr/bin/env python3
"""Compare bilinear and RGB-guided semantic-logit upsampling on labeled frames."""

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PreparedStaticTransform, StaticSemanticDataset
from model import MultiClassFastGuidedFilterRefiner, RVMForVideoSemanticSegmentation
from semantic_utils import (
    ConfusionMatrix,
    DEFAULT_CLASS_NAMES,
    semantic_boundary_mask,
    torch_load,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--input-width", type=int, default=None)
    parser.add_argument("--input-height", type=int, default=None)
    parser.add_argument("--base-scale", type=float, default=0.5)
    parser.add_argument("--guided-radius", type=int, default=1)
    parser.add_argument("--guided-eps", type=float, default=1e-4)
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not 0 < args.base_scale < 1:
        parser.error("--base-scale must be between 0 and 1")
    if args.boundary_tolerance < 0:
        parser.error("--boundary-tolerance cannot be negative")
    return args


def amp_context(enabled):
    if enabled and torch.cuda.is_available():
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def dilate(mask, radius):
    if radius <= 0:
        return mask
    kernel = radius * 2 + 1
    return F.max_pool2d(
        mask.unsqueeze(1).float(), kernel_size=kernel, stride=1, padding=radius
    ).squeeze(1).bool()


class BoundaryAccumulator:
    def __init__(self, tolerance, ignore_index):
        self.tolerance = tolerance
        self.ignore_index = ignore_index
        self.predicted = 0
        self.target = 0
        self.matched_predicted = 0
        self.matched_target = 0

    def update(self, prediction, target):
        valid = target.ne(self.ignore_index)
        predicted_boundary = semantic_boundary_mask(prediction, self.ignore_index, 0) & valid
        target_boundary = semantic_boundary_mask(target, self.ignore_index, 0) & valid
        self.predicted += int(predicted_boundary.sum())
        self.target += int(target_boundary.sum())
        self.matched_predicted += int(
            (predicted_boundary & dilate(target_boundary, self.tolerance)).sum()
        )
        self.matched_target += int(
            (target_boundary & dilate(predicted_boundary, self.tolerance)).sum()
        )

    def compute(self):
        precision = self.matched_predicted / max(self.predicted, 1)
        recall = self.matched_target / max(self.target, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        return {
            "boundary_precision": precision,
            "boundary_recall": recall,
            "boundary_f1": f1,
            "predicted_boundary_pixels": self.predicted,
            "target_boundary_pixels": self.target,
        }


def main():
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch_load(args.checkpoint, "cpu")
    class_names = list(checkpoint.get("class_names", DEFAULT_CLASS_NAMES))
    if class_names != DEFAULT_CLASS_NAMES:
        raise ValueError(f"Expected fixed 13-class mapping, got {class_names}")
    width = args.input_width or int(checkpoint.get("input_width", 640))
    height = args.input_height or int(checkpoint.get("input_height", 360))
    base_width = max(1, round(width * args.base_scale))
    base_height = max(1, round(height * args.base_scale))

    dataset = StaticSemanticDataset(
        args.images,
        args.annotations,
        transform=PreparedStaticTransform((height, width)),
        num_classes=len(class_names),
        ignore_index=args.ignore_index,
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model = RVMForVideoSemanticSegmentation(
        checkpoint.get("variant", "mobilenetv3"), len(class_names)
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(device)
    guided = MultiClassFastGuidedFilterRefiner(args.guided_radius, args.guided_eps).eval().to(device)

    matrices = {
        "bilinear": ConfusionMatrix(len(class_names), device),
        "guided": ConfusionMatrix(len(class_names), device),
    }
    boundaries = {
        "bilinear": BoundaryAccumulator(args.boundary_tolerance, args.ignore_index),
        "guided": BoundaryAccumulator(args.boundary_tolerance, args.ignore_index),
    }
    seconds = {"network": 0.0, "bilinear": 0.0, "guided": 0.0}
    extra_peak_bytes = {"bilinear": 0, "guided": 0}
    frames = 0

    with torch.inference_mode():
        for images, masks in tqdm(loader, dynamic_ncols=True, desc="Guided upsample benchmark"):
            images = images[:, 0].to(device, non_blocking=True)
            masks = masks[:, 0].to(device, non_blocking=True)
            base_images = F.interpolate(
                images, size=(base_height, base_width), mode="bilinear", align_corners=False
            )

            synchronize(device)
            started = time.perf_counter()
            with amp_context(args.amp and device.type == "cuda"):
                base_logits = model(base_images)[0]
            synchronize(device)
            seconds["network"] += time.perf_counter() - started

            started = time.perf_counter()
            bilinear_before = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            bilinear_logits = F.interpolate(
                base_logits, size=(height, width), mode="bilinear", align_corners=False
            )
            synchronize(device)
            seconds["bilinear"] += time.perf_counter() - started
            if device.type == "cuda":
                extra_peak_bytes["bilinear"] = max(
                    extra_peak_bytes["bilinear"],
                    torch.cuda.max_memory_allocated(device) - bilinear_before,
                )

            started = time.perf_counter()
            guided_before = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            guided_logits = guided(
                base_images.float(), base_logits.float(), images.float()
            ).to(base_logits.dtype)
            synchronize(device)
            seconds["guided"] += time.perf_counter() - started
            if device.type == "cuda":
                extra_peak_bytes["guided"] = max(
                    extra_peak_bytes["guided"],
                    torch.cuda.max_memory_allocated(device) - guided_before,
                )

            for name, logits in (("bilinear", bilinear_logits), ("guided", guided_logits)):
                prediction = logits.argmax(dim=1)
                matrices[name].update(masks, prediction, args.ignore_index)
                boundaries[name].update(prediction, masks)
            frames += images.shape[0]

    results = {
        "checkpoint": str(args.checkpoint),
        "images": str(args.images),
        "annotations": str(args.annotations),
        "frames": frames,
        "fine_resolution": [width, height],
        "base_resolution": [base_width, base_height],
        "base_scale": args.base_scale,
        "guided_radius": args.guided_radius,
        "guided_eps": args.guided_eps,
        "boundary_tolerance": args.boundary_tolerance,
        "network_ms_per_frame": 1000.0 * seconds["network"] / max(frames, 1),
        "methods": {},
    }
    for name in ("bilinear", "guided"):
        metrics = matrices[name].compute(class_names)
        results["methods"][name] = {
            "miou": metrics["miou"],
            "pixel_accuracy": metrics["pixel_accuracy"],
            "class_iou": metrics["class_iou"],
            **boundaries[name].compute(),
            "upsample_ms_per_frame": 1000.0 * seconds[name] / max(frames, 1),
            "extra_peak_cuda_memory_mib": extra_peak_bytes[name] / (1024 ** 2),
        }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
