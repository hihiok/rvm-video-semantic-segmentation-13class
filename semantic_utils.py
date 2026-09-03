"""Shared losses, metrics, checkpoint helpers, and 13-class constants."""

import csv
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor, distributed as dist
from torch.nn import functional as F


DEFAULT_CLASS_NAMES = [
    "background",
    "sky",
    "person",
    "plant",
    "building",
    "flower",
    "food",
    "water",
    "desert",
    "ice_or_snow",
    "text",
    "ball",
    "mountain",
]

DEFAULT_PALETTE = [
    (0, 0, 0),
    (70, 130, 180),
    (220, 20, 60),
    (34, 139, 34),
    (190, 190, 190),
    (255, 105, 180),
    (255, 165, 0),
    (0, 191, 255),
    (210, 180, 140),
    (224, 255, 255),
    (255, 215, 0),
    (148, 0, 211),
    (139, 90, 43),
]


def seed_everything(seed: int, rank: int = 0):
    seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_class_names(value: str) -> List[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError(f"Invalid unique comma-separated class names: {names}")
    return names


def parse_class_weights(value: Optional[str], num_classes: int):
    if value is None:
        return None
    weights = [float(item.strip()) for item in value.split(",")]
    if len(weights) != num_classes:
        raise ValueError(f"Expected {num_classes} weights, got {len(weights)}")
    return torch.tensor(weights, dtype=torch.float32)


def flatten_video_logits_target(logits: Tensor, target: Tensor):
    """Convert [B,T,C,H,W]/[B,T,H,W] into frame batches for losses."""
    if logits.ndim == 5:
        if target.ndim != 4 or logits.shape[:2] != target.shape[:2]:
            raise ValueError(f"Video logits/target mismatch: {logits.shape}, {target.shape}")
        logits = logits.flatten(0, 1)
        target = target.flatten(0, 1)
    if logits.ndim != 4 or target.ndim != 3:
        raise ValueError(f"Expected 4D logits and 3D target, got {logits.shape}, {target.shape}")
    return logits, target


def multiclass_dice_loss(
    logits: Tensor,
    target: Tensor,
    num_classes: int,
    ignore_index: int = 255,
    include_background: bool = True,
    epsilon: float = 1e-6,
):
    logits, target = flatten_video_logits_target(logits, target)
    valid = target.ne(ignore_index)
    safe_target = target.masked_fill(~valid, 0)
    probabilities = logits.softmax(dim=1)
    one_hot = F.one_hot(safe_target, num_classes=num_classes).permute(0, 3, 1, 2)
    one_hot = one_hot.to(probabilities.dtype)
    valid = valid.unsqueeze(1)
    probabilities = probabilities * valid
    one_hot = one_hot * valid
    dims = (0, 2, 3)
    intersection = (probabilities * one_hot).sum(dims)
    denominator = probabilities.sum(dims) + one_hot.sum(dims)
    dice = (2 * intersection + epsilon) / (denominator + epsilon)
    if not include_background:
        dice = dice[1:]
    return 1 - dice.mean()


def semantic_loss(
    logits: Tensor,
    target: Tensor,
    num_classes: int,
    class_weights: Optional[Tensor] = None,
    ignore_index: int = 255,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
):
    flat_logits, flat_target = flatten_video_logits_target(logits, target)
    ce = F.cross_entropy(flat_logits, flat_target, weight=class_weights, ignore_index=ignore_index)
    dice = multiclass_dice_loss(
        flat_logits, flat_target, num_classes=num_classes, ignore_index=ignore_index
    )
    return {"total": ce_weight * ce + dice_weight * dice, "cross_entropy": ce, "dice": dice}


def _gaussian_kernel(device, dtype):
    kernel = torch.tensor(
        [
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1],
        ],
        device=device,
        dtype=dtype,
    )
    return (kernel / 256).reshape(1, 1, 5, 5)


def _gaussian_convolution(image: Tensor, kernel: Tensor):
    batch, channels, height, width = image.shape
    flat = image.reshape(batch * channels, 1, height, width)
    flat = F.pad(flat, (2, 2, 2, 2), mode="reflect")
    return F.conv2d(flat, kernel).reshape(batch, channels, height, width)


def _laplacian_level(image: Tensor, kernel: Tensor):
    height = image.shape[-2] - image.shape[-2] % 2
    width = image.shape[-1] - image.shape[-1] % 2
    image = image[..., :height, :width]
    down = _gaussian_convolution(image, kernel)[..., ::2, ::2]
    expanded = torch.zeros(
        (*down.shape[:-2], down.shape[-2] * 2, down.shape[-1] * 2),
        device=image.device,
        dtype=image.dtype,
    )
    expanded[..., ::2, ::2] = down * 4
    up = _gaussian_convolution(expanded, kernel)
    return image - up, down


def multiclass_laplacian_loss(
    logits: Tensor,
    target: Tensor,
    num_classes: int,
    ignore_index: int = 255,
    max_levels: int = 5,
):
    """RVM-style multi-scale Laplacian loss over all semantic probabilities."""
    if max_levels < 1:
        raise ValueError("max_levels must be positive")
    logits, target = flatten_video_logits_target(logits, target)
    valid = target.ne(ignore_index).unsqueeze(1)
    safe_target = target.masked_fill(~valid.squeeze(1), 0)
    probability = logits.softmax(dim=1) * valid
    one_hot = F.one_hot(safe_target, num_classes=num_classes).permute(0, 3, 1, 2)
    one_hot = one_hot.to(probability.dtype) * valid
    kernel = _gaussian_kernel(probability.device, probability.dtype)

    loss = probability.sum() * 0.0
    used_levels = 0
    current_probability, current_target, current_valid = probability, one_hot, valid
    for level in range(max_levels):
        if min(current_probability.shape[-2:]) < 4:
            break
        probability_detail, current_probability = _laplacian_level(
            current_probability, kernel
        )
        target_detail, current_target = _laplacian_level(current_target, kernel)
        height, width = probability_detail.shape[-2:]
        level_valid = current_valid[..., :height, :width]
        denominator = level_valid.sum().clamp_min(1) * num_classes
        loss = loss + (2 ** level) * (
            (probability_detail - target_detail).abs() * level_valid
        ).sum() / denominator
        used_levels += 1
        current_valid = F.interpolate(
            level_valid.float(), size=current_probability.shape[-2:], mode="nearest"
        ).bool()
    return loss / max(used_levels, 1)


def rvm_temporal_derivative_loss(
    logits: Tensor,
    target: Tensor,
    ignore_index: int = 255,
    beta: float = 0.1,
):
    """Match predicted and GT frame-to-frame changes, adapted from RVM alpha coherence."""
    if logits.ndim != 5 or target.ndim != 4:
        raise ValueError(
            f"Expected logits [B,T,C,H,W] and target [B,T,H,W], got "
            f"{logits.shape} and {target.shape}"
        )
    if logits.shape[:2] != target.shape[:2] or logits.shape[-2:] != target.shape[-2:]:
        raise ValueError(f"Temporal logits/target mismatch: {logits.shape}, {target.shape}")
    if beta < 0:
        raise ValueError("beta must be nonnegative")
    if logits.shape[1] < 2:
        return logits.sum() * 0.0, 0, 0

    valid = (
        target[:, 1:].ne(ignore_index)
        & target[:, :-1].ne(ignore_index)
    )
    valid_count = int(valid.sum().detach().item())
    if valid_count == 0:
        return logits.sum() * 0.0, 0, 0

    safe_target = target.masked_fill(target.eq(ignore_index), 0)
    target_probability = F.one_hot(
        safe_target, num_classes=logits.shape[2]
    ).movedim(-1, 2).to(logits.dtype)
    probability = logits.softmax(dim=2)
    prediction_delta = probability[:, 1:] - probability[:, :-1]
    target_delta = target_probability[:, 1:] - target_probability[:, :-1]
    if beta == 0:
        channel_loss = (prediction_delta - target_delta).abs()
    else:
        channel_loss = F.smooth_l1_loss(
            prediction_delta, target_delta, reduction="none", beta=beta
        )
    pixel_loss = channel_loss.mean(dim=2)
    changed = valid & target[:, 1:].ne(target[:, :-1])
    return pixel_loss.masked_select(valid).mean(), valid_count, int(changed.sum().detach().item())


def semantic_boundary_mask(target: Tensor, ignore_index: int = 255, radius: int = 1):
    """Return pixels at, or close to, a valid semantic-label boundary."""
    if target.ndim not in (3, 4):
        raise ValueError(f"Expected [B,H,W] or [B,T,H,W], got {target.shape}")
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    original_shape = target.shape
    flat = target.flatten(0, 1) if target.ndim == 4 else target
    valid = flat.ne(ignore_index)
    boundary = torch.zeros_like(valid)

    horizontal = (
        valid[:, :, 1:]
        & valid[:, :, :-1]
        & flat[:, :, 1:].ne(flat[:, :, :-1])
    )
    boundary[:, :, 1:] |= horizontal
    boundary[:, :, :-1] |= horizontal
    vertical = (
        valid[:, 1:, :]
        & valid[:, :-1, :]
        & flat[:, 1:, :].ne(flat[:, :-1, :])
    )
    boundary[:, 1:, :] |= vertical
    boundary[:, :-1, :] |= vertical

    if radius:
        kernel = radius * 2 + 1
        boundary = F.max_pool2d(
            boundary.unsqueeze(1).float(), kernel_size=kernel, stride=1, padding=radius
        ).squeeze(1).bool()
    if target.ndim == 4:
        boundary = boundary.unflatten(0, original_shape[:2])
    return boundary


def causal_temporal_consistency_loss(
    logits: Tensor,
    target: Tensor,
    ignore_index: int = 255,
    boundary_radius: int = 2,
    temperature: float = 1.0,
):
    """
    Match the current prediction to the detached previous prediction only where
    consecutive ground-truth labels are unchanged and away from boundaries.

    The stable-GT mask avoids penalizing genuine motion or semantic changes. The
    causal, detached previous distribution also matches recurrent inference.
    """
    if logits.ndim != 5 or target.ndim != 4:
        raise ValueError(
            f"Expected logits [B,T,C,H,W] and target [B,T,H,W], got "
            f"{logits.shape} and {target.shape}"
        )
    if logits.shape[:2] != target.shape[:2] or logits.shape[-2:] != target.shape[-2:]:
        raise ValueError(f"Temporal logits/target mismatch: {logits.shape}, {target.shape}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if logits.shape[1] < 2:
        return logits.sum() * 0.0, 0

    boundary = semantic_boundary_mask(target, ignore_index, boundary_radius)
    stable = (
        target[:, 1:].ne(ignore_index)
        & target[:, :-1].ne(ignore_index)
        & target[:, 1:].eq(target[:, :-1])
        & ~boundary[:, 1:]
        & ~boundary[:, :-1]
    )
    stable_count = int(stable.sum().detach().item())
    if stable_count == 0:
        return logits.sum() * 0.0, 0

    previous_probability = (logits[:, :-1].detach() / temperature).softmax(dim=2)
    current_log_probability = (logits[:, 1:] / temperature).log_softmax(dim=2)
    pixel_kl = F.kl_div(
        current_log_probability, previous_probability, reduction="none"
    ).sum(dim=2)
    loss = pixel_kl.masked_select(stable).mean() * (temperature ** 2)
    return loss, stable_count


class ConfusionMatrix:
    def __init__(self, num_classes: int, device):
        self.num_classes = num_classes
        self.matrix = torch.zeros((num_classes, num_classes), dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, target: Tensor, prediction: Tensor, ignore_index: int = 255):
        target = target.reshape(-1)
        prediction = prediction.reshape(-1)
        valid = target.ne(ignore_index) & target.ge(0) & target.lt(self.num_classes)
        indices = self.num_classes * target[valid] + prediction[valid]
        counts = torch.bincount(indices, minlength=self.num_classes**2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes)

    def synchronize(self):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.matrix, op=dist.ReduceOp.SUM)

    def compute(self, class_names: List[str]) -> Dict[str, object]:
        tp = self.matrix.diag()
        gt = self.matrix.sum(1)
        predicted = self.matrix.sum(0)
        union = gt + predicted - tp
        iou = tp / union.clamp_min(1)
        recall = tp / gt.clamp_min(1)
        precision = tp / predicted.clamp_min(1)
        dice = 2 * tp / (gt + predicted).clamp_min(1)
        valid = union.gt(0)
        return {
            "pixel_accuracy": float(tp.sum() / self.matrix.sum().clamp_min(1)),
            "miou": float(iou[valid].mean()) if valid.any() else 0.0,
            "mean_dice": float(dice[valid].mean()) if valid.any() else 0.0,
            "class_iou": {name: float(iou[i]) if valid[i] else None for i, name in enumerate(class_names)},
            "class_recall": {name: float(recall[i]) if gt[i] > 0 else None for i, name in enumerate(class_names)},
            "class_precision": {name: float(precision[i]) if predicted[i] > 0 else None for i, name in enumerate(class_names)},
            "class_dice": {name: float(dice[i]) if valid[i] else None for i, name in enumerate(class_names)},
            "ground_truth_pixels": {name: int(gt[i]) for i, name in enumerate(class_names)},
        }


def colorize_mask(mask: np.ndarray, palette=DEFAULT_PALETTE):
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for index, rgb in enumerate(palette):
        color[mask == index] = rgb
    return color


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def atomic_torch_save(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def append_metrics_csv(path: Path, row: Dict[str, object]):
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {}
    for key, value in row.items():
        if isinstance(value, dict):
            flat.update({f"{key}/{subkey}": subvalue for subkey, subvalue in value.items()})
        else:
            flat[key] = value
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(flat))
        if write_header:
            writer.writeheader()
        writer.writerow(flat)


def format_metrics(metrics: Dict[str, object]) -> str:
    per_class = ", ".join(
        f"{name}={'N/A' if value is None else f'{value:.4f}'}"
        for name, value in metrics["class_iou"].items()
    )
    return (
        f"loss={metrics.get('loss', float('nan')):.4f}, "
        f"mIoU={metrics['miou']:.4f}, mDice={metrics['mean_dice']:.4f}, "
        f"pixel_acc={metrics['pixel_accuracy']:.4f}; IoU[{per_class}]"
    )
