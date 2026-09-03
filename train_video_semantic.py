#!/usr/bin/env python3
"""Train recurrent RVM semantic segmentation on continuous video clips."""

import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Sampler, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from dataset import VideoClipDataset, VideoTrainTransform, VideoValidTransform
from model import RVMForVideoSemanticSegmentation, load_compatible_weights
from semantic_utils import (
    DEFAULT_CLASS_NAMES,
    ConfusionMatrix,
    append_metrics_csv,
    atomic_torch_save,
    format_metrics,
    parse_class_names,
    parse_class_weights,
    seed_everything,
    seed_worker,
    semantic_loss,
    torch_load,
    unwrap_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-images", default="images/train")
    parser.add_argument("--train-annotations", default="annotations/train")
    parser.add_argument("--val-images", default="images/val")
    parser.add_argument("--val-annotations", default="annotations/val")
    parser.add_argument("--class-names", default=",".join(DEFAULT_CLASS_NAMES))
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--variant", choices=["mobilenetv3", "resnet50"], default="mobilenetv3")
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--clip-length", type=int, default=5)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--train-clip-step", type=int, default=5)
    parser.add_argument("--val-clip-step", type=int, default=None)
    parser.add_argument("--temporal-reverse-probability", type=float, default=0.2)
    parser.add_argument(
        "--tbptt-chunk",
        type=int,
        default=0,
        help="0 means full-clip BPTT; positive values detach recurrence between chunks.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2, help="Clips per GPU")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-train-clips", type=int, default=0, help="Smoke-test limit; 0 uses all clips")
    parser.add_argument("--max-val-clips", type=int, default=0, help="Smoke-test limit; 0 uses all clips")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--class-weights", default=None)
    parser.add_argument("--train-scale-min", type=float, default=0.5)
    parser.add_argument("--train-scale-max", type=float, default=2.0)
    parser.add_argument("--val-resize-mode", choices=["letterbox", "stretch"], default="letterbox")
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--clip-grad-norm", type=float, default=5.0)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("output/rvm_video_semantic_13class"))
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sync-bn", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.resume and args.init_checkpoint:
        parser.error("--resume and --init-checkpoint are mutually exclusive")
    if args.gradient_accumulation < 1:
        parser.error("--gradient-accumulation must be >= 1")
    return args


def distributed_info(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    device = (
        torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    if distributed:
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    rank = dist.get_rank() if distributed else 0
    return distributed, rank, world_size, local_rank, device


class DistributedEvalSampler(Sampler):
    def __init__(self, dataset, rank, world_size):
        self.dataset, self.rank, self.world_size = dataset, rank, world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        return max(0, (len(self.dataset) - self.rank + self.world_size - 1) // self.world_size)


def make_dataloaders(args, num_classes, distributed, rank, world_size):
    train_dataset = VideoClipDataset(
        args.data_root / args.train_images,
        args.data_root / args.train_annotations,
        num_classes=num_classes,
        clip_length=args.clip_length,
        frame_stride=args.frame_stride,
        clip_step=args.train_clip_step,
        transform=VideoTrainTransform(
            args.input_size,
            (args.train_scale_min, args.train_scale_max),
            ignore_index=args.ignore_index,
        ),
        ignore_index=args.ignore_index,
        temporal_reverse_probability=args.temporal_reverse_probability,
        minimum_valid_frames=min(2, args.clip_length),
    )
    val_step = args.val_clip_step or args.clip_length * args.frame_stride
    val_dataset = VideoClipDataset(
        args.data_root / args.val_images,
        args.data_root / args.val_annotations,
        num_classes=num_classes,
        clip_length=args.clip_length,
        frame_stride=args.frame_stride,
        clip_step=val_step,
        transform=VideoValidTransform(args.input_size, args.val_resize_mode, args.ignore_index),
        ignore_index=args.ignore_index,
    )
    if args.max_train_clips > 0:
        train_dataset = Subset(train_dataset, range(min(args.max_train_clips, len(train_dataset))))
    if args.max_val_clips > 0:
        val_dataset = Subset(val_dataset, range(min(args.max_val_clips, len(val_dataset))))
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    val_sampler = DistributedEvalSampler(val_dataset, rank, world_size) if distributed else None
    common = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
        worker_init_fn=seed_worker,
    )
    train_loader = DataLoader(train_dataset, shuffle=train_sampler is None, sampler=train_sampler, drop_last=True, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, sampler=val_sampler, drop_last=False, **common)
    return train_loader, val_loader, train_sampler


def build_model_and_optimizer(args, class_names, device, distributed):
    model = RVMForVideoSemanticSegmentation(
        args.variant,
        num_classes=len(class_names),
        pretrained_backbone=args.init_checkpoint is None and args.resume is None,
        temporal_residual=getattr(args, "temporal_residual_adapter", False),
        temporal_hidden_channels=getattr(args, "temporal_hidden_channels", 16),
        temporal_scale=getattr(args, "temporal_adapter_scale", 0.25),
    ).to(device)
    if args.init_checkpoint:
        report = load_compatible_weights(
            model,
            torch_load(args.init_checkpoint, map_location="cpu"),
            target_class_names=class_names,
        )
        print(json.dumps(report, indent=2))
    if args.sync_bn and distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    head_parameters = (
        list(model.aspp.parameters())
        + list(model.decoder.parameters())
        + list(model.project_seg.parameters())
    )
    if model.temporal_residual_adapter is not None:
        head_parameters += list(model.temporal_residual_adapter.parameters())
    optimizer = AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.backbone_learning_rate},
            {"params": head_parameters, "lr": args.learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: max(0.0, 1.0 - epoch / max(args.epochs, 1)) ** 0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    return model, optimizer, scheduler, scaler


def forward_clip(model, images, tbptt_chunk=0):
    """Full BPTT by default; optionally carry detached state between chunks."""
    time = images.shape[1]
    if tbptt_chunk <= 0 or tbptt_chunk >= time:
        return model(images)[0]
    recurrence = [None] * 4
    outputs = []
    for start in range(0, time, tbptt_chunk):
        logits, *recurrence = model(images[:, start : start + tbptt_chunk], *recurrence)
        outputs.append(logits)
        recurrence = [state.detach() if state is not None else None for state in recurrence]
    return torch.cat(outputs, dim=1)


def amp_context(enabled):
    return torch.cuda.amp.autocast(enabled=enabled) if torch.cuda.is_available() else nullcontext()


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, args, class_weights, rank):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss, seen = 0.0, 0
    progress = tqdm(loader, dynamic_ncols=True, disable=rank != 0, desc=f"Train {epoch:03d}")
    for step, (images, masks) in enumerate(progress):
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        with amp_context(args.amp and device.type == "cuda"):
            logits = forward_clip(model, images, args.tbptt_chunk)
            losses = semantic_loss(
                logits,
                masks,
                logits.shape[2] if logits.ndim == 5 else logits.shape[1],
                class_weights,
                args.ignore_index,
                args.ce_weight,
                args.dice_weight,
            )
            scaled_loss = losses["total"] / args.gradient_accumulation
        scaler.scale(scaled_loss).backward()
        should_step = (step + 1) % args.gradient_accumulation == 0 or step + 1 == len(loader)
        if should_step:
            if args.clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        batch_size = images.shape[0]
        running_loss += losses["total"].detach().item() * batch_size
        seen += batch_size
        if rank == 0 and (step + 1) % args.print_every == 0:
            progress.set_postfix(loss=f"{running_loss / max(seen, 1):.4f}")
    totals = torch.tensor([running_loss, seen], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals)
    return float(totals[0] / totals[1].clamp_min(1))


@torch.no_grad()
def validate(model, loader, device, args, class_names, class_weights, rank):
    model.eval()
    matrix = ConfusionMatrix(len(class_names), device)
    total_loss, seen = 0.0, 0
    for images, masks in tqdm(loader, dynamic_ncols=True, disable=rank != 0, desc="Validate"):
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        with amp_context(args.amp and device.type == "cuda"):
            logits = forward_clip(model, images, args.tbptt_chunk)
            losses = semantic_loss(logits, masks, len(class_names), class_weights, args.ignore_index, args.ce_weight, args.dice_weight)
        matrix.update(masks, logits.argmax(dim=2), args.ignore_index)
        total_loss += losses["total"].item() * images.shape[0]
        seen += images.shape[0]
    totals = torch.tensor([total_loss, seen], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals)
    matrix.synchronize()
    metrics = matrix.compute(class_names)
    metrics["loss"] = float(totals[0] / totals[1].clamp_min(1))
    return metrics


def load_resume(path, model, optimizer, scheduler, scaler, class_names):
    checkpoint = torch_load(path, "cpu")
    if checkpoint.get("class_names") != class_names:
        raise ValueError("Resume checkpoint class mapping does not match")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint["epoch"]) + 1, float(checkpoint.get("best_miou", -1.0))


def checkpoint_payload(model, optimizer, scheduler, scaler, epoch, best_miou, args, class_names):
    serialized_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    return {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_miou": best_miou,
        "variant": args.variant,
        "num_classes": len(class_names),
        "class_names": class_names,
        "input_size": args.input_size,
        "clip_length": args.clip_length,
        "frame_stride": args.frame_stride,
        "video_training": True,
        "args": serialized_args,
    }


def main():
    args = parse_args()
    class_names = parse_class_names(args.class_names)
    if class_names != DEFAULT_CLASS_NAMES:
        print(f"Warning: non-default class mapping requested: {class_names}")
    distributed, rank, world_size, local_rank, device = distributed_info(args)
    seed_everything(args.seed, rank)
    class_weights = parse_class_weights(args.class_weights, len(class_names))
    if class_weights is not None:
        class_weights = class_weights.to(device)
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Classes: {dict(enumerate(class_names))}")
        print(f"Video clips: T={args.clip_length}, stride={args.frame_stride}; device={device}; world_size={world_size}")
    train_loader, val_loader, train_sampler = make_dataloaders(args, len(class_names), distributed, rank, world_size)
    model, optimizer, scheduler, scaler = build_model_and_optimizer(args, class_names, device, distributed)
    start_epoch, best_miou = 0, -1.0
    if args.resume:
        start_epoch, best_miou = load_resume(args.resume, model, optimizer, scheduler, scaler, class_names)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None, broadcast_buffers=False)
    if args.evaluate_only:
        metrics = validate(model, val_loader, device, args, class_names, class_weights, rank)
        if rank == 0:
            print(format_metrics(metrics))
        if distributed:
            dist.destroy_process_group()
        return
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        started = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, args, class_weights, rank)
        metrics = validate(model, val_loader, device, args, class_names, class_weights, rank)
        scheduler.step()
        if rank == 0:
            print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, {format_metrics(metrics)}, time={time.time() - started:.1f}s")
            append_metrics_csv(args.output_dir / "metrics.csv", {"epoch": epoch, "train_loss": train_loss, **metrics})
            improved = metrics["miou"] > best_miou
            best_miou = max(best_miou, metrics["miou"])
            payload = checkpoint_payload(model, optimizer, scheduler, scaler, epoch, best_miou, args, class_names)
            atomic_torch_save(payload, args.output_dir / "last.pth")
            if improved:
                atomic_torch_save(payload, args.output_dir / "best_miou.pth")
            if (epoch + 1) % args.save_every == 0:
                atomic_torch_save(payload, args.output_dir / f"epoch_{epoch:03d}.pth")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
