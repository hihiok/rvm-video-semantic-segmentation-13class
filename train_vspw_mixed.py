#!/usr/bin/env python3
"""Fine-tune RVM on VSPW clips while replaying COCO/ADE13 photographs."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from dataset import (
    StaticSemanticDataset,
    VideoClipDataset,
    VideoTrainTransform,
    VideoValidTransform,
    resolve_static_split,
)
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
from model.segmentation import extract_state_dict
from train_video_semantic import (
    DistributedEvalSampler,
    amp_context,
    build_model_and_optimizer,
    distributed_info,
    forward_clip,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Converted VSPW_13cls")
    parser.add_argument("--static-root", type=Path, required=True, help="COCO+ADE13 dataset")
    parser.add_argument("--train-images", default="images/train")
    parser.add_argument("--train-annotations", default="annotations/train")
    parser.add_argument("--val-images", default="images/val")
    parser.add_argument("--val-annotations", default="annotations/val")
    parser.add_argument("--static-train-images", default=None)
    parser.add_argument("--static-train-annotations", default=None)
    parser.add_argument("--static-val-images", default=None)
    parser.add_argument("--static-val-annotations", default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("output/rvm_vspw_mixed_13class"))
    parser.add_argument("--class-names", default=",".join(DEFAULT_CLASS_NAMES))
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--variant", choices=("mobilenetv3", "resnet50"), default="mobilenetv3")
    parser.add_argument(
        "--input-size", type=int, default=None,
        help="Legacy square input; overrides --input-width and --input-height when supplied",
    )
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--input-height", type=int, default=360)
    parser.add_argument("--stage2-epochs", type=int, default=20)
    parser.add_argument("--stage3-epochs", type=int, default=60)
    parser.add_argument("--stage2-clip-length", type=int, default=5)
    parser.add_argument("--stage3-clip-length", type=int, default=8)
    parser.add_argument("--stage2-video-batches", type=int, default=1)
    parser.add_argument("--stage2-static-batches", type=int, default=1)
    parser.add_argument("--stage3-video-batches", type=int, default=2)
    parser.add_argument("--stage3-static-batches", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--train-clip-step", type=int, default=0, help="0 uses stage clip length")
    parser.add_argument("--val-clip-step", type=int, default=0, help="0 uses stage clip length")
    parser.add_argument("--temporal-reverse-probability", type=float, default=0.2)
    parser.add_argument("--tbptt-chunk", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2, help="Video clips per GPU")
    parser.add_argument("--static-batch-size", type=int, default=8, help="Static images per GPU")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-train-clips", type=int, default=0)
    parser.add_argument("--max-val-clips", type=int, default=0)
    parser.add_argument("--max-static-train-images", type=int, default=0)
    parser.add_argument("--max-static-val-images", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--backbone-learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--class-weights", default=None)
    parser.add_argument("--video-loss-weight", type=float, default=1.0)
    parser.add_argument("--static-loss-weight", type=float, default=1.0)
    parser.add_argument("--train-scale-min", type=float, default=0.5)
    parser.add_argument("--train-scale-max", type=float, default=1.5)
    parser.add_argument("--val-resize-mode", choices=("letterbox", "stretch"), default="letterbox")
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--clip-grad-norm", type=float, default=5.0)
    parser.add_argument("--static-validation-weight", type=float, default=0.5)
    parser.add_argument("--static-retention-tolerance", type=float, default=0.03)
    parser.add_argument("--baseline-validation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sync-bn", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    if args.input_size is not None:
        args.input_width = args.input_height = args.input_size
    if args.input_width < 1 or args.input_height < 1:
        parser.error("--input-width and --input-height must be positive")
    if bool(args.init_checkpoint) == bool(args.resume):
        parser.error("Supply exactly one of --init-checkpoint and --resume")
    if args.stage2_epochs < 0 or args.stage3_epochs < 0 or args.stage2_epochs + args.stage3_epochs < 1:
        parser.error("Stage lengths must be nonnegative and their total must be positive")
    for field in (
        "stage2_clip_length", "stage3_clip_length", "stage2_video_batches",
        "stage2_static_batches", "stage3_video_batches", "stage3_static_batches",
        "batch_size", "static_batch_size", "gradient_accumulation",
    ):
        if getattr(args, field) < 1:
            parser.error(f"--{field.replace('_', '-')} must be at least 1")
    if not 0 <= args.static_validation_weight <= 1:
        parser.error("--static-validation-weight must be between 0 and 1")
    args.epochs = args.stage2_epochs + args.stage3_epochs
    return args


def stage_for_epoch(args, epoch):
    if epoch < args.stage2_epochs:
        return {
            "name": "stage2_mixed_adaptation",
            "clip_length": args.stage2_clip_length,
            "video_batches": args.stage2_video_batches,
            "static_batches": args.stage2_static_batches,
        }
    return {
        "name": "stage3_temporal_finetuning",
        "clip_length": args.stage3_clip_length,
        "video_batches": args.stage3_video_batches,
        "static_batches": args.stage3_static_batches,
    }


def _subset(dataset, maximum):
    if maximum <= 0:
        return dataset
    return Subset(dataset, range(min(maximum, len(dataset))))


def make_loaders(args, stage, num_classes, distributed, rank, world_size):
    train_transform = VideoTrainTransform(
        (args.input_height, args.input_width),
        (args.train_scale_min, args.train_scale_max),
        ignore_index=args.ignore_index,
    )
    valid_transform = VideoValidTransform(
        (args.input_height, args.input_width), args.val_resize_mode, args.ignore_index
    )
    clip_length = stage["clip_length"]
    train_video = VideoClipDataset(
        args.data_root / args.train_images,
        args.data_root / args.train_annotations,
        num_classes=num_classes,
        clip_length=clip_length,
        frame_stride=args.frame_stride,
        clip_step=args.train_clip_step or clip_length * args.frame_stride,
        transform=train_transform,
        ignore_index=args.ignore_index,
        temporal_reverse_probability=args.temporal_reverse_probability,
        minimum_valid_frames=min(2, clip_length),
    )
    val_video = VideoClipDataset(
        args.data_root / args.val_images,
        args.data_root / args.val_annotations,
        num_classes=num_classes,
        clip_length=clip_length,
        frame_stride=args.frame_stride,
        clip_step=args.val_clip_step or clip_length * args.frame_stride,
        transform=valid_transform,
        ignore_index=args.ignore_index,
    )
    train_video = _subset(train_video, args.max_train_clips)
    val_video = _subset(val_video, args.max_val_clips)

    train_paths = resolve_static_split(
        args.static_root, "train", args.static_train_images, args.static_train_annotations
    )
    val_paths = resolve_static_split(
        args.static_root, "val", args.static_val_images, args.static_val_annotations
    )
    train_static = StaticSemanticDataset(
        train_paths.image_root, train_paths.mask_root, train_transform,
        num_classes=num_classes, ignore_index=args.ignore_index,
        max_samples=args.max_static_train_images,
    )
    val_static = StaticSemanticDataset(
        val_paths.image_root, val_paths.mask_root, valid_transform,
        num_classes=num_classes, ignore_index=args.ignore_index,
        max_samples=args.max_static_val_images,
    )

    video_sampler = (
        DistributedSampler(train_video, num_replicas=world_size, rank=rank, shuffle=True)
        if distributed else None
    )
    static_sampler = (
        DistributedSampler(train_static, num_replicas=world_size, rank=rank, shuffle=True)
        if distributed else None
    )
    video_eval_sampler = DistributedEvalSampler(val_video, rank, world_size) if distributed else None
    static_eval_sampler = DistributedEvalSampler(val_static, rank, world_size) if distributed else None
    common = {
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.workers > 0,
        "worker_init_fn": seed_worker,
    }
    loaders = {
        "video_train": DataLoader(
            train_video, batch_size=args.batch_size, shuffle=video_sampler is None,
            sampler=video_sampler, drop_last=False, **common,
        ),
        "video_val": DataLoader(
            val_video, batch_size=args.batch_size, shuffle=False,
            sampler=video_eval_sampler, drop_last=False, **common,
        ),
        "static_train": DataLoader(
            train_static, batch_size=args.static_batch_size, shuffle=static_sampler is None,
            sampler=static_sampler, drop_last=False, **common,
        ),
        "static_val": DataLoader(
            val_static, batch_size=args.static_batch_size, shuffle=False,
            sampler=static_eval_sampler, drop_last=False, **common,
        ),
    }
    if not len(loaders["video_train"]) or not len(loaders["static_train"]):
        raise RuntimeError("Both VSPW and static replay loaders must contain training batches")
    if rank == 0:
        print(json.dumps({
            "stage": stage,
            "vspw_train_clips": len(train_video),
            "vspw_val_clips": len(val_video),
            "static_train_images": len(train_static),
            "static_val_images": len(val_static),
            "static_train_paths": {
                "images": str(train_paths.image_root), "annotations": str(train_paths.mask_root),
            },
            "static_val_paths": {
                "images": str(val_paths.image_root), "annotations": str(val_paths.mask_root),
            },
        }, indent=2))
    return loaders, video_sampler, static_sampler


def mixed_batch_sources(video_count, video_batches, static_batches):
    """Replay static batches after every video group, including a short final group."""
    if video_count < 0 or video_batches < 1 or static_batches < 1:
        raise ValueError("Invalid video count or source ratio")
    for start in range(0, video_count, video_batches):
        for _ in range(min(video_batches, video_count - start)):
            yield "video"
        for _ in range(static_batches):
            yield "static"


def _infinite_batches(loader):
    while True:
        yield from loader


def train_mixed_epoch(model, loaders, optimizer, scaler, device, epoch, args, stage, weights, rank):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    video_iter = iter(loaders["video_train"])
    static_iter = _infinite_batches(loaders["static_train"])
    sources = list(mixed_batch_sources(
        len(loaders["video_train"]), stage["video_batches"], stage["static_batches"]
    ))
    totals = {"video_loss": 0.0, "video_samples": 0, "static_loss": 0.0, "static_samples": 0}
    progress = tqdm(sources, disable=rank != 0, dynamic_ncols=True, desc=f"Mixed {epoch:03d}")
    for index, source in enumerate(progress):
        images, masks = next(video_iter if source == "video" else static_iter)
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with amp_context(args.amp and device.type == "cuda"):
            logits = forward_clip(model, images, args.tbptt_chunk if source == "video" else 0)
            losses = semantic_loss(
                logits, masks, logits.shape[2], weights, args.ignore_index,
                args.ce_weight, args.dice_weight,
            )
            domain_weight = args.video_loss_weight if source == "video" else args.static_loss_weight
            loss = losses["total"] * domain_weight / args.gradient_accumulation
        scaler.scale(loss).backward()
        if (index + 1) % args.gradient_accumulation == 0 or index + 1 == len(sources):
            if args.clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        count = images.shape[0]
        totals[f"{source}_loss"] += losses["total"].detach().item() * count
        totals[f"{source}_samples"] += count
        if rank == 0 and (index + 1) % args.print_every == 0:
            progress.set_postfix(
                video=f"{totals['video_loss'] / max(totals['video_samples'], 1):.4f}",
                static=f"{totals['static_loss'] / max(totals['static_samples'], 1):.4f}",
            )

    values = torch.tensor(list(totals.values()), dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values)
    totals = dict(zip(totals, values.tolist()))
    return {
        "video_train_loss": totals["video_loss"] / max(totals["video_samples"], 1),
        "static_train_loss": totals["static_loss"] / max(totals["static_samples"], 1),
        "video_train_samples": int(totals["video_samples"]),
        "static_train_samples": int(totals["static_samples"]),
    }


@torch.no_grad()
def validate_domain(model, loader, device, args, class_names, weights, rank, domain):
    model.eval()
    matrix = ConfusionMatrix(len(class_names), device)
    total_loss, seen, stable_flips, stable_pixels = 0.0, 0, 0, 0
    for images, masks in tqdm(loader, disable=rank != 0, dynamic_ncols=True, desc=f"Validate {domain}"):
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        with amp_context(args.amp and device.type == "cuda"):
            logits = forward_clip(model, images, args.tbptt_chunk if domain == "video" else 0)
            losses = semantic_loss(
                logits, masks, len(class_names), weights, args.ignore_index,
                args.ce_weight, args.dice_weight,
            )
        prediction = logits.argmax(dim=2)
        matrix.update(masks, prediction, args.ignore_index)
        total_loss += losses["total"].item() * images.shape[0]
        seen += images.shape[0]
        if domain == "video" and masks.shape[1] > 1:
            valid = (
                masks[:, 1:].ne(args.ignore_index)
                & masks[:, :-1].ne(args.ignore_index)
                & masks[:, 1:].eq(masks[:, :-1])
            )
            stable_pixels += valid.sum().item()
            stable_flips += (prediction[:, 1:].ne(prediction[:, :-1]) & valid).sum().item()

    totals = torch.tensor([total_loss, seen, stable_flips, stable_pixels], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals)
    matrix.synchronize()
    metrics = matrix.compute(class_names)
    metrics["loss"] = float(totals[0] / totals[1].clamp_min(1))
    if domain == "video":
        metrics["stable_gt_flip_rate"] = float(totals[2] / totals[3].clamp_min(1))
        metrics["stable_gt_pixels"] = int(totals[3])
    return metrics


def balanced_score(video_metrics, static_metrics, static_weight):
    return (1.0 - static_weight) * video_metrics["miou"] + static_weight * static_metrics["miou"]


def checkpoint_payload(model, optimizer, scheduler, scaler, epoch, args, class_names, records, baseline):
    serialized_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    stage = stage_for_epoch(args, epoch)
    return {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_miou": records["best_balanced"],
        "best_video_miou": records["best_video"],
        "best_static_miou": records["best_static"],
        "best_balanced_score": records["best_balanced"],
        "variant": args.variant,
        "num_classes": len(class_names),
        "class_names": class_names,
        "input_size": args.input_width,
        "input_width": args.input_width,
        "input_height": args.input_height,
        "clip_length": stage["clip_length"],
        "frame_stride": args.frame_stride,
        "video_training": True,
        "mixed_replay_training": True,
        "training_stage": stage["name"],
        "baseline_metrics": baseline,
        "args": serialized_args,
    }


def load_resume(path, model, optimizer, scheduler, scaler, class_names):
    checkpoint = torch_load(path, "cpu")
    if checkpoint.get("class_names") != class_names:
        raise ValueError("Resume checkpoint class mapping does not match the fixed 13 classes")
    if not checkpoint.get("mixed_replay_training"):
        raise ValueError("--resume requires a checkpoint produced by train_vspw_mixed.py")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    records = {
        "best_video": float(checkpoint.get("best_video_miou", -1.0)),
        "best_static": float(checkpoint.get("best_static_miou", -1.0)),
        "best_balanced": float(checkpoint.get("best_balanced_score", -1.0)),
    }
    return int(checkpoint["epoch"]) + 1, records, checkpoint.get("baseline_metrics")


def prefixed_metrics(prefix, metrics):
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def verify_stage1_checkpoint(path, model, class_names):
    """Reject partial/wrong-taxonomy initialization instead of training from random weights."""
    checkpoint = torch_load(path, "cpu")
    source = extract_state_dict(checkpoint)
    target = model.state_dict()
    compatible = sum(
        key in target and value.shape == target[key].shape
        for key, value in source.items()
    )
    ratio = compatible / max(len(target), 1)
    head_key = "project_seg.conv.weight"
    if head_key not in source or source[head_key].shape != target[head_key].shape:
        raise ValueError(
            f"Initial checkpoint must already contain the compatible {len(class_names)}-class "
            f"segmentation head {head_key}; received "
            f"{None if head_key not in source else tuple(source[head_key].shape)}"
        )
    source_names = checkpoint.get("class_names") if isinstance(checkpoint, dict) else None
    if source_names and (len(source_names) != len(class_names) or set(source_names) != set(class_names)):
        raise ValueError(f"Initial checkpoint has a different semantic taxonomy: {source_names}")
    if ratio < 0.8:
        raise ValueError(
            f"Initial checkpoint matches only {compatible}/{len(target)} model tensors "
            f"({ratio:.1%}); refusing unsafe partial initialization"
        )
    return {"compatible_tensors": compatible, "target_tensors": len(target), "compatibility_ratio": ratio}


def main(argv=None):
    args = parse_args(argv)
    class_names = parse_class_names(args.class_names)
    if class_names != DEFAULT_CLASS_NAMES:
        raise ValueError(f"This workflow requires the fixed 13-class mapping: {DEFAULT_CLASS_NAMES}")
    distributed, rank, world_size, local_rank, device = distributed_info(args)
    seed_everything(args.seed, rank)
    weights = parse_class_weights(args.class_weights, len(class_names))
    if weights is not None:
        weights = weights.to(device)
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(json.dumps({
            "class_names": class_names, "device": str(device), "world_size": world_size,
            "input_width": args.input_width, "input_height": args.input_height,
            "stage2_epochs": args.stage2_epochs, "stage3_epochs": args.stage3_epochs,
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
            "resume": str(args.resume) if args.resume else None,
        }, indent=2))

    model, optimizer, scheduler, scaler = build_model_and_optimizer(args, class_names, device, distributed)
    if args.init_checkpoint:
        compatibility = verify_stage1_checkpoint(args.init_checkpoint, model, class_names)
        if rank == 0:
            print(json.dumps({"stage1_checkpoint_verification": compatibility}, indent=2))
    records = {"best_video": -1.0, "best_static": -1.0, "best_balanced": -1.0}
    start_epoch, baseline = 0, None
    if args.resume:
        start_epoch, records, baseline = load_resume(
            args.resume, model, optimizer, scheduler, scaler, class_names
        )
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[local_rank] if device.type == "cuda" else None,
            broadcast_buffers=False,
        )

    initial_stage = stage_for_epoch(args, min(start_epoch, args.epochs - 1))
    loaders, video_sampler, static_sampler = make_loaders(
        args, initial_stage, len(class_names), distributed, rank, world_size
    )
    active_stage = initial_stage["name"]
    if baseline is None and (args.baseline_validation or args.evaluate_only):
        baseline = {
            "video": validate_domain(model, loaders["video_val"], device, args, class_names, weights, rank, "video"),
            "static": validate_domain(model, loaders["static_val"], device, args, class_names, weights, rank, "static"),
        }
        baseline["balanced_score"] = balanced_score(
            baseline["video"], baseline["static"], args.static_validation_weight
        )
        if rank == 0:
            (args.output_dir / "baseline_metrics.json").write_text(
                json.dumps(baseline, indent=2), encoding="utf-8"
            )
            print(f"Baseline VSPW: {format_metrics(baseline['video'])}")
            print(f"Baseline COCO+ADE: {format_metrics(baseline['static'])}")
    if args.evaluate_only:
        if distributed:
            dist.destroy_process_group()
        return

    for epoch in range(start_epoch, args.epochs):
        stage = stage_for_epoch(args, epoch)
        if stage["name"] != active_stage:
            loaders, video_sampler, static_sampler = make_loaders(
                args, stage, len(class_names), distributed, rank, world_size
            )
            active_stage = stage["name"]
        if video_sampler is not None:
            video_sampler.set_epoch(epoch)
        if static_sampler is not None:
            static_sampler.set_epoch(epoch)

        started = time.time()
        train_metrics = train_mixed_epoch(
            model, loaders, optimizer, scaler, device, epoch, args, stage, weights, rank
        )
        video_metrics = validate_domain(
            model, loaders["video_val"], device, args, class_names, weights, rank, "video"
        )
        static_metrics = validate_domain(
            model, loaders["static_val"], device, args, class_names, weights, rank, "static"
        )
        scheduler.step()
        score = balanced_score(video_metrics, static_metrics, args.static_validation_weight)
        static_floor = (
            baseline["static"]["miou"] - args.static_retention_tolerance if baseline else float("-inf")
        )
        retained = static_metrics["miou"] >= static_floor

        if rank == 0:
            print(
                f"Epoch {epoch:03d} [{stage['name']}]: "
                f"video_train_loss={train_metrics['video_train_loss']:.4f}, "
                f"static_train_loss={train_metrics['static_train_loss']:.4f}, "
                f"video_mIoU={video_metrics['miou']:.4f}, "
                f"static_mIoU={static_metrics['miou']:.4f}, "
                f"balanced={score:.4f}, retained={retained}, "
                f"stable_gt_flip_rate={video_metrics['stable_gt_flip_rate']:.6f}, "
                f"time={time.time() - started:.1f}s"
            )
            append_metrics_csv(args.output_dir / "metrics.csv", {
                "epoch": epoch,
                "stage": stage["name"],
                **train_metrics,
                **prefixed_metrics("video", video_metrics),
                **prefixed_metrics("static", static_metrics),
                "balanced_score": score,
                "static_retained": retained,
                "static_floor": static_floor,
            })

            improved_video = video_metrics["miou"] > records["best_video"]
            improved_static = static_metrics["miou"] > records["best_static"]
            improved_balanced = score > records["best_balanced"] and retained
            records["best_video"] = max(records["best_video"], video_metrics["miou"])
            records["best_static"] = max(records["best_static"], static_metrics["miou"])
            if improved_balanced:
                records["best_balanced"] = score
            payload = checkpoint_payload(
                model, optimizer, scheduler, scaler, epoch, args, class_names, records, baseline
            )
            atomic_torch_save(payload, args.output_dir / "last.pth")
            if improved_video:
                atomic_torch_save(payload, args.output_dir / "best_video_miou.pth")
            if improved_static:
                atomic_torch_save(payload, args.output_dir / "best_static_miou.pth")
            if improved_balanced:
                atomic_torch_save(payload, args.output_dir / "best_balanced.pth")
                atomic_torch_save(payload, args.output_dir / "best_miou.pth")
            if (epoch + 1) % args.save_every == 0:
                atomic_torch_save(payload, args.output_dir / f"epoch_{epoch:03d}.pth")
            if not retained:
                print(
                    "WARNING: static validation mIoU fell below the anti-forgetting floor; "
                    "this epoch is ineligible for best_balanced.pth"
                )
        if distributed:
            dist.barrier()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
