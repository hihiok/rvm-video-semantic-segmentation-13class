#!/usr/bin/env python3
"""Train the existing FSD/UltraFace scene network as an 8-label tagger at 640x360.

This script does NOT rewrite the FSD backbone. It imports and instantiates the
existing factory `create_Mb_Tiny_RFB_fd_3_scene_noRFB(2, 8)`, uses the existing
FSD YUV444 scene transforms, and changes only the data/objective interface from
single-label CE to partial-label BCEWithLogitsLoss.

Important resolution detail:
The reference FSD code uses a scalar `define_img_size(240)` which resolves to the
legacy UltraFace 4:3 240x320 setting. For this 16:9 task we bootstrap the FSD
module with a supported scalar size (`640`) but pass the actual explicit scene
transform size `[360, 640]`. Runtime checks require each transformed sample to be
`[1, 360, 640]` and the FSD factory output to be `[B, 8]`.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset

LABELS = ["night", "indoor", "rain_snow", "office", "outdoor", "landscape", "sports", "objective_image"]
DISPLAY_NAMES = {
    "night": "夜景", "indoor": "室内", "rain_snow": "雨/雪", "office": "办公场景",
    "outdoor": "户外", "landscape": "风景", "sports": "运动", "objective_image": "客观图",
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "y", "on")


def parse_args():
    p = argparse.ArgumentParser(description="FSD 8-label multi-label scene training at explicit 640x360")
    p.add_argument("--fsd-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--input-width", type=int, default=640)
    p.add_argument("--input-height", type=int, default=360)
    p.add_argument("--fd-bootstrap-size", type=int, default=640,
                   help="Only initializes the existing FSD config; actual scene tensor uses input-height/input-width")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--base-net-lr", type=float, default=None)
    p.add_argument("--extra-layers-lr", type=float, default=None)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--milestones", default="95,150")
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=20260904)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument("--base-net", type=Path, default=None)
    p.add_argument("--pretrained-ssd", type=Path, default=None)
    p.add_argument("--resume-train-state", type=Path, default=None)
    p.add_argument("--amp", type=str2bool, default=True)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--max-train-steps", type=int, default=0,
                   help="0 means full epoch; positive value is only for smoke/debug")
    p.add_argument("--max-eval-batches", type=int, default=0,
                   help="0 means full validation/test; positive value is only for smoke/debug")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"Empty manifest: {path}")
    return rows


def _to_tensor_image(value):
    if isinstance(value, torch.Tensor):
        x = value
    else:
        x = torch.from_numpy(np.asarray(value))
    if x.ndim == 2:
        x = x.unsqueeze(0)
    elif x.ndim == 3 and x.shape[0] not in (1, 3) and x.shape[-1] in (1, 3):
        x = x.permute(2, 0, 1)
    if x.ndim != 3:
        raise RuntimeError(f"Unexpected transformed image shape: {tuple(x.shape)}")
    return x.float().contiguous()


def apply_fsd_scene_transform(transform, image_bgr):
    """Support the two scene-transform call conventions used in FSD forks."""
    first_error = None
    try:
        out = transform(image_bgr)
    except Exception as e:
        first_error = e
        try:
            out = transform(image_bgr, None, None)
        except Exception as e2:
            raise RuntimeError(
                "FSD scene transform failed with both transform(image) and "
                "transform(image,None,None). First error={!r}; second={!r}".format(first_error, e2)
            )
    if isinstance(out, (tuple, list)):
        if not out:
            raise RuntimeError("FSD transform returned empty tuple/list")
        out = out[0]
    return _to_tensor_image(out)


class ManifestDataset(Dataset):
    def __init__(self, records, transform, expected_hw):
        self.records = records
        self.transform = transform
        self.expected_hw = tuple(expected_hw)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        image = cv2.imread(r["image"], cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {r['image']}")
        x = apply_fsd_scene_transform(self.transform, image)
        if x.shape[0] != 1 or tuple(x.shape[-2:]) != self.expected_hw:
            raise RuntimeError(
                f"Expected transformed FSD scene tensor [1,{self.expected_hw[0]},{self.expected_hw[1]}], "
                f"got {tuple(x.shape)} for {r['image']}"
            )
        y = torch.tensor([float(r["labels"][l]) for l in LABELS], dtype=torch.float32)
        return x, y


def compute_pos_weight(records):
    pos = np.zeros(len(LABELS), dtype=np.float64)
    neg = np.zeros(len(LABELS), dtype=np.float64)
    for r in records:
        for j, l in enumerate(LABELS):
            v = int(r["labels"][l])
            if v == 1:
                pos[j] += 1
            elif v == 0:
                neg[j] += 1
    w = neg / np.maximum(pos, 1.0)
    w = np.clip(w, 0.5, 8.0)
    return torch.tensor(w, dtype=torch.float32), pos.astype(int), neg.astype(int)


def masked_bce(logits, targets, pos_weight):
    mask = targets >= 0
    if not bool(mask.any()):
        return logits.sum() * 0.0
    t = targets.clamp(0, 1)
    raw = F.binary_cross_entropy_with_logits(logits, t, reduction="none", pos_weight=pos_weight)
    return (raw * mask).sum() / mask.sum().clamp_min(1)


def binary_ap(y, score):
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    positives = int((y == 1).sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score)
    yy = y[order]
    tp = np.cumsum(yy == 1)
    fp = np.cumsum(yy == 0)
    precision = tp / np.maximum(tp + fp, 1)
    return float(precision[yy == 1].sum() / positives)


def per_class_metrics(gt, scores, thresholds):
    rows = []
    for j, label in enumerate(LABELS):
        mask = gt[:, j] >= 0
        y = gt[mask, j].astype(np.int64)
        s = scores[mask, j]
        pred = (s >= thresholds[j]).astype(np.int64)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        accuracy = (tp + tn) / max(len(y), 1)
        tpr = tp / max(tp + fn, 1)
        tnr = tn / max(tn + fp, 1)
        balanced = 0.5 * (tpr + tnr)
        rows.append({
            "label": label,
            "display_name": DISPLAY_NAMES[label],
            "threshold": float(thresholds[j]),
            "known": int(mask.sum()),
            "positive": int((y == 1).sum()),
            "negative": int((y == 0).sum()),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
            "balanced_accuracy": float(balanced),
            "ap": binary_ap(y, s),
        })
    summary = {
        "macro_f1": float(np.mean([r["f1"] for r in rows])),
        "macro_balanced_accuracy": float(np.mean([r["balanced_accuracy"] for r in rows])),
        "macro_ap": float(np.nanmean([r["ap"] for r in rows])),
    }
    return rows, summary


def calibrate_thresholds(gt, scores):
    result = np.full(len(LABELS), 0.5, dtype=np.float32)
    grid = np.linspace(0.05, 0.95, 91)
    for j in range(len(LABELS)):
        mask = gt[:, j] >= 0
        y = gt[mask, j].astype(np.int64)
        s = scores[mask, j]
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            pred = (s >= t).astype(np.int64)
            tp = ((pred == 1) & (y == 1)).sum()
            fp = ((pred == 1) & (y == 0)).sum()
            fn = ((pred == 0) & (y == 1)).sum()
            pr = tp / max(tp + fp, 1)
            rc = tp / max(tp + fn, 1)
            f1 = 2 * pr * rc / max(pr + rc, 1e-12)
            if f1 > best_f1:
                best_f1, best_t = float(f1), float(t)
        result[j] = best_t
    return result


def evaluate(model, loader, device, use_amp, max_batches=0):
    model.eval()
    all_gt = []
    all_scores = []
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.numpy()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            if logits.ndim != 2 or logits.shape[1] != len(LABELS):
                raise RuntimeError(f"Expected FSD scene logits [B,{len(LABELS)}], got {tuple(logits.shape)}")
            all_scores.append(torch.sigmoid(logits).float().cpu().numpy())
            all_gt.append(y)
    if not all_gt:
        raise RuntimeError("No evaluation batches were produced")
    return np.concatenate(all_gt), np.concatenate(all_scores)


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def main():
    args = parse_args()
    if (args.input_width, args.input_height) != (640, 360):
        raise ValueError(
            f"This branch is the fixed 640x360 baseline; got width={args.input_width}, height={args.input_height}"
        )
    os.environ.setdefault("OMP_NUM_THREADS", str(args.cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.cpu_threads))
    torch.set_num_threads(args.cpu_threads)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not (args.fsd_root / "vision" / "ssd" / "mb_tiny_RFB_fd_3.py").exists():
        raise FileNotFoundError(f"Not an FSD repo: {args.fsd_root}")
    sys.path.insert(0, str(args.fsd_root.resolve()))

    from vision.ssd.config.fd_config import define_img_size
    define_img_size(args.fd_bootstrap_size)
    from vision.ssd.data_preprocessing import YUV444TrainAugmentation_scene, YUV444TestTransform_scene
    from vision.ssd.mb_tiny_RFB_fd_3 import create_Mb_Tiny_RFB_fd_3_scene_noRFB

    scene_hw = [args.input_height, args.input_width]
    train_tf = YUV444TrainAugmentation_scene(scene_hw)
    eval_tf = YUV444TestTransform_scene(scene_hw)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required")
    torch.backends.cudnn.benchmark = True

    train_records = load_jsonl(args.data_root / "train.jsonl")
    val_records = load_jsonl(args.data_root / "val.jsonl")
    test_records = load_jsonl(args.data_root / "test.jsonl")
    expected_hw = (args.input_height, args.input_width)
    train_ds = ManifestDataset(train_records, train_tf, expected_hw)
    val_ds = ManifestDataset(val_records, eval_tf, expected_hw)
    test_ds = ManifestDataset(test_records, eval_tf, expected_hw)

    # Fail before the long run if the existing FSD transform does not honor 640x360.
    sample_x, _ = train_ds[0]
    logging.info("FIRST_TRANSFORMED_SAMPLE_SHAPE=%s", tuple(sample_x.shape))

    loader_kw = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)

    pos_weight, pos_count, neg_count = compute_pos_weight(train_records)
    pos_weight = pos_weight.to(device)
    logging.info(
        "TRAIN_SUPERVISION %s",
        {l: {"pos": int(pos_count[i]), "neg": int(neg_count[i]), "pos_weight": float(pos_weight[i].cpu())}
         for i, l in enumerate(LABELS)},
    )

    model = create_Mb_Tiny_RFB_fd_3_scene_noRFB(2, len(LABELS))
    if args.base_net:
        model.init_from_base_net_scene(str(args.base_net))
    elif args.pretrained_ssd:
        model.init_from_pretrained_ssd_scene(str(args.pretrained_ssd))
    model.to(device)

    # Mandatory spatial-compatibility check for the existing FSD scene head.
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 1, args.input_height, args.input_width, device=device)
        dummy_logits = model(dummy)
    logging.info("FSD_FACTORY_DUMMY_OUTPUT_SHAPE=%s", tuple(dummy_logits.shape))
    if tuple(dummy_logits.shape) != (1, len(LABELS)):
        raise RuntimeError(
            f"Existing FSD scene factory is not 640x360-compatible: expected (1,{len(LABELS)}), "
            f"got {tuple(dummy_logits.shape)}"
        )

    base_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr
    params = [
        {"params": model.base_net.parameters(), "lr": base_lr},
        {"params": itertools.chain(model.source_layer_add_ons.parameters(), model.extras.parameters()), "lr": extra_lr},
        {"params": model.scene_headers.parameters(), "lr": args.lr},
    ]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    milestones = [int(v.strip()) for v in args.milestones.split(",") if v.strip()]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    start_epoch = 0
    best_f1 = -1.0
    if args.resume_train_state:
        ck = torch.load(args.resume_train_state, map_location="cpu")
        ck_w = int(ck.get("input_width", args.input_width))
        ck_h = int(ck.get("input_height", args.input_height))
        if (ck_w, ck_h) != (args.input_width, args.input_height):
            raise RuntimeError(f"Resume resolution mismatch: checkpoint={ck_w}x{ck_h}, run={args.input_width}x{args.input_height}")
        model.load_state_dict(ck["model"], strict=True)
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck.get("scaler", scaler.state_dict()))
        start_epoch = int(ck["epoch"]) + 1
        best_f1 = float(ck.get("best_macro_f1", -1.0))

    history = args.output_dir / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_sum = 0.0
        steps = 0
        t0 = time.time()
        for step, (x, y) in enumerate(train_loader):
            if args.max_train_steps > 0 and step >= args.max_train_steps:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                logits = model(x)
                if logits.ndim != 2 or logits.shape[1] != len(LABELS):
                    raise RuntimeError(f"Expected logits [B,{len(LABELS)}], got {tuple(logits.shape)}")
                loss = masked_bce(logits, y, pos_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch} step={step}: {loss}")
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.item())
            steps += 1
            if step > 0 and step % args.print_every == 0:
                logging.info(
                    "epoch=%03d step=%05d/%05d loss=%.5f lr=%.3e",
                    epoch, step, len(train_loader), loss_sum / max(steps, 1), optimizer.param_groups[0]["lr"],
                )
        if steps == 0:
            raise RuntimeError("Training produced zero optimizer steps")
        scheduler.step()

        gt_val, sc_val = evaluate(model, val_loader, device, args.amp, args.max_eval_batches)
        rows, summary = per_class_metrics(gt_val, sc_val, np.full(len(LABELS), 0.5))
        rec = {
            "epoch": epoch,
            "train_loss": loss_sum / max(steps, 1),
            "steps": steps,
            "seconds": time.time() - t0,
            "input_width": args.input_width,
            "input_height": args.input_height,
            **summary,
        }
        with history.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        logging.info("VAL %s", json.dumps(rec))

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_macro_f1": best_f1,
            "labels": LABELS,
            "input_width": args.input_width,
            "input_height": args.input_height,
            "fd_bootstrap_size": args.fd_bootstrap_size,
        }
        torch.save(state, args.output_dir / "last_train_state.pth")
        if summary["macro_f1"] > best_f1:
            best_f1 = summary["macro_f1"]
            state["best_macro_f1"] = best_f1
            torch.save(state, args.output_dir / "best_train_state.pth")
            save_csv(args.output_dir / "best_val_per_class_0p5.csv", rows)

    best = torch.load(args.output_dir / "best_train_state.pth", map_location="cpu")
    model.load_state_dict(best["model"], strict=True)
    gt_val, sc_val = evaluate(model, val_loader, device, args.amp, args.max_eval_batches)
    thresholds = calibrate_thresholds(gt_val, sc_val)
    gt_test, sc_test = evaluate(model, test_loader, device, args.amp, args.max_eval_batches)
    test_rows, test_summary = per_class_metrics(gt_test, sc_test, thresholds)
    save_csv(args.output_dir / "test_per_class_calibrated.csv", test_rows)
    (args.output_dir / "test_summary.json").write_text(
        json.dumps(test_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    threshold_dict = {l: float(thresholds[i]) for i, l in enumerate(LABELS)}
    (args.output_dir / "thresholds.json").write_text(
        json.dumps(threshold_dict, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # FSD-compatible classifier-only checkpoint, loadable by existing net.load().
    unwrap(model).save(str(args.output_dir / "best_fsd_multilabel8_640x360.pth"))
    metadata = {
        "labels": LABELS,
        "display_names": DISPLAY_NAMES,
        "thresholds": threshold_dict,
        "input_width": args.input_width,
        "input_height": args.input_height,
        "input_tensor_shape": [1, args.input_height, args.input_width],
        "fd_bootstrap_size": args.fd_bootstrap_size,
        "factory": "create_Mb_Tiny_RFB_fd_3_scene_noRFB(2, 8)",
        "preprocess": "existing FSD YUV444 scene transform with explicit size [360,640]",
        "loss": "masked BCEWithLogitsLoss",
        "output": "8 raw logits; sigmoid + per-class threshold",
    }
    (args.output_dir / "deployment_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logging.info("TEST %s", json.dumps(test_summary))


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
