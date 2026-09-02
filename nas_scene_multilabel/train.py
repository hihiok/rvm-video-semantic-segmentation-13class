#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, balanced_accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import DISPLAY_NAMES, LABELS
from model import create_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--base-channel", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260902)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--init-backbone", type=Path, default=None)
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument("--cpu-threads", type=int, default=4)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_records(path: Path):
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    if not out:
        raise RuntimeError(f"Empty manifest: {path}")
    return out


class MultiLabelDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        with Image.open(r["image"]) as im:
            x = self.transform(im.convert("RGB"))
        y = torch.tensor([float(r["labels"][l]) for l in LABELS], dtype=torch.float32)
        return x, y


def build_transforms(size):
    norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    train_t = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.65, 1.0), ratio=(0.75, 1.3333)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.03),
        transforms.RandomGrayscale(p=0.03),
        transforms.ToTensor(),
        norm,
        transforms.RandomErasing(p=0.10, scale=(0.02, 0.12), ratio=(0.5, 2.0), value=0),
    ])
    eval_t = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        norm,
    ])
    return train_t, eval_t


def compute_pos_weight(records):
    pos = np.zeros(len(LABELS), dtype=np.float64)
    neg = np.zeros(len(LABELS), dtype=np.float64)
    for r in records:
        for j, l in enumerate(LABELS):
            v = r["labels"][l]
            if v == 1:
                pos[j] += 1
            elif v == 0:
                neg[j] += 1
    w = neg / np.maximum(pos, 1.0)
    w = np.clip(w, 0.5, 8.0)
    return torch.tensor(w, dtype=torch.float32), pos.astype(int), neg.astype(int)


def masked_bce(logits, target, pos_weight):
    mask = target >= 0
    target01 = target.clamp(0, 1)
    raw = F.binary_cross_entropy_with_logits(logits, target01, reduction="none", pos_weight=pos_weight)
    return (raw * mask).sum() / mask.sum().clamp_min(1)


def metrics_from_arrays(gt, scores, thresholds):
    rows = []
    for j, l in enumerate(LABELS):
        mask = gt[:, j] >= 0
        y = gt[mask, j].astype(int)
        s = scores[mask, j]
        pred = (s >= thresholds[j]).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        acc = float((pred == y).mean()) if len(y) else math.nan
        bal = float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) == 2 else math.nan
        try:
            ap = float(average_precision_score(y, s)) if np.any(y == 1) else math.nan
        except Exception:
            ap = math.nan
        rows.append({
            "label": l, "display_name": DISPLAY_NAMES[l], "known": int(mask.sum()),
            "positive": int((y == 1).sum()), "negative": int((y == 0).sum()),
            "threshold": float(thresholds[j]), "precision": float(pr), "recall": float(rc),
            "f1": float(f1), "accuracy": acc, "balanced_accuracy": bal, "ap": ap,
        })
    summary = {
        "macro_f1": float(np.mean([r["f1"] for r in rows])),
        "macro_precision": float(np.mean([r["precision"] for r in rows])),
        "macro_recall": float(np.mean([r["recall"] for r in rows])),
        "macro_accuracy": float(np.mean([r["accuracy"] for r in rows])),
        "macro_balanced_accuracy": float(np.nanmean([r["balanced_accuracy"] for r in rows])),
        "macro_ap": float(np.nanmean([r["ap"] for r in rows])),
    }
    return rows, summary


def calibrate_thresholds(gt, scores):
    ts = np.full(len(LABELS), 0.5, dtype=np.float32)
    grid = np.linspace(0.05, 0.95, 91)
    for j in range(len(LABELS)):
        mask = gt[:, j] >= 0
        y = gt[mask, j].astype(int)
        s = scores[mask, j]
        best = (-1.0, 0.5)
        for t in grid:
            pred = (s >= t).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
            if f1 > best[0]:
                best = (float(f1), float(t))
        ts[j] = best[1]
    return ts


def evaluate(model, loader, device):
    model.eval()
    scores, gts = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
            scores.append(torch.sigmoid(logits).float().cpu().numpy())
            gts.append(y.numpy())
    return np.concatenate(gts), np.concatenate(scores)


def save_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def try_load_backbone(model, path):
    if path is None:
        return
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
    own = model.state_dict()
    matched = {}
    for k, v in state.items():
        k2 = k.replace("module.", "")
        candidates = [k2, k2.replace("base_net.", "backbone."), k2.replace("model.", "backbone.")]
        for c in candidates:
            if c in own and own[c].shape == v.shape:
                matched[c] = v; break
    model.load_state_dict(matched, strict=False)
    print(f"INIT_BACKBONE matched={len(matched)}/{len(own)} from {path}")


def main():
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", str(args.cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.cpu_threads))
    torch.set_num_threads(args.cpu_threads)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA is required for formal training")
    torch.backends.cudnn.benchmark = True

    train_records = load_records(args.data_root / "train.jsonl")
    val_records = load_records(args.data_root / "val.jsonl")
    test_records = load_records(args.data_root / "test.jsonl")
    train_t, eval_t = build_transforms(args.input_size)
    train_ds = MultiLabelDataset(train_records, train_t)
    val_ds = MultiLabelDataset(val_records, eval_t)
    test_ds = MultiLabelDataset(test_records, eval_t)

    loader_kw = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)

    pos_weight, pos_count, neg_count = compute_pos_weight(train_records)
    pos_weight = pos_weight.to(device)
    print("TRAIN_SUPERVISION", {l: {"pos": int(pos_count[i]), "neg": int(neg_count[i]), "pos_weight": float(pos_weight[i])} for i, l in enumerate(LABELS)})

    model = create_model(len(LABELS), args.base_channel, args.dropout).to(device).to(memory_format=torch.channels_last)
    try_load_backbone(model, args.init_backbone)
    params = sum(p.numel() for p in model.parameters())
    print(f"MODEL_PARAMS={params} ({params/1e6:.4f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    start_epoch, best_f1 = 0, -1.0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = int(ck["epoch"]) + 1
        best_f1 = float(ck.get("best_macro_f1", -1.0))

    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = args.warmup_epochs * len(train_loader)
    global_step = start_epoch * len(train_loader)

    def set_lr(step):
        if step < warmup_steps:
            factor = max(step + 1, 1) / max(warmup_steps, 1)
        else:
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            factor = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        lr = args.lr * factor
        for g in optimizer.param_groups:
            g["lr"] = lr
        return lr

    history_path = args.output_dir / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_sum = 0.0; n = 0; t0 = time.time()
        for step, (x, y) in enumerate(train_loader):
            lr = set_lr(global_step); global_step += 1
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss = masked_bce(logits, y, pos_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            loss_sum += float(loss.item()); n += 1
            if step % args.print_every == 0:
                print(f"epoch={epoch:03d} step={step:05d}/{len(train_loader)} loss={loss.item():.5f} lr={lr:.3e}")

        gt_val, sc_val = evaluate(model, val_loader, device)
        rows05, sum05 = metrics_from_arrays(gt_val, sc_val, np.full(len(LABELS), 0.5))
        rec = {"epoch": epoch, "train_loss": loss_sum/max(n,1), "seconds": time.time()-t0, **sum05}
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print("VAL", json.dumps(rec, indent=2))

        state = {
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "best_macro_f1": best_f1, "labels": LABELS, "input_size": args.input_size,
            "base_channel": args.base_channel,
        }
        torch.save(state, args.output_dir / "last.pth")
        if sum05["macro_f1"] > best_f1:
            best_f1 = sum05["macro_f1"]
            state["best_macro_f1"] = best_f1
            torch.save(state, args.output_dir / "best_macro_f1.pth")
            save_rows(args.output_dir / "best_val_per_class_0p5.csv", rows05)

    best = torch.load(args.output_dir / "best_macro_f1.pth", map_location="cpu")
    model.load_state_dict(best["model"])
    gt_val, sc_val = evaluate(model, val_loader, device)
    thresholds = calibrate_thresholds(gt_val, sc_val)
    gt_test, sc_test = evaluate(model, test_loader, device)
    test_rows, test_summary = metrics_from_arrays(gt_test, sc_test, thresholds)
    save_rows(args.output_dir / "test_per_class_calibrated.csv", test_rows)
    (args.output_dir / "test_summary.json").write_text(json.dumps(test_summary, indent=2), encoding="utf-8")
    (args.output_dir / "thresholds.json").write_text(json.dumps({l: float(thresholds[i]) for i,l in enumerate(LABELS)}, indent=2), encoding="utf-8")
    best["thresholds"] = {l: float(thresholds[i]) for i,l in enumerate(LABELS)}
    best["test_summary"] = test_summary
    torch.save(best, args.output_dir / "best_deploy.pth")
    print("TEST", json.dumps(test_summary, indent=2))


if __name__ == "__main__":
    main()
