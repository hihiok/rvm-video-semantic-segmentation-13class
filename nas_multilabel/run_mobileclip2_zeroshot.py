#!/usr/bin/env python3
"""Evaluate MobileCLIP2-S0 as a 9-label NAS photo tagger.

Primary result is strict zero-shot:
  * fixed prompt ensembles from labels.py
  * independent positive-vs-contrast score per label
  * fixed threshold 0.5
  * unknown GT (-1) masked from all metrics

A separate oracle threshold sweep is emitted for diagnosis only and MUST NOT be
reported as zero-shot performance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import open_clip
from mobileclip.modules.common.mobileone import reparameterize_model

from labels import DISPLAY_NAMES, LABELS, PROMPTS


class ManifestDataset(Dataset):
    def __init__(self, records, preprocess):
        self.records = records
        self.preprocess = preprocess

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        with Image.open(r["image_path"]) as im:
            x = self.preprocess(im.convert("RGB"))
        gt = torch.tensor([int(r["labels"][l]) for l in LABELS], dtype=torch.int8)
        return x, gt, idx


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--pretrained", default="dfndr2b")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--amp", choices=["fp32", "fp16", "bf16"], default="fp16")
    p.add_argument("--benchmark-warmup", type=int, default=50)
    p.add_argument("--benchmark-runs", type=int, default=200)
    p.add_argument("--visualize-errors", type=int, default=16)
    return p.parse_args()


def load_manifest(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise RuntimeError(f"Empty manifest: {path}")
    for r in records:
        missing = [l for l in LABELS if l not in r["labels"]]
        if missing:
            raise RuntimeError(f"Manifest sample {r.get('sample_id')} misses labels: {missing}")
    return records


def autocast_ctx(device: torch.device, amp: str):
    enabled = device.type == "cuda" and amp != "fp32"
    dtype = torch.float16 if amp == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def encode_prompt_ensemble(model, tokenizer, prompts: List[str], device, amp) -> torch.Tensor:
    tokens = tokenizer(prompts).to(device)
    with torch.no_grad(), autocast_ctx(device, amp):
        z = normalize(model.encode_text(tokens))
    z = normalize(z.float().mean(dim=0, keepdim=True))
    return z.squeeze(0)


def build_text_features(model, tokenizer, device, amp):
    pos, neg = [], []
    for label in LABELS:
        pos.append(encode_prompt_ensemble(model, tokenizer, PROMPTS[label]["positive"], device, amp))
        neg.append(encode_prompt_ensemble(model, tokenizer, PROMPTS[label]["negative"], device, amp))
    return torch.stack(pos, dim=0), torch.stack(neg, dim=0)


def get_logit_scale(model) -> float:
    if hasattr(model, "logit_scale"):
        try:
            return float(model.logit_scale.exp().detach().clamp(max=100).cpu())
        except Exception:
            pass
    return 100.0


def infer(model, loader, pos_text, neg_text, device, amp, logit_scale):
    scores, gts, indices = [], [], []
    for images, gt, idx in tqdm(loader, desc="MobileCLIP2-S0 zero-shot"):
        images = images.to(device, non_blocking=True)
        with torch.no_grad(), autocast_ctx(device, amp):
            feat = normalize(model.encode_image(images)).float()
        # Independent binary prompt pair per tag.  Do NOT apply a 9-way softmax.
        sim_pos = feat @ pos_text.T
        sim_neg = feat @ neg_text.T
        prob = torch.sigmoid(logit_scale * (sim_pos - sim_neg))
        scores.append(prob.cpu().numpy())
        gts.append(gt.numpy())
        indices.extend(idx.numpy().tolist())
    return np.concatenate(scores), np.concatenate(gts), indices


def safe_auc(gt, score):
    try:
        return float(roc_auc_score(gt, score)) if len(np.unique(gt)) == 2 else math.nan
    except Exception:
        return math.nan


def safe_ap(gt, score):
    try:
        return float(average_precision_score(gt, score)) if np.any(gt == 1) else math.nan
    except Exception:
        return math.nan


def compute_metrics(gt: np.ndarray, scores: np.ndarray, threshold: float):
    rows = []
    all_gt, all_pred = [], []
    for j, label in enumerate(LABELS):
        mask = gt[:, j] >= 0
        y = gt[mask, j].astype(int)
        s = scores[mask, j]
        p = (s >= threshold).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
        acc = accuracy_score(y, p)
        bal = balanced_accuracy_score(y, p) if len(np.unique(y)) == 2 else math.nan
        rows.append({
            "label": label,
            "display_name": DISPLAY_NAMES[label],
            "known": int(mask.sum()),
            "positive": int((y == 1).sum()),
            "negative": int((y == 0).sum()),
            "threshold": threshold,
            "precision": float(pr),
            "recall": float(rc),
            "f1": float(f1),
            "accuracy": float(acc),
            "balanced_accuracy": float(bal),
            "average_precision": safe_ap(y, s),
            "roc_auc": safe_auc(y, s),
        })
        all_gt.extend(y.tolist())
        all_pred.extend(p.tolist())
    micro_pr, micro_rc, micro_f1, _ = precision_recall_fscore_support(
        np.asarray(all_gt), np.asarray(all_pred), average="binary", zero_division=0
    )
    summary = {
        "macro_f1": float(np.mean([r["f1"] for r in rows])),
        "macro_precision": float(np.mean([r["precision"] for r in rows])),
        "macro_recall": float(np.mean([r["recall"] for r in rows])),
        "macro_accuracy": float(np.mean([r["accuracy"] for r in rows])),
        "macro_average_precision": float(np.nanmean([r["average_precision"] for r in rows])),
        "micro_precision_known_pairs": float(micro_pr),
        "micro_recall_known_pairs": float(micro_rc),
        "micro_f1_known_pairs": float(micro_f1),
        "threshold": threshold,
        "metric_protocol": "partial-GT; unknown=-1 masked; fixed zero-shot threshold",
    }
    return rows, summary


def oracle_threshold_sweep(gt, scores):
    rows = []
    thresholds = np.linspace(0.05, 0.95, 91)
    for j, label in enumerate(LABELS):
        mask = gt[:, j] >= 0
        y = gt[mask, j].astype(int)
        s = scores[mask, j]
        best = None
        for t in thresholds:
            pred = (s >= t).astype(int)
            pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
            row = (float(f1), float(pr), float(rc), float(t))
            if best is None or row > best:
                best = row
        rows.append({
            "label": label,
            "display_name": DISPLAY_NAMES[label],
            "oracle_best_f1": best[0],
            "precision_at_oracle": best[1],
            "recall_at_oracle": best[2],
            "oracle_threshold": best[3],
            "warning": "DIAGNOSTIC ONLY - threshold selected on test GT; not zero-shot result",
        })
    return rows


def write_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def benchmark_encoder(model, sample: torch.Tensor, device, amp, warmup: int, runs: int):
    x = sample.unsqueeze(0).to(device)
    if device.type == "cuda":
        for _ in range(warmup):
            with torch.no_grad(), autocast_ctx(device, amp):
                model.encode_image(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.no_grad(), autocast_ctx(device, amp):
                model.encode_image(x)
            end.record()
            torch.cuda.synchronize()
            times.append(float(start.elapsed_time(end)))
    else:
        times = []
        for _ in range(warmup):
            with torch.no_grad():
                model.encode_image(x)
        for _ in range(runs):
            t0 = time.perf_counter()
            with torch.no_grad():
                model.encode_image(x)
            times.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(times)
    mean = float(a.mean())
    return {
        "device": str(device),
        "amp": amp,
        "batch_size": 1,
        "warmup": warmup,
        "runs": runs,
        "mean_ms": mean,
        "median_ms": float(np.median(a)),
        "p90_ms": float(np.percentile(a, 90)),
        "p95_ms": float(np.percentile(a, 95)),
        "fps_from_mean_encoder_only": 1000.0 / mean if mean > 0 else math.inf,
        "scope": "image encoder only; excludes image decode/preprocess, text encoder, host-device copy and V516 runtime",
    }


def font(size=18):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"]:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def make_error_sheet(records, gt, scores, label_idx, kind, out_path: Path, limit: int, threshold: float):
    label = LABELS[label_idx]
    known = gt[:, label_idx] >= 0
    pred = scores[:, label_idx] >= threshold
    if kind == "fp":
        ids = np.where(known & (gt[:, label_idx] == 0) & pred)[0]
        ids = sorted(ids, key=lambda i: scores[i, label_idx], reverse=True)[:limit]
    else:
        ids = np.where(known & (gt[:, label_idx] == 1) & (~pred))[0]
        ids = sorted(ids, key=lambda i: scores[i, label_idx])[:limit]
    if not ids:
        return
    cell_w, cell_h = 320, 240
    cols = 4
    rows = math.ceil(len(ids) / cols)
    sheet = Image.new("RGB", (cell_w * cols, cell_h * rows), "white")
    fnt = font(16)
    for k, i in enumerate(ids):
        try:
            with Image.open(records[i]["image_path"]) as im:
                im = ImageOps.fit(im.convert("RGB"), (cell_w, cell_h - 40))
        except Exception:
            continue
        x, y = (k % cols) * cell_w, (k // cols) * cell_h
        sheet.paste(im, (x, y))
        d = ImageDraw.Draw(sheet)
        txt = f"{records[i]['sample_id']}  score={scores[i,label_idx]:.3f}  GT={gt[i,label_idx]}"
        d.rectangle((x, y + cell_h - 40, x + cell_w, y + cell_h), fill="white")
        d.text((x + 4, y + cell_h - 30), txt, fill="black", font=fnt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def write_report(path: Path, metrics, summary, latency, oracle):
    lines = [
        "# MobileCLIP2-S0 NAS 9-label Zero-shot Report",
        "",
        "## Protocol",
        "",
        "- Model: MobileCLIP2-S0 pretrained=dfndr2b.",
        "- 9 tags are evaluated independently; there is no 9-way softmax.",
        "- Primary prediction uses fixed positive-vs-contrast prompt pairs and threshold 0.5.",
        "- Ground truth uses partial labels: -1 means unknown and is excluded from metrics.",
        "- Oracle threshold results are diagnostic only and are not zero-shot performance.",
        "",
        "## Primary zero-shot summary",
        "",
        f"- macro-F1: **{summary['macro_f1']:.4f}**",
        f"- macro-precision: {summary['macro_precision']:.4f}",
        f"- macro-recall: {summary['macro_recall']:.4f}",
        f"- macro-AP: {summary['macro_average_precision']:.4f}",
        f"- micro-F1 over known label/image pairs: {summary['micro_f1_known_pairs']:.4f}",
        "",
        "| Label | Known | Pos | Neg | Precision | Recall | F1 | Accuracy | AP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in metrics:
        lines.append(
            f"| {r['display_name']} ({r['label']}) | {r['known']} | {r['positive']} | {r['negative']} | "
            f"{r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['accuracy']:.4f} | {r['average_precision']:.4f} |"
        )
    lines += [
        "",
        "## GPU image-encoder latency (not V516)",
        "",
        f"- mean: {latency['mean_ms']:.3f} ms/frame",
        f"- p95: {latency['p95_ms']:.3f} ms/frame",
        f"- encoder-only FPS from mean: {latency['fps_from_mean_encoder_only']:.2f}",
        f"- scope: {latency['scope']}",
        "",
        "## Diagnostic oracle thresholds",
        "",
        "These thresholds were selected on the same GT and MUST NOT be reported as zero-shot accuracy.",
        "",
        "| Label | Oracle threshold | Best F1 |",
        "|---|---:|---:|",
    ]
    for r in oracle:
        lines.append(f"| {r['display_name']} ({r['label']}) | {r['oracle_threshold']:.2f} | {r['oracle_best_f1']:.4f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_manifest(args.manifest)

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print("Loading MobileCLIP2-S0 / dfndr2b ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "MobileCLIP2-S0", pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer("MobileCLIP2-S0")
    model = model.to(device)
    model.eval()  # required because S0 has BatchNorm
    model = reparameterize_model(model)
    model.eval()

    pos_text, neg_text = build_text_features(model, tokenizer, device, args.amp)
    logit_scale = get_logit_scale(model)
    print(f"logit_scale={logit_scale:.4f}")

    ds = ManifestDataset(records, preprocess)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=(device.type == "cuda"), persistent_workers=args.workers > 0,
    )
    scores, gt, indices = infer(model, loader, pos_text, neg_text, device, args.amp, logit_scale)
    assert indices == list(range(len(records))), "Unexpected DataLoader ordering"

    metrics, summary = compute_metrics(gt, scores, args.threshold)
    oracle = oracle_threshold_sweep(gt, scores)
    write_csv(args.output_dir / "per_class_metrics.csv", metrics)
    write_csv(args.output_dir / "oracle_threshold_diagnostic.csv", oracle)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    pred_rows = []
    for i, r in enumerate(records):
        row = {"sample_id": r["sample_id"], "image_path": r["image_path"], "source_path": r.get("source_path", "")}
        for j, label in enumerate(LABELS):
            row[f"gt_{label}"] = int(gt[i, j])
            row[f"score_{label}"] = float(scores[i, j])
            row[f"pred_{label}"] = int(scores[i, j] >= args.threshold)
        pred_rows.append(row)
    write_csv(args.output_dir / "predictions.csv", pred_rows)

    latency = benchmark_encoder(model, ds[0][0], device, args.amp, args.benchmark_warmup, args.benchmark_runs)
    (args.output_dir / "latency.json").write_text(json.dumps(latency, indent=2), encoding="utf-8")

    vis_dir = args.output_dir / "error_visualizations"
    for j, label in enumerate(LABELS):
        make_error_sheet(records, gt, scores, j, "fp", vis_dir / f"{label}_false_positive.jpg", args.visualize_errors, args.threshold)
        make_error_sheet(records, gt, scores, j, "fn", vis_dir / f"{label}_false_negative.jpg", args.visualize_errors, args.threshold)

    prompt_dump = {
        "model": "MobileCLIP2-S0",
        "pretrained": args.pretrained,
        "threshold": args.threshold,
        "logit_scale": logit_scale,
        "prompts": PROMPTS,
    }
    (args.output_dir / "prompts.json").write_text(json.dumps(prompt_dump, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(args.output_dir / "REPORT.md", metrics, summary, latency, oracle)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(latency, indent=2, ensure_ascii=False))
    print(f"REPORT={args.output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
