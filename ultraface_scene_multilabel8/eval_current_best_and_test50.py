#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, os, random, shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model import create_ultraface_slim_scene8

LABELS = ["night", "indoor", "rain_snow", "office", "outdoor", "landscape", "sports", "objective_image"]
DISPLAY = {
    "night": "night",
    "indoor": "indoor",
    "rain_snow": "rain/snow",
    "office": "office",
    "outdoor": "outdoor",
    "landscape": "landscape",
    "sports": "sports",
    "objective_image": "objective",
}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate current best UltraFace slim scene8 checkpoint and visualize 50 test images")
    p.add_argument("--checkpoint", type=Path, required=True, help="best_train_state.pth from the running V1 training")
    p.add_argument("--data-root", type=Path, required=True, help="manifest root containing train/val/test.jsonl")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--num-vis", type=int, default=50)
    p.add_argument("--seed", type=int, default=20260909)
    p.add_argument("--amp", action="store_true")
    return p.parse_args()


def load_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


class EvalDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        im = cv2.imread(r["image"], cv2.IMREAD_COLOR)
        if im is None:
            raise RuntimeError(f"cannot read image: {r['image']}")
        if im.shape[0] != 360 or im.shape[1] != 640:
            im = cv2.resize(im, (640, 360), interpolation=cv2.INTER_LINEAR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)
        im = (im - 127.0) / 128.0
        x = torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1))).float()
        y = torch.tensor([float(r["labels"][k]) for k in LABELS], dtype=torch.float32)
        return x, y, idx


def worker_init(_):
    cv2.setNumThreads(0)
    torch.set_num_threads(1)


def make_loader(rows, batch, workers):
    kw = dict(batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True, worker_init_fn=worker_init)
    if workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(EvalDataset(rows), **kw)


def load_model(checkpoint: Path, device):
    ck = torch.load(checkpoint, map_location="cpu")
    if "model" not in ck:
        raise RuntimeError(f"checkpoint does not contain model state: {checkpoint}")
    model = create_ultraface_slim_scene8(dropout=0.1)
    model.load_state_dict(ck["model"], strict=True)
    model.to(device).eval()
    return model, ck


def infer_all(model, loader, device, amp):
    gt, scores, indices = [], [], []
    with torch.no_grad():
        for x, y, idx in loader:
            x = x.to(device, non_blocking=True)
            use_amp = amp and device.type == "cuda"
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            if logits.ndim != 2 or logits.shape[1] != 8:
                raise RuntimeError(f"expected [B,8] logits, got {tuple(logits.shape)}")
            gt.append(y.numpy())
            scores.append(torch.sigmoid(logits).float().cpu().numpy())
            indices.extend(idx.tolist())
    return np.concatenate(gt), np.concatenate(scores), np.asarray(indices, dtype=np.int64)


def binary_ap(y, s):
    if int((y == 1).sum()) == 0:
        return float("nan")
    order = np.argsort(-s)
    yy = y[order]
    tp = np.cumsum(yy == 1)
    fp = np.cumsum(yy == 0)
    precision = tp / np.maximum(tp + fp, 1)
    return float(precision[yy == 1].sum() / max(int((yy == 1).sum()), 1))


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
        acc = (tp + tn) / max(len(y), 1)
        tpr = tp / max(tp + fn, 1)
        tnr = tn / max(tn + fp, 1)
        ba = 0.5 * (tpr + tnr)
        rows.append({
            "label": label,
            "threshold": float(thresholds[j]),
            "known": int(mask.sum()),
            "positive": int((y == 1).sum()),
            "negative": int((y == 0).sum()),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(acc),
            "balanced_accuracy": float(ba),
            "ap": binary_ap(y, s),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    summary = {
        "macro_precision": float(np.mean([r["precision"] for r in rows])),
        "macro_recall": float(np.mean([r["recall"] for r in rows])),
        "macro_f1": float(np.mean([r["f1"] for r in rows])),
        "macro_balanced_accuracy": float(np.mean([r["balanced_accuracy"] for r in rows])),
        "macro_ap": float(np.nanmean([r["ap"] for r in rows])),
    }
    return rows, summary


def calibrate(gt, scores):
    out = np.full(8, 0.5, dtype=np.float32)
    grid = np.linspace(0.05, 0.95, 91)
    for j in range(8):
        mask = gt[:, j] >= 0
        y = gt[mask, j].astype(np.int64)
        s = scores[mask, j]
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            pred = (s >= t).astype(np.int64)
            tp = int(((pred == 1) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            pr = tp / max(tp + fp, 1)
            rc = tp / max(tp + fn, 1)
            f1 = 2 * pr * rc / max(pr + rc, 1e-12)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        out[j] = best_t
    return out


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def choose_balanced_50(rows, gt, num_vis, seed):
    rng = random.Random(seed)
    chosen = []
    chosen_set = set()
    per_class = max(3, num_vis // (2 * len(LABELS)))
    for j in range(len(LABELS)):
        pos = [i for i in range(len(rows)) if gt[i, j] == 1]
        rng.shuffle(pos)
        for i in pos[:per_class]:
            if i not in chosen_set:
                chosen.append(i); chosen_set.add(i)
    rest = list(range(len(rows)))
    rng.shuffle(rest)
    for i in rest:
        if len(chosen) >= num_vis:
            break
        if i not in chosen_set:
            chosen.append(i); chosen_set.add(i)
    return chosen[:num_vis]


def draw_visual(src_path: str, gt_row, score_row, thresholds, index: int, out_path: Path):
    im = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"cannot read {src_path}")
    if im.shape[1] > 960:
        scale = 960.0 / im.shape[1]
        im = cv2.resize(im, (960, int(im.shape[0] * scale)))
    panel_h = 280
    canvas = np.full((im.shape[0] + panel_h, im.shape[1], 3), 245, dtype=np.uint8)
    canvas[:im.shape[0]] = im
    y0 = im.shape[0] + 28
    known_pos = [LABELS[j] for j in range(8) if gt_row[j] == 1]
    known_neg = [LABELS[j] for j in range(8) if gt_row[j] == 0]
    pred_pos = [LABELS[j] for j in range(8) if score_row[j] >= thresholds[j]]
    cv2.putText(canvas, f"#{index:02d} GT+ : {', '.join(known_pos) if known_pos else '(none)'}", (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20,20,20), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"PRED+: {', '.join(pred_pos) if pred_pos else '(none)'}", (12, y0+28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20,20,20), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Known negative labels: {len(known_neg)} | unknown labels omitted from GT", (12, y0+56), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80,80,80), 1, cv2.LINE_AA)
    yy = y0 + 88
    for j, label in enumerate(LABELS):
        gt_txt = "1" if gt_row[j] == 1 else "0" if gt_row[j] == 0 else "?"
        pred = int(score_row[j] >= thresholds[j])
        ok = (gt_row[j] < 0) or (pred == int(gt_row[j]))
        color = (40,120,40) if ok else (40,40,200)
        text = f"{DISPLAY[label]:10s} score={score_row[j]:.3f}  th={thresholds[j]:.2f}  pred={pred}  gt={gt_txt}"
        cv2.putText(canvas, text, (12 + (j % 2) * (im.shape[1] // 2), yy + (j // 2) * 36), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])


def make_contact_sheet(vis_paths, out_path: Path, cols=5):
    tiles = []
    tw, th = 320, 260
    for p in vis_paths:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            continue
        h, w = im.shape[:2]
        crop_h = min(h, int(w * 0.8))
        im = im[:crop_h]
        tile = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
        tiles.append(tile)
    if not tiles:
        return
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows * th, cols * tw, 3), 255, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r*th:(r+1)*th, c*tw:(c+1)*tw] = tile
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    a = parse_args()
    random.seed(a.seed); np.random.seed(a.seed); cv2.setNumThreads(0)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = a.output_dir / "best_train_state_snapshot.pth"
    shutil.copy2(a.checkpoint, snapshot)

    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(a.device)
    model, ck = load_model(snapshot, device)
    print("CHECKPOINT_EPOCH", ck.get("epoch", "unknown"), flush=True)
    print("CHECKPOINT_BEST_MACRO_F1_0P5", ck.get("best_macro_f1", "unknown"), flush=True)

    val_rows = load_jsonl(a.data_root / "val.jsonl")
    test_rows = load_jsonl(a.data_root / "test.jsonl")
    val_loader = make_loader(val_rows, a.batch_size, a.workers)
    test_loader = make_loader(test_rows, a.batch_size, a.workers)

    gt_val, sc_val, _ = infer_all(model, val_loader, device, a.amp)
    thresholds = calibrate(gt_val, sc_val)
    val_metrics, val_summary = per_class_metrics(gt_val, sc_val, thresholds)
    save_csv(a.output_dir / "val_per_class_calibrated.csv", val_metrics)

    gt_test, sc_test, test_indices = infer_all(model, test_loader, device, a.amp)
    test_metrics, test_summary = per_class_metrics(gt_test, sc_test, thresholds)
    save_csv(a.output_dir / "test_per_class_calibrated.csv", test_metrics)
    (a.output_dir / "thresholds.json").write_text(json.dumps(dict(zip(LABELS, map(float, thresholds))), indent=2), encoding="utf-8")
    report = {
        "checkpoint": str(snapshot),
        "checkpoint_epoch": ck.get("epoch"),
        "checkpoint_best_macro_f1_0p5": ck.get("best_macro_f1"),
        "val_summary_calibrated": val_summary,
        "test_summary_calibrated": test_summary,
        "labels": LABELS,
    }
    (a.output_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    chosen = choose_balanced_50(test_rows, gt_test, min(a.num_vis, len(test_rows)), a.seed)
    vis_dir = a.output_dir / "test50_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    pred_rows = []
    vis_paths = []
    for rank, i in enumerate(chosen, 1):
        r = test_rows[i]
        gt = gt_test[i]
        sc = sc_test[i]
        pred = (sc >= thresholds).astype(np.int64)
        out = vis_dir / f"{rank:02d}_{Path(r['image']).stem}.jpg"
        draw_visual(r["image"], gt, sc, thresholds, rank, out)
        vis_paths.append(out)
        row = {
            "rank": rank,
            "image": r["image"],
            "source": r.get("source", ""),
            "detail": r.get("detail", ""),
            "gt_positive": ";".join([LABELS[j] for j in range(8) if gt[j] == 1]),
            "pred_positive": ";".join([LABELS[j] for j in range(8) if pred[j] == 1]),
        }
        for j, label in enumerate(LABELS):
            row[f"score_{label}"] = float(sc[j])
            row[f"threshold_{label}"] = float(thresholds[j])
            row[f"gt_{label}"] = int(gt[j])
            row[f"pred_{label}"] = int(pred[j])
        pred_rows.append(row)
    save_csv(a.output_dir / "test50_predictions.csv", pred_rows)
    make_contact_sheet(vis_paths, a.output_dir / "test50_contact_sheet.jpg", cols=5)

    print("\nTEST_PER_CLASS_CALIBRATED")
    print("label,precision,recall,f1,balanced_accuracy,ap,known,pos,neg,threshold")
    for r in test_metrics:
        print(f"{r['label']},{r['precision']:.4f},{r['recall']:.4f},{r['f1']:.4f},{r['balanced_accuracy']:.4f},{r['ap']:.4f},{r['known']},{r['positive']},{r['negative']},{r['threshold']:.2f}")
    print("TEST_SUMMARY", json.dumps(test_summary), flush=True)
    print("TEST50_DIR", vis_dir, flush=True)
    print("CONTACT_SHEET", a.output_dir / "test50_contact_sheet.jpg", flush=True)


if __name__ == "__main__":
    main()
