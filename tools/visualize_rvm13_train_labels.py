#!/usr/bin/env python3
"""Visualize GT labels for randomly selected RVM13 train videos.

Creates one MP4 per selected video. Each output frame is a horizontal triptych:
original RGB | colorized GT label | GT overlay.

This script does not run the model and does not modify the dataset.
"""

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dataset.video_semantic import discover_video_sequences
from semantic_utils import DEFAULT_CLASS_NAMES, DEFAULT_PALETTE, colorize_mask


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--mask-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-videos", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260819)
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--overlay-alpha", type=float, default=0.45)
    p.add_argument("--max-frames", type=int, default=0, help="0 means all frames")
    return p.parse_args()


def read_mask(path: Path):
    with Image.open(path) as f:
        if f.mode not in ("L", "P", "I", "I;16"):
            raise ValueError(f"Mask must be one channel: {path}, mode={f.mode}")
        mask = np.array(f, dtype=np.int64)
    allowed = set(range(len(DEFAULT_CLASS_NAMES))) | {255}
    values = set(np.unique(mask).tolist())
    bad = sorted(values - allowed)
    if bad:
        raise ValueError(f"Unexpected mask IDs in {path}: {bad}")
    return mask


def add_title(panel, text):
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        panel,
        text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def add_legend(frame, mask):
    present = [
        int(v) for v in np.unique(mask)
        if 0 <= int(v) < len(DEFAULT_CLASS_NAMES)
    ]
    x0 = 8
    y0 = 44
    line_h = 22
    width = 245
    height = 10 + line_h * len(present)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.62, frame, 0.38, 0)

    for row, cid in enumerate(present):
        y = y0 + 20 + row * line_h
        rgb = DEFAULT_PALETTE[cid]
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        cv2.rectangle(frame, (x0 + 8, y - 12), (x0 + 24, y + 2), bgr, -1)
        cv2.putText(
            frame,
            f"{cid}: {DEFAULT_CLASS_NAMES[cid]}",
            (x0 + 32, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return frame


def make_triptych(image_bgr, mask, alpha):
    if image_bgr.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Image/mask shape mismatch: image={image_bgr.shape[:2]}, mask={mask.shape[:2]}"
        )

    vis_mask = mask.copy()
    vis_mask[vis_mask == 255] = 0
    color_rgb = colorize_mask(vis_mask.astype(np.uint8))
    color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)

    original = add_title(image_bgr.copy(), "Original")
    gt = add_title(color_bgr.copy(), "GT label")

    overlay = image_bgr.copy()
    valid_fg = (mask != 0) & (mask != 255)
    blended = cv2.addWeighted(image_bgr, 1.0 - alpha, color_bgr, alpha, 0)
    overlay[valid_fg] = blended[valid_fg]
    overlay = add_title(overlay, "GT overlay")
    overlay = add_legend(overlay, mask)

    return np.concatenate([original, gt, overlay], axis=1)


def main():
    args = parse_args()
    cv2.setNumThreads(1)

    if args.num_videos != 5:
        print(f"WARNING: requested {args.num_videos} videos; task default is 5")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("overlay alpha must be within [0,1]")

    sequences = discover_video_sequences(args.image_root, args.mask_root)
    if len(sequences) < args.num_videos:
        raise RuntimeError(f"Only {len(sequences)} videos discovered")

    rng = random.Random(args.seed)
    chosen = rng.sample(sequences, args.num_videos)
    chosen = sorted(chosen, key=lambda s: s.name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "label_visualization_manifest.csv"

    rows = []
    for idx, seq in enumerate(chosen, start=1):
        frame_pairs = list(zip(seq.image_paths, seq.mask_paths))
        if args.max_frames > 0:
            frame_pairs = frame_pairs[: args.max_frames]
        if not frame_pairs:
            raise RuntimeError(f"Empty sequence: {seq.name}")

        first = cv2.imread(str(frame_pairs[0][0]))
        if first is None:
            raise RuntimeError(f"Cannot read {frame_pairs[0][0]}")
        h, w = first.shape[:2]

        safe_name = seq.name.replace("/", "__")
        out_path = args.output_dir / f"{idx:02d}_{safe_name}.mp4"
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (w * 3, h),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create {out_path}")

        class_pixels = np.zeros(len(DEFAULT_CLASS_NAMES), dtype=np.int64)
        frame_count = 0
        try:
            for image_path, mask_path in frame_pairs:
                image = cv2.imread(str(image_path))
                if image is None:
                    raise RuntimeError(f"Cannot read {image_path}")
                if image.shape[:2] != (h, w):
                    raise RuntimeError(f"Resolution changed inside video {seq.name}")

                mask = read_mask(mask_path)
                if mask.shape != (h, w):
                    raise RuntimeError(
                        f"Image/mask resolution mismatch: {image_path} vs {mask_path}"
                    )

                valid = (mask >= 0) & (mask < len(DEFAULT_CLASS_NAMES))
                counts = np.bincount(
                    mask[valid].reshape(-1),
                    minlength=len(DEFAULT_CLASS_NAMES),
                )
                class_pixels += counts

                panel = make_triptych(image, mask, args.overlay_alpha)
                writer.write(panel)
                frame_count += 1
        finally:
            writer.release()

        cap = cv2.VideoCapture(str(out_path))
        ok, _ = cap.read()
        encoded = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if not ok:
            raise RuntimeError(f"Generated MP4 is unreadable: {out_path}")

        positive_classes = [
            DEFAULT_CLASS_NAMES[i]
            for i, count in enumerate(class_pixels)
            if count > 0
        ]
        rows.append({
            "video": seq.name,
            "input_frames": len(seq.image_paths),
            "visualized_frames": frame_count,
            "encoded_frames": encoded,
            "width": w,
            "height": h,
            "classes_present": ",".join(positive_classes),
            "output": str(out_path),
            "status": "PASS",
        })
        print(
            f"[{idx}/{len(chosen)}] {seq.name}: frames={frame_count}, "
            f"classes={positive_classes}, output={out_path}"
        )

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("TRAIN_LABEL_VISUALIZATION=PASS")
    print("VIDEOS=", len(rows))
    print("SEED=", args.seed)
    print("OUTPUT_DIR=", args.output_dir)
    print("MANIFEST=", manifest_path)


if __name__ == "__main__":
    main()
