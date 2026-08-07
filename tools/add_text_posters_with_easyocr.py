#!/usr/bin/env python3
"""OCR-filter VIPSeg painting_or_poster instances before adding them as text."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vipseg-root", type=Path, required=True, help="Original VIPSeg root with images/ and panomasks/")
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--converted-root", type=Path, required=True, help="Strict 13-class output to update")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--languages", default="ch_sim,en")
    parser.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-confidence", type=float, default=0.20)
    parser.add_argument("--min-characters", type=int, default=2)
    parser.add_argument("--min-polygon-overlap", type=float, default=0.50)
    parser.add_argument("--min-text-area-ratio", type=float, default=0.001)
    parser.add_argument("--crop-padding-ratio", type=float, default=0.05)
    parser.add_argument("--audit-jsonl", type=Path, default=None)
    parser.add_argument("--qa-dir", type=Path, default=None)
    parser.add_argument("--qa-limit-per-decision", type=int, default=100)
    parser.add_argument("--max-frames", type=int, default=0, help="0 processes all frames")
    parser.add_argument("--overwrite-audit", action="store_true")
    return parser.parse_args()


def source_raw_id(categories_path, name):
    categories = json.loads(categories_path.read_text(encoding="utf-8"))
    for item in categories:
        if item["name"].strip().lower() == name.lower():
            return int(item["id"]) + 1
    raise ValueError(f"Source category not found: {name}")


def find_image(image_dir, stem):
    matches = [image_dir / f"{stem}{suffix}" for suffix in IMAGE_EXTENSIONS]
    matches = [path for path in matches if path.exists()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one image for {stem} under {image_dir}, got {matches}")
    return matches[0]


def effective_character_count(text):
    return sum(character.isalnum() for character in str(text))


def padded_bbox(instance_mask, padding_ratio):
    rows, columns = np.nonzero(instance_mask)
    if len(rows) == 0:
        return None
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(columns.min()), int(columns.max()) + 1
    padding = int(round(max(bottom - top, right - left) * padding_ratio))
    return (
        max(0, left - padding),
        max(0, top - padding),
        min(instance_mask.shape[1], right + padding),
        min(instance_mask.shape[0], bottom + padding),
    )


def evaluate_detections(crop_rgb, carrier_mask, reader, args, cv2):
    raw_results = reader.readtext(crop_rgb, detail=1, paragraph=False)
    carrier_pixels = max(1, int(carrier_mask.sum()))
    accepted_area = 0
    evidence = []
    for result in raw_results:
        if len(result) < 3:
            continue
        box, text, confidence = result[0], str(result[1]), float(result[2])
        polygon = np.asarray(box, dtype=np.float32).round().astype(np.int32)
        polygon[:, 0] = np.clip(polygon[:, 0], 0, crop_rgb.shape[1] - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, crop_rgb.shape[0] - 1)
        polygon_mask = np.zeros(carrier_mask.shape, dtype=np.uint8)
        cv2.fillPoly(polygon_mask, [polygon], 1)
        polygon_pixels = max(1, int(polygon_mask.sum()))
        overlap_pixels = int((polygon_mask.astype(bool) & carrier_mask).sum())
        overlap = overlap_pixels / polygon_pixels
        characters = effective_character_count(text)
        passes = confidence >= args.min_confidence and characters >= args.min_characters and overlap >= args.min_polygon_overlap
        if passes:
            accepted_area += overlap_pixels
        evidence.append({
            "text": text,
            "confidence": confidence,
            "effective_characters": characters,
            "polygon": polygon.tolist(),
            "carrier_overlap": overlap,
            "overlap_pixels": overlap_pixels,
            "passes_detection_gate": passes,
        })
    area_ratio = accepted_area / carrier_pixels
    accepted = accepted_area > 0 and area_ratio >= args.min_text_area_ratio
    return accepted, area_ratio, evidence


def load_completed_records(audit_path):
    completed = {}
    if not audit_path.exists():
        return completed
    for line in audit_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        key = (item["split"], item["video"], item["frame"], int(item["instance_value"]))
        completed[key] = item
    return completed


def save_qa(crop_rgb, accepted, key, qa_dir, counters, limit):
    decision = "accepted" if accepted else "rejected"
    if qa_dir is None or counters[decision] >= limit:
        return
    target = qa_dir / decision
    target.mkdir(parents=True, exist_ok=True)
    split, video, frame, instance_value = key
    Image.fromarray(crop_rgb, mode="RGB").save(target / f"{split}_{video}_{Path(frame).stem}_{instance_value}.jpg")
    counters[decision] += 1


def main():
    args = parse_args()
    import cv2
    try:
        import easyocr
    except ImportError as error:
        raise RuntimeError("Install EasyOCR first: python -m pip install easyocr") from error

    categories_path = args.metadata_root / "panoVIPSeg_categories.json"
    poster_raw_id = source_raw_id(categories_path, "painting_or_poster")
    languages = [item.strip() for item in args.languages.split(",") if item.strip()]
    reader = easyocr.Reader(languages, gpu=args.gpu)
    audit_path = args.audit_jsonl or args.converted_root / "painting_or_poster_ocr_audit.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite_audit and audit_path.exists():
        audit_path.unlink()
    completed = load_completed_records(audit_path)
    qa_dir = args.qa_dir or args.converted_root / "qa_poster_ocr"
    qa_counters = {"accepted": 0, "rejected": 0}
    total_frames = 0
    total_instances = len(completed)
    accepted_instances = sum(bool(item.get("accepted_as_text", False)) for item in completed.values())

    with audit_path.open("a", encoding="utf-8") as audit_file:
        for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
            video_ids = [line.strip() for line in (args.metadata_root / f"{split}.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
            mask_paths = []
            for video in video_ids:
                mask_paths.extend((args.vipseg_root / "panomasks" / video).glob("*.png"))
            for raw_mask_path in tqdm(sorted(mask_paths), desc=f"OCR posters {split}", dynamic_ncols=True):
                if args.max_frames and total_frames >= args.max_frames:
                    break
                total_frames += 1
                video = raw_mask_path.parent.name
                with Image.open(raw_mask_path) as raw_file:
                    raw_mask = np.asarray(raw_file)
                instance_values = [
                    int(value) for value in np.unique(raw_mask)
                    if int(value) > 124 and int(value) // 100 == poster_raw_id
                ]
                if not instance_values:
                    continue
                image_path = find_image(args.vipseg_root / "images" / video, raw_mask_path.stem)
                converted_path = args.converted_root / "annotations" / split / video / f"{raw_mask_path.stem}.png"
                if not converted_path.exists():
                    raise FileNotFoundError(f"Run strict VIPSeg conversion first: {converted_path}")
                with Image.open(image_path) as image_file:
                    image_rgb = np.asarray(image_file.convert("RGB"))
                with Image.open(converted_path) as converted_file:
                    converted = np.array(converted_file, dtype=np.uint8, copy=True)
                frame_changed = False
                for instance_value in instance_values:
                    key = (split, video, raw_mask_path.name, instance_value)
                    instance_mask = raw_mask == instance_value
                    # Always reconstruct this source class from audited decisions,
                    # so reruns never retain stale broad-mapping labels.
                    if np.any(converted[instance_mask] == 10):
                        frame_changed = True
                    converted[instance_mask] = 0
                    if key in completed:
                        if completed[key].get("accepted_as_text", False):
                            converted[instance_mask] = 10
                            frame_changed = True
                        continue
                    total_instances += 1
                    bbox = padded_bbox(instance_mask, args.crop_padding_ratio)
                    left, top, right, bottom = bbox
                    crop_rgb = image_rgb[top:bottom, left:right]
                    crop_carrier = instance_mask[top:bottom, left:right]
                    accepted, area_ratio, evidence = evaluate_detections(crop_rgb, crop_carrier, reader, args, cv2)
                    if accepted:
                        converted[instance_mask] = 10
                        frame_changed = True
                        accepted_instances += 1
                    record = {
                        "split": split,
                        "video": video,
                        "frame": raw_mask_path.name,
                        "instance_value": instance_value,
                        "bbox_xyxy": list(bbox),
                        "accepted_as_text": accepted,
                        "accepted_text_area_ratio": area_ratio,
                        "thresholds": {
                            "min_confidence": args.min_confidence,
                            "min_characters": args.min_characters,
                            "min_polygon_overlap": args.min_polygon_overlap,
                            "min_text_area_ratio": args.min_text_area_ratio,
                        },
                        "detections": evidence,
                    }
                    audit_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    audit_file.flush()
                    save_qa(crop_rgb, accepted, key, qa_dir, qa_counters, args.qa_limit_per_decision)
                if frame_changed:
                    Image.fromarray(converted, mode="L").save(converted_path)
    summary = {
        "frames_scanned": total_frames,
        "poster_instances_evaluated": total_instances,
        "poster_instances_accepted": accepted_instances,
        "acceptance_rate": accepted_instances / max(1, total_instances),
        "audit_jsonl": str(audit_path),
        "qa_dir": str(qa_dir),
    }
    (args.converted_root / "painting_or_poster_ocr_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
