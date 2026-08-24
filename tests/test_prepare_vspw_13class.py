import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "prepare_vspw_13class.py"
CHECKER = REPO / "tools" / "check_video_dataset.py"


def write_source_frame(root, video, stem, raw_mask):
    image_dir = root / "data" / video / "origin"
    mask_dir = root / "data" / video / "mask"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    image = np.full((*raw_mask.shape, 3), 127, dtype=np.uint8)
    Image.fromarray(image).save(image_dir / f"{stem}.jpg")
    Image.fromarray(raw_mask.astype(np.uint8)).save(mask_dir / f"{stem}.png")


def test_filters_no_target_frames_and_splits_temporal_runs(tmp_path):
    source, output = tmp_path / "source", tmp_path / "VSPW_13cls"
    (source / "train.txt").parent.mkdir(parents=True)
    (source / "train.txt").write_text("video_a\n", encoding="utf-8")
    # raw 1 -> sky(target 1), raw 2 -> road(non-target/background), raw 3 -> person(target 2)
    write_source_frame(source, "video_a", "0000", np.array([[1, 253], [2, 2]]))
    write_source_frame(source, "video_a", "0001", np.array([[0, 253], [255, 2]]))
    write_source_frame(source, "video_a", "0002", np.array([[3, 2], [2, 2]]))
    write_source_frame(source, "video_a", "0003", np.array([[3, 253], [2, 2]]))

    categories = [
        {"id": 0, "name": "sky"}, {"id": 1, "name": "road"},
        {"id": 2, "name": "person"},
    ]
    categories_json = tmp_path / "categories.json"
    categories_json.write_text(json.dumps(categories), encoding="utf-8")
    target_classes = [
        {"id": index, "name": name, "source_names": sources}
        for index, (name, sources) in enumerate([
            ("background", []), ("sky", ["sky"]), ("person", ["person"]),
            ("plant", []), ("building", []), ("flower", []), ("food", []),
            ("water", []), ("desert", []), ("ice_or_snow", []), ("text", []),
            ("ball", []), ("mountain", []),
        ])
    ]
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({"target_classes": target_classes}), encoding="utf-8")

    process = subprocess.run([
        sys.executable, str(TOOL), "--vspw-root", str(source),
        "--categories-json", str(categories_json), "--mapping", str(mapping),
        "--output-root", str(output), "--splits", "train", "--workers", "1",
        "--copy-mode", "copy",
    ], text=True, capture_output=True)
    assert process.returncode == 0, process.stderr

    kept_images = sorted(path.relative_to(output) for path in (output / "images").rglob("*.jpg"))
    assert kept_images == [
        Path("images/train/video_a/segment_0000/0000.jpg"),
        Path("images/train/video_a/segment_0001/0002.jpg"),
        Path("images/train/video_a/segment_0001/0003.jpg"),
    ]
    assert not list((output / "images").rglob("0001.jpg"))
    first_mask = np.asarray(Image.open(output / "annotations/train/video_a/segment_0000/0000.png"))
    second_mask = np.asarray(Image.open(output / "annotations/train/video_a/segment_0001/0002.png"))
    third_mask = np.asarray(Image.open(output / "annotations/train/video_a/segment_0001/0003.png"))
    np.testing.assert_array_equal(first_mask, np.array([[1, 255], [0, 0]], dtype=np.uint8))
    np.testing.assert_array_equal(second_mask, np.array([[2, 0], [0, 0]], dtype=np.uint8))
    np.testing.assert_array_equal(third_mask, np.array([[2, 255], [0, 0]], dtype=np.uint8))

    summary = json.loads((output / "dataset_summary.json").read_text())
    assert summary["source_ignore_aliases"] == {"253": 255}
    assert summary["splits"]["train"]["source_frames"] == 4
    assert summary["splits"]["train"]["kept_frames"] == 3
    assert summary["splits"]["train"]["dropped_no_target_frames"] == 1
    assert summary["splits"]["train"]["source_ignore_alias_253_frames"] == 3
    assert summary["splits"]["train"]["source_ignore_alias_253_pixels"] == 3
    assert summary["splits"]["train"]["segments"] == 2
    assert (output / "_SUCCESS").read_text().strip() == "complete"
    decisions = (output / "metadata/frame_filter_train.tsv").read_text()
    assert "0001.jpg\tdrop\tno_target_1_12" in decisions
    check = subprocess.run([
        sys.executable, str(CHECKER), "--data-root", str(output),
        "--splits", "train", "--require-target",
    ], text=True, capture_output=True)
    assert check.returncode == 0, check.stderr


def test_rejects_mismatched_official_image_and_mask_sizes(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    (source / "train.txt").parent.mkdir(parents=True)
    (source / "train.txt").write_text("video_a\n", encoding="utf-8")
    write_source_frame(source, "video_a", "0000", np.ones((2, 2), dtype=np.uint8))
    Image.fromarray(np.ones((3, 2), dtype=np.uint8)).save(source / "data/video_a/mask/0000.png")
    categories = tmp_path / "categories.json"
    categories.write_text(json.dumps([{"id": 0, "name": "sky"}]))
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({"target_classes": [
        {"id": index, "name": str(index), "source_names": ["sky"] if index == 1 else []}
        for index in range(13)
    ]}))
    process = subprocess.run([
        sys.executable, str(TOOL), "--vspw-root", str(source),
        "--categories-json", str(categories), "--mapping", str(mapping),
        "--output-root", str(output), "--splits", "train", "--workers", "1",
    ], text=True, capture_output=True)
    assert process.returncode != 0
    assert "size mismatch" in process.stderr
