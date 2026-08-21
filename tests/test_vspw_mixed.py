"""Focused regression tests for static replay and the two-stage curriculum."""

import random

import numpy as np
import pytest
import torch
from PIL import Image

from dataset import StaticSemanticDataset, VideoTrainTransform, resolve_static_split
from train_vspw_mixed import balanced_score, mixed_batch_sources, parse_args, stage_for_epoch


def make_split(root, split, values=(1, 12)):
    image_root = root / "images" / split
    mask_root = root / "annotations" / split
    image_root.mkdir(parents=True)
    mask_root.mkdir(parents=True)
    for index, value in enumerate(values):
        image = np.full((24, 32, 3), 100 + index, dtype=np.uint8)
        mask = np.full((24, 32), value, dtype=np.uint8)
        Image.fromarray(image).save(image_root / f"sample_{index:03d}.jpg")
        Image.fromarray(mask).save(mask_root / f"sample_{index:03d}.png")
    return image_root, mask_root


def test_static_split_discovery(tmp_path):
    expected_images, expected_masks = make_split(tmp_path, "train")
    resolved = resolve_static_split(tmp_path, "train")
    assert resolved.image_root == expected_images.resolve()
    assert resolved.mask_root == expected_masks.resolve()


def test_static_dataset_returns_independent_single_frame_clips(tmp_path):
    images, masks = make_split(tmp_path, "train")
    random.seed(7)
    dataset = StaticSemanticDataset(
        images, masks, VideoTrainTransform(size=32, scale_range=(1, 1)), num_classes=13
    )
    image, mask = dataset[0]
    assert image.shape == (1, 3, 32, 32)
    assert mask.shape == (1, 32, 32)
    assert set(mask.unique().tolist()).issubset({1, 255})


def test_static_dataset_rejects_unknown_labels(tmp_path):
    images, masks = make_split(tmp_path, "train", values=(13,))
    dataset = StaticSemanticDataset(images, masks, VideoTrainTransform(size=32))
    with pytest.raises(ValueError, match="invalid IDs"):
        dataset[0]


def test_source_schedule_keeps_replay_after_partial_final_group():
    assert list(mixed_batch_sources(5, 2, 1)) == [
        "video", "video", "static", "video", "video", "static", "video", "static"
    ]


def test_stage_transition_increases_clip_length_and_video_ratio(tmp_path):
    args = parse_args([
        "--data-root", str(tmp_path / "vspw"),
        "--static-root", str(tmp_path / "static"),
        "--init-checkpoint", str(tmp_path / "initial.pth"),
        "--stage2-epochs", "2", "--stage3-epochs", "3",
    ])
    assert stage_for_epoch(args, 0)["clip_length"] == 5
    assert stage_for_epoch(args, 1)["video_batches"] == 1
    assert stage_for_epoch(args, 2)["clip_length"] == 8
    assert stage_for_epoch(args, 2)["video_batches"] == 2


def test_balanced_score_uses_both_validation_domains():
    assert balanced_score({"miou": 0.8}, {"miou": 0.6}, 0.5) == pytest.approx(0.7)
