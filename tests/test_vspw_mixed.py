"""Focused regression tests for static replay and the two-stage curriculum."""

import random

import numpy as np
import pytest
import torch
from PIL import Image

from dataset import (
    PreparedStaticTransform,
    StaticSemanticDataset,
    VideoTrainTransform,
    VideoValidTransform,
    letterbox_geometry,
    resolve_static_split,
    split_video_sequences_on_gaps,
)
from dataset.video_semantic import VideoSequence
from train_vspw_mixed import balanced_score, mixed_batch_sources, parse_args, stage_for_epoch
from train_vspw_mixed import configure_trainable_scope
from model import MultiClassFastGuidedFilterRefiner, RVMForVideoSemanticSegmentation
from semantic_utils import causal_temporal_consistency_loss
from inference_video_semantic import scene_cut_score


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


def test_rectangular_video_transforms_preserve_exact_16x9_shape():
    image = Image.fromarray(np.full((72, 128, 3), 100, dtype=np.uint8))
    mask = Image.fromarray(np.full((72, 128), 1, dtype=np.uint8))
    random.seed(3)
    train_images, train_masks = VideoTrainTransform(
        size=(360, 640), scale_range=(1, 1)
    )([image, image], [mask, mask])
    valid_images, valid_masks = VideoValidTransform(size=(360, 640))([image], [mask])
    assert train_images.shape == (2, 3, 360, 640)
    assert train_masks.shape == (2, 360, 640)
    assert valid_images.shape == (1, 3, 360, 640)
    assert valid_masks.shape == (1, 360, 640)
    assert letterbox_geometry(1280, 720, (360, 640)) == (640, 360, (0, 0, 0, 0))


def test_prepared_static_transform_skips_geometry_and_checks_size():
    image = Image.fromarray(np.full((360, 640, 3), 100, dtype=np.uint8))
    mask = Image.fromarray(np.full((360, 640), 12, dtype=np.uint8))
    images, masks = PreparedStaticTransform((360, 640))([image], [mask])
    assert images.shape == (1, 3, 360, 640)
    assert masks.shape == (1, 360, 640)
    assert masks.unique().tolist() == [12]
    with pytest.raises(ValueError, match="already be 640x360"):
        PreparedStaticTransform((360, 640))(
            [Image.fromarray(np.zeros((320, 640, 3), dtype=np.uint8))],
            [Image.fromarray(np.zeros((320, 640), dtype=np.uint8))],
        )


def test_filtered_vspw_frame_gaps_split_recurrent_sequences(tmp_path):
    paths = tuple(tmp_path / f"{index:05d}.png" for index in (1, 2, 4, 5))
    sequence = VideoSequence("video", paths, paths)
    segments = split_video_sequences_on_gaps([sequence], 1)
    assert [len(item.image_paths) for item in segments] == [2, 2]
    assert segments[0].name == "video#segment0000"
    assert segments[1].name == "video#segment0001"


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
    assert (args.input_width, args.input_height) == (640, 360)


def test_balanced_score_uses_both_validation_domains():
    assert balanced_score({"miou": 0.8}, {"miou": 0.6}, 0.5) == pytest.approx(0.7)


def test_temporal_consistency_is_zero_for_identical_predictions():
    logits = torch.randn(1, 3, 4, 8, 8)
    logits[:, 1:] = logits[:, :1]
    target = torch.ones(1, 3, 8, 8, dtype=torch.long)
    loss, pixels = causal_temporal_consistency_loss(
        logits, target, boundary_radius=0
    )
    assert pixels == 2 * 8 * 8
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_temporal_consistency_penalizes_flips_only_on_stable_gt():
    logits = torch.full((1, 2, 3, 6, 6), -4.0)
    logits[:, 0, 0] = 4.0
    logits[:, 1, 1] = 4.0
    target = torch.zeros(1, 2, 6, 6, dtype=torch.long)
    loss, pixels = causal_temporal_consistency_loss(
        logits, target, boundary_radius=0
    )
    assert pixels == 36
    assert loss.item() > 1.0


def test_recurrent_scope_freezes_spatial_parameters():
    model = RVMForVideoSemanticSegmentation("mobilenetv3", 13)
    trainable = configure_trainable_scope(model, "recurrent")
    assert trainable
    assert all(".gru." in name for name in trainable)
    assert all(
        parameter.requires_grad == (".gru." in name)
        for name, parameter in model.named_parameters()
    )


def test_multiclass_guided_filter_supports_arbitrary_class_count():
    refiner = MultiClassFastGuidedFilterRefiner(radius=1, eps=1e-4)
    base_rgb = torch.rand(2, 3, 8, 12)
    base_logits = torch.rand(2, 13, 8, 12)
    fine_rgb = torch.rand(2, 3, 16, 24)
    output = refiner(base_rgb, base_logits, fine_rgb)
    assert output.shape == (2, 13, 16, 24)
    assert torch.isfinite(output).all()


def test_temporal_preserve_arguments_are_recorded_per_stage(tmp_path):
    args = parse_args([
        "--data-root", str(tmp_path / "vspw"),
        "--static-root", str(tmp_path / "static"),
        "--init-checkpoint", str(tmp_path / "initial.pth"),
        "--stage2-trainable-scope", "recurrent",
        "--stage3-trainable-scope", "recurrent",
        "--stage2-temporal-weight", "0.05",
        "--stage3-temporal-weight", "0.10",
    ])
    assert stage_for_epoch(args, 0)["trainable_scope"] == "recurrent"
    assert stage_for_epoch(args, 0)["temporal_weight"] == pytest.approx(0.05)
    assert stage_for_epoch(args, args.stage2_epochs)["temporal_weight"] == pytest.approx(0.10)


def test_scene_cut_score_detects_abrupt_full_frame_change():
    black = np.zeros((32, 48, 3), dtype=np.uint8)
    white = np.full((32, 48, 3), 255, dtype=np.uint8)
    first_score, previous = scene_cut_score(None, black)
    second_score, _ = scene_cut_score(previous, white)
    assert first_score is None
    assert second_score == pytest.approx(1.0)
