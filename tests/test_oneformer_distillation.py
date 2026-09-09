import hashlib
import json
import random

import numpy as np
import pytest
import torch
from PIL import Image

from dataset import DistillationPreparedStaticTransform, DistillationVideoTrainTransform
from distillation.cache import load_cached_probabilities, save_cached_probabilities
from distillation.losses import confidence_gated_kd_loss
from distillation.oneformer_teacher import load_mapping
from semantic_utils import DEFAULT_CLASS_NAMES
from train_vspw_mixed import parse_args, stage_for_epoch, verify_teacher_cache_manifest
from tools.select_distillation_checkpoint import select_checkpoint


def test_mapping_is_fixed_complete_and_masks_unsupported_class():
    mapping = load_mapping("configs/oneformer_ade20k_to_13class.json")
    assert mapping["target_class_names"] == DEFAULT_CLASS_NAMES
    assert len(mapping["source_to_target_ids"]) == 150
    assert mapping["source_to_target_ids"][16] == 12  # mountain
    assert mapping["source_to_target_ids"][68] == 12  # hill
    assert mapping["source_to_target_ids"][43] == 10  # signboard
    assert mapping["source_to_target_ids"][123] == 0  # trade name stays strict/background
    assert mapping["unsupported_target_ids"] == [9]


def test_quantized_cache_roundtrip_is_normalizable(tmp_path):
    probability = torch.rand(13, 9, 16)
    probability /= probability.sum(0, keepdim=True)
    path = tmp_path / "clip" / "frame.npz"
    save_cached_probabilities(path, probability)
    restored = load_cached_probabilities(path)
    assert restored.dtype == torch.uint8
    assert restored.shape == probability.shape
    normalized = restored.float() / restored.float().sum(0, keepdim=True).clamp_min(1)
    assert torch.allclose(normalized.sum(0), torch.ones(9, 16), atol=1e-6)


def test_kd_loss_gates_low_confidence_and_ice_pixels():
    logits = torch.zeros(1, 13, 2, 2, requires_grad=True)
    teacher = torch.zeros_like(logits)
    teacher[:, 1, 0, 0] = 1.0  # usable, agrees with GT
    teacher[:, :, 0, 1] = 1 / 13  # low confidence
    teacher[:, 2, 1, 0] = 1.0  # ice GT must be excluded
    teacher[:, 2, 1, 1] = 1.0  # usable disagreement, downweighted
    target = torch.tensor([[[1, 1], [9, 1]]])
    result = confidence_gated_kd_loss(
        logits, teacher, target, confidence_threshold=0.6,
        disagreement_weight=0.25, unsupported_target_ids=(9,),
    )
    assert result["pixels"] == 2
    assert result["agreement"] == pytest.approx(0.5)
    assert result["loss"].item() > 0
    result["loss"].backward()
    assert torch.isfinite(logits.grad).all()


def test_static_probability_flip_stays_aligned():
    image = Image.fromarray(np.zeros((4, 8, 3), dtype=np.uint8))
    mask_array = np.zeros((4, 8), dtype=np.uint8)
    mask_array[:, 0] = 1
    mask = Image.fromarray(mask_array)
    probability = torch.zeros(13, 2, 4, dtype=torch.uint8)
    probability[1, :, 0] = 255
    transform = DistillationPreparedStaticTransform((4, 8), hflip_probability=1.0)
    _, transformed_mask, transformed_probability = transform([image], [mask], [probability])
    assert transformed_mask[0, :, -1].eq(1).all()
    assert transformed_probability[0, 1, :, -1].gt(0).all()


def test_video_transform_returns_matching_teacher_shape():
    random.seed(4)
    image = Image.fromarray(np.zeros((18, 32, 3), dtype=np.uint8))
    mask = Image.fromarray(np.ones((18, 32), dtype=np.uint8))
    probability = torch.zeros(13, 9, 16, dtype=torch.uint8)
    probability[1].fill_(255)
    output = DistillationVideoTrainTransform((18, 32), scale_range=(1, 1))(
        [image, image], [mask, mask], [probability, probability]
    )
    assert output[0].shape == (2, 3, 18, 32)
    assert output[1].shape == (2, 18, 32)
    assert output[2].shape == (2, 13, 18, 32)


def test_distillation_arguments_are_stage_specific(tmp_path):
    args = parse_args([
        "--data-root", str(tmp_path / "vspw"), "--static-root", str(tmp_path / "static"),
        "--init-checkpoint", str(tmp_path / "model.pth"),
        "--teacher-cache-root", str(tmp_path / "cache"),
        "--stage2-kd-weight", "0.5", "--stage3-kd-weight", "0.2",
    ])
    assert stage_for_epoch(args, 0)["kd_weight"] == pytest.approx(0.5)
    assert stage_for_epoch(args, args.stage2_epochs)["kd_weight"] == pytest.approx(0.2)
    assert args.kd_unsupported_target_ids == (9,)


def test_checkpoint_selector_prefers_valid_spatial_checkpoint(tmp_path):
    from model import RVMForVideoSemanticSegmentation
    state = RVMForVideoSemanticSegmentation("mobilenetv3", 13).state_dict()
    torch.save({"model": state, "class_names": DEFAULT_CLASS_NAMES, "epoch": 12}, tmp_path / "best_miou.pth")
    torch.save(
        {"model": state, "class_names": DEFAULT_CLASS_NAMES, "epoch": 11},
        tmp_path / "best_spatial_preserved.pth",
    )
    report, errors = select_checkpoint(tmp_path)
    assert report["path"].endswith("best_spatial_preserved.pth")
    assert report["epoch"] == 11
    assert not errors


def test_teacher_cache_manifest_is_bound_to_mapping_and_image_root(tmp_path):
    image_root = tmp_path / "images"
    cache_root = tmp_path / "cache"
    image_root.mkdir()
    cache_root.mkdir()
    mapping = tmp_path / "mapping.json"
    mapping.write_text("{}")
    manifest = {
        "format": "oneformer_rvm13_uint8_probability_cache_v1",
        "teacher_model": "shi-labs/oneformer_ade20k_swin_large",
        "mapping_sha256": hashlib.sha256(mapping.read_bytes()).hexdigest(),
        "class_names": DEFAULT_CLASS_NAMES,
        "image_root": str(image_root.resolve()),
        "cache_shape": [13, 45, 80],
        "status": "complete",
    }
    (cache_root / "MANIFEST.json").write_text(json.dumps(manifest))
    assert verify_teacher_cache_manifest(cache_root, image_root, mapping, DEFAULT_CLASS_NAMES)
    manifest["mapping_sha256"] = "wrong"
    (cache_root / "MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="mapping_sha256"):
        verify_teacher_cache_manifest(cache_root, image_root, mapping, DEFAULT_CLASS_NAMES)
