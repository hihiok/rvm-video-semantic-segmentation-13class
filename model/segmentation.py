"""RVM adapted to recurrent multi-class video semantic segmentation."""

from typing import Dict, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .decoder import Projection, RecurrentDecoder
from .lraspp import LRASPP
from .mobilenetv3 import MobileNetV3LargeEncoder
from .resnet import ResNet50Encoder


class RVMForVideoSemanticSegmentation(nn.Module):
    """
    RVM encoder + LR-ASPP + ConvGRU decoder + multi-class projection.

    ``src`` can be an image tensor ``[B,C,H,W]`` or a video clip
    ``[B,T,C,H,W]``. For clips, recurrent state is propagated in chronological
    order by every ConvGRU and gradients flow through time.
    """

    def __init__(
        self,
        variant: str = "mobilenetv3",
        num_classes: int = 13,
        pretrained_backbone: bool = False,
    ):
        super().__init__()
        if variant not in ("mobilenetv3", "resnet50"):
            raise ValueError(f"Unsupported variant: {variant}")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        self.variant = variant
        self.num_classes = num_classes
        if variant == "mobilenetv3":
            self.backbone = MobileNetV3LargeEncoder(pretrained_backbone)
            self.aspp = LRASPP(960, 128)
            self.decoder = RecurrentDecoder([16, 24, 40, 128], [80, 40, 32, 16])
        else:
            self.backbone = ResNet50Encoder(pretrained_backbone)
            self.aspp = LRASPP(2048, 256)
            self.decoder = RecurrentDecoder([64, 256, 512, 256], [128, 64, 32, 16])

        self.project_seg = Projection(16, num_classes)

    def forward(
        self,
        src: Tensor,
        r1: Optional[Tensor] = None,
        r2: Optional[Tensor] = None,
        r3: Optional[Tensor] = None,
        r4: Optional[Tensor] = None,
        downsample_ratio: float = 1.0,
    ):
        if src.ndim not in (4, 5):
            raise ValueError(f"Expected [B,C,H,W] or [B,T,C,H,W], got {src.shape}")
        if not 0 < downsample_ratio <= 1:
            raise ValueError("downsample_ratio must be in (0, 1].")

        src_sm = (
            self._interpolate(src, scale_factor=downsample_ratio)
            if downsample_ratio != 1
            else src
        )
        f1, f2, f3, f4 = self.backbone(src_sm)
        f4 = self.aspp(f4)
        hidden, *rec = self.decoder(src_sm, f1, f2, f3, f4, r1, r2, r3, r4)
        logits = self.project_seg(hidden)
        if logits.shape[-2:] != src.shape[-2:]:
            logits = self._interpolate(logits, size=src.shape[-2:])
        return [logits, *rec]

    @staticmethod
    def _interpolate(tensor: Tensor, scale_factor=None, size=None):
        kwargs = dict(
            size=size,
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=False,
        )
        if scale_factor is not None:
            kwargs["recompute_scale_factor"] = False
        if tensor.ndim == 5:
            batch, time = tensor.shape[:2]
            return F.interpolate(tensor.flatten(0, 1), **kwargs).unflatten(0, (batch, time))
        return F.interpolate(tensor, **kwargs)


# Backwards-compatible import name used by the earlier single-frame project.
RVMForSemanticSegmentation = RVMForVideoSemanticSegmentation


def extract_state_dict(checkpoint) -> Dict[str, Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a PyTorch state_dict.")
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in checkpoint.items()
        if isinstance(value, Tensor)
    }


def load_compatible_weights(
    model: nn.Module,
    checkpoint,
    target_class_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """
    Load RVM or semantic weights and expand/reorder the segmentation head.

    If the source checkpoint has ``class_names``, rows in the 1x1 semantic
    projection are copied by class name. This preserves all twelve old classes
    when initializing the new thirteen-class model; ``mountain`` remains at its
    random initialization.
    """
    source = extract_state_dict(checkpoint)
    target = model.state_dict()
    head_keys = {"project_seg.conv.weight", "project_seg.conv.bias"}
    compatible = {
        key: value
        for key, value in source.items()
        if key in target and target[key].shape == value.shape and key not in head_keys
    }
    result = model.load_state_dict(compatible, strict=False)

    copied_classes = []
    source_names = checkpoint.get("class_names") if isinstance(checkpoint, dict) else None
    if source_names and target_class_names:
        source_index = {name: index for index, name in enumerate(source_names)}
        with torch.no_grad():
            for target_index, name in enumerate(target_class_names):
                source_index_value = source_index.get(name)
                if source_index_value is None:
                    continue
                for key in head_keys:
                    if key not in source or key not in target:
                        continue
                    source_tensor = source[key]
                    target_tensor = target[key]
                    if source_index_value >= source_tensor.shape[0] or target_index >= target_tensor.shape[0]:
                        continue
                    target_tensor[target_index].copy_(source_tensor[source_index_value])
                copied_classes.append(name)
    elif all(key in source for key in head_keys):
        # Same-size head without class metadata: only copy if shapes match.
        head_compatible = {
            key: source[key] for key in head_keys if source[key].shape == target[key].shape
        }
        model.load_state_dict(head_compatible, strict=False)

    skipped = sorted(
        key for key, value in source.items()
        if key not in target or (key not in head_keys and target[key].shape != value.shape)
    )
    return {
        "loaded_tensors": len(compatible),
        "copied_head_classes": copied_classes,
        "skipped": skipped,
        "missing": list(result.missing_keys),
        "unexpected": list(result.unexpected_keys),
    }
