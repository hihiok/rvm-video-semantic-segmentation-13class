"""RVM adapted to recurrent multi-class video semantic segmentation."""

from typing import Dict, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .decoder import ConvGRU, Projection, RecurrentDecoder
from .lraspp import LRASPP
from .mobilenetv3 import MobileNetV3LargeEncoder
from .resnet import ResNet50Encoder


class TemporalResidualAdapter(nn.Module):
    """Causal, low-resolution ConvGRU that predicts a residual over frozen logits."""

    def __init__(
        self,
        num_classes: int,
        hidden_channels: int = 16,
        scale: float = 0.25,
    ):
        super().__init__()
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")
        if not 0 < scale <= 1:
            raise ValueError("scale must be in (0, 1]")
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self.scale = scale
        self.input_projection = nn.Sequential(
            nn.Conv2d(num_classes + 3, hidden_channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.gru = ConvGRU(hidden_channels)
        self.output_projection = nn.Conv2d(hidden_channels, num_classes, 1)
        # The complete video model is exactly the frozen single-frame baseline
        # at initialization, including every frame after the first one.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    @staticmethod
    def _resize_sequence(tensor: Tensor, size):
        batch, time = tensor.shape[:2]
        return F.interpolate(
            tensor.flatten(0, 1), size=size, mode="bilinear", align_corners=False
        ).unflatten(0, (batch, time))

    def forward(
        self,
        src: Tensor,
        base_logits: Tensor,
        state: Optional[Tensor] = None,
    ):
        squeeze_time = src.ndim == 4
        if squeeze_time:
            src = src.unsqueeze(1)
            base_logits = base_logits.unsqueeze(1)
        if src.ndim != 5 or base_logits.ndim != 5:
            raise ValueError("Temporal adapter expects image or video tensors")
        if src.shape[:2] != base_logits.shape[:2] or src.shape[-2:] != base_logits.shape[-2:]:
            raise ValueError(f"Temporal adapter input mismatch: {src.shape}, {base_logits.shape}")

        height, width = src.shape[-2:]
        small_size = (
            max(1, int(round(height * self.scale))),
            max(1, int(round(width * self.scale))),
        )
        small_src = self._resize_sequence(src, small_size)
        small_probability = self._resize_sequence(base_logits.softmax(dim=2), small_size)
        batch, time = src.shape[:2]
        features = self.input_projection(
            torch.cat([small_src, small_probability], dim=2).flatten(0, 1)
        ).unflatten(0, (batch, time))

        reset_frame = state is None
        hidden, state = self.gru(features, state)
        residual = self.output_projection(hidden.flatten(0, 1)).unflatten(0, (batch, time))
        residual = self._resize_sequence(residual, (height, width))
        if reset_frame:
            # A reset frame is an exact spatial-only result. The updated state
            # is still returned so temporal corrections can start next frame.
            first_frame_gate = torch.ones(
                (1, time, 1, 1, 1), device=residual.device, dtype=residual.dtype
            )
            first_frame_gate[:, 0] = 0
            residual = residual * first_frame_gate
        if squeeze_time:
            residual = residual[:, 0]
        return residual, state


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
        temporal_residual: bool = False,
        temporal_hidden_channels: int = 16,
        temporal_scale: float = 0.25,
    ):
        super().__init__()
        if variant not in ("mobilenetv3", "resnet50"):
            raise ValueError(f"Unsupported variant: {variant}")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        self.variant = variant
        self.num_classes = num_classes
        self.uses_temporal_residual = temporal_residual
        if variant == "mobilenetv3":
            self.backbone = MobileNetV3LargeEncoder(pretrained_backbone)
            self.aspp = LRASPP(960, 128)
            self.decoder = RecurrentDecoder([16, 24, 40, 128], [80, 40, 32, 16])
        else:
            self.backbone = ResNet50Encoder(pretrained_backbone)
            self.aspp = LRASPP(2048, 256)
            self.decoder = RecurrentDecoder([64, 256, 512, 256], [128, 64, 32, 16])

        self.project_seg = Projection(16, num_classes)
        self.temporal_residual_adapter = (
            TemporalResidualAdapter(num_classes, temporal_hidden_channels, temporal_scale)
            if temporal_residual
            else None
        )

    def _forward_core(
        self,
        src: Tensor,
        r1: Optional[Tensor] = None,
        r2: Optional[Tensor] = None,
        r3: Optional[Tensor] = None,
        r4: Optional[Tensor] = None,
    ):
        f1, f2, f3, f4 = self.backbone(src)
        f4 = self.aspp(f4)
        hidden, *rec = self.decoder(src, f1, f2, f3, f4, r1, r2, r3, r4)
        return self.project_seg(hidden), rec

    def forward_spatial(self, src: Tensor, downsample_ratio: float = 1.0):
        """Run each frame independently through the frozen Stage-1 network."""
        if src.ndim not in (4, 5):
            raise ValueError(f"Expected [B,C,H,W] or [B,T,C,H,W], got {src.shape}")
        if not 0 < downsample_ratio <= 1:
            raise ValueError("downsample_ratio must be in (0, 1].")
        original_size = src.shape[-2:]
        src_sm = (
            self._interpolate(src, scale_factor=downsample_ratio)
            if downsample_ratio != 1
            else src
        )
        if src_sm.ndim == 5:
            batch, time = src_sm.shape[:2]
            logits, _ = self._forward_core(src_sm.flatten(0, 1))
            logits = logits.unflatten(0, (batch, time))
        else:
            logits, _ = self._forward_core(src_sm)
        if logits.shape[-2:] != original_size:
            logits = self._interpolate(logits, size=original_size)
        return logits

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

        if self.temporal_residual_adapter is not None:
            base_logits = self.forward_spatial(src, downsample_ratio)
            residual, temporal_state = self.temporal_residual_adapter(src, base_logits, r1)
            return [base_logits + residual, temporal_state, None, None, None]

        src_sm = (
            self._interpolate(src, scale_factor=downsample_ratio)
            if downsample_ratio != 1
            else src
        )
        logits, rec = self._forward_core(src_sm, r1, r2, r3, r4)
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
