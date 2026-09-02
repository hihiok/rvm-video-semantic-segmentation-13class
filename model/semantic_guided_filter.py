"""Edge-aware multi-class logit upsampling guided by RGB frames."""

from torch import Tensor, nn

from .fast_guided_filter import FastGuidedFilter


class MultiClassFastGuidedFilterRefiner(nn.Module):
    """
    Apply the original RVM fast-guided-filter idea to arbitrary semantic logits.

    Unlike RVM's matting refiner, this module uses a one-channel luminance guide,
    so one guide can refine any number of class-logit channels through broadcast
    guided-filter coefficients. It has no learned parameters and can therefore be
    compared with bilinear upsampling using an existing checkpoint.
    """

    def __init__(self, radius: int = 1, eps: float = 1e-4):
        super().__init__()
        if radius < 1:
            raise ValueError("radius must be at least 1")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.radius = radius
        self.eps = eps
        self.filter = FastGuidedFilter(radius, eps)

    @staticmethod
    def _luminance(rgb: Tensor):
        if rgb.shape[-3] != 3:
            raise ValueError(f"Expected RGB guide with 3 channels, got {rgb.shape}")
        weights = rgb.new_tensor((0.299, 0.587, 0.114))
        shape = [1] * rgb.ndim
        shape[-3] = 3
        return (rgb * weights.view(shape)).sum(dim=-3, keepdim=True)

    def forward_single_frame(
        self,
        base_rgb: Tensor,
        base_logits: Tensor,
        fine_rgb: Tensor,
    ):
        if base_rgb.ndim != 4 or base_logits.ndim != 4 or fine_rgb.ndim != 4:
            raise ValueError("Single-frame guided refinement expects 4D tensors")
        if base_rgb.shape[0] != base_logits.shape[0] or base_rgb.shape[0] != fine_rgb.shape[0]:
            raise ValueError("Guides and logits must have the same batch size")
        if base_rgb.shape[-2:] != base_logits.shape[-2:]:
            raise ValueError("Base RGB and base logits must have identical spatial sizes")
        base_guide = self._luminance(base_rgb)
        fine_guide = self._luminance(fine_rgb)
        return self.filter(base_guide, base_logits, fine_guide)

    def forward(self, base_rgb: Tensor, base_logits: Tensor, fine_rgb: Tensor):
        if base_rgb.ndim == 5:
            if base_logits.ndim != 5 or fine_rgb.ndim != 5:
                raise ValueError("Video guided refinement requires all inputs to be 5D")
            batch, time = base_rgb.shape[:2]
            if base_logits.shape[:2] != (batch, time) or fine_rgb.shape[:2] != (batch, time):
                raise ValueError("Video guides and logits must share batch/time dimensions")
            output = self.forward_single_frame(
                base_rgb.flatten(0, 1),
                base_logits.flatten(0, 1),
                fine_rgb.flatten(0, 1),
            )
            return output.unflatten(0, (batch, time))
        return self.forward_single_frame(base_rgb, base_logits, fine_rgb)
