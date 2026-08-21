"""Static semantic-segmentation replay for recurrent video fine-tuning."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from .video_semantic import IMAGE_EXTENSIONS, MASK_EXTENSIONS, normalize_spatial_size


@dataclass(frozen=True)
class StaticSplitPaths:
    image_root: Path
    mask_root: Path


class PreparedStaticTransform:
    """Use pre-sized offline images directly; no per-epoch resize/crop/padding."""

    def __init__(self, size, hflip_probability=0.0):
        self.height, self.width = normalize_spatial_size(size)
        self.hflip_probability = hflip_probability

    def __call__(self, images, masks):
        output_images, output_masks = [], []
        do_flip = random.random() < self.hflip_probability
        for image, mask in zip(images, masks):
            if image.size != (self.width, self.height) or mask.size != (self.width, self.height):
                raise ValueError(
                    f"Prepared static image/mask must already be {self.width}x{self.height}; "
                    f"received {image.size} and {mask.size}. Run tools/prepare_static_16x9.py first."
                )
            if do_flip:
                image, mask = F.hflip(image), F.hflip(mask)
            output_images.append(F.to_tensor(image))
            output_masks.append(torch.from_numpy(np.array(mask, dtype=np.int64, copy=True)))
        return torch.stack(output_images), torch.stack(output_masks)


def resolve_static_split(root, split: str, image_override=None, mask_override=None):
    """Resolve common COCO/ADE layouts without guessing between ambiguous roots."""
    root = Path(root).expanduser().resolve()
    if image_override or mask_override:
        if not image_override or not mask_override:
            raise ValueError("Image and annotation overrides must be supplied together")
        images = Path(image_override).expanduser()
        masks = Path(mask_override).expanduser()
        images = images if images.is_absolute() else root / images
        masks = masks if masks.is_absolute() else root / masks
        if not images.is_dir() or not masks.is_dir():
            raise FileNotFoundError(f"Missing static split directories: {images}, {masks}")
        return StaticSplitPaths(images.resolve(), masks.resolve())

    aliases = ("train", "training") if split == "train" else ("val", "valid", "validation")
    candidates = []
    for alias in aliases:
        for image_name in ("images", "image", "imgs", "JPEGImages"):
            for mask_name in ("annotations", "annotation", "masks", "mask", "labels", "label", "SegmentationClass"):
                candidates.extend(
                    [
                        (root / image_name / alias, root / mask_name / alias),
                        (root / alias / image_name, root / alias / mask_name),
                    ]
                )
    for images, masks in candidates:
        if images.is_dir() and masks.is_dir():
            return StaticSplitPaths(images.resolve(), masks.resolve())
    raise FileNotFoundError(
        f"Could not discover static {split} images/masks under {root}. "
        f"Supply --static-{split}-images and --static-{split}-annotations explicitly."
    )


def discover_static_pairs(image_root, mask_root) -> List[Tuple[Path, Path]]:
    image_root = Path(image_root).expanduser().resolve()
    mask_root = Path(mask_root).expanduser().resolve()
    images = sorted(
        path for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise RuntimeError(f"No static images found under {image_root}")

    pairs = []
    missing = []
    for image_path in images:
        relative = image_path.relative_to(image_root)
        candidates = [mask_root / relative.with_suffix(suffix) for suffix in MASK_EXTENSIONS]
        candidates = [path for path in candidates if path.is_file()]
        if len(candidates) != 1:
            missing.append(str(relative))
            continue
        pairs.append((image_path, candidates[0]))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} static images have no unique matching mask under {mask_root}; "
            f"first examples: {missing[:8]}"
        )
    return pairs


class StaticSemanticDataset(Dataset):
    """Return independent one-frame clips so recurrence never crosses photographs."""

    def __init__(
        self,
        image_root,
        mask_root,
        transform=None,
        num_classes: int = 13,
        ignore_index: int = 255,
        max_samples: int = 0,
    ):
        self.pairs = discover_static_pairs(image_root, mask_root)
        if max_samples > 0:
            self.pairs = self.pairs[:max_samples]
        self.transform = transform
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        with Image.open(mask_path) as handle:
            if handle.mode not in ("L", "P", "I", "I;16"):
                raise ValueError(f"Static mask must have one indexed channel: {mask_path}")
            mask = handle.copy()
        if image.size != mask.size:
            raise ValueError(f"Static image/mask sizes differ: {image_path}, {mask_path}")

        values = np.asarray(mask)
        valid = values != self.ignore_index
        if valid.any() and (values[valid].min() < 0 or values[valid].max() >= self.num_classes):
            raise ValueError(
                f"Static mask {mask_path} contains invalid IDs {np.unique(values[valid]).tolist()}; "
                f"expected 0..{self.num_classes - 1} or {self.ignore_index}"
            )
        if self.transform is None:
            raise ValueError("StaticSemanticDataset requires a video-compatible transform")
        return self.transform([image], [mask])
