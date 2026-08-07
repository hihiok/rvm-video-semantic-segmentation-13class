"""Sequence-aware dataset and temporally consistent augmentation."""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTENSIONS = (".png", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class VideoSequence:
    name: str
    image_paths: Tuple[Path, ...]
    mask_paths: Tuple[Path, ...]


def _find_mask(mask_video_dir: Path, image_path: Path) -> Path:
    matches = [mask_video_dir / f"{image_path.stem}{suffix}" for suffix in MASK_EXTENSIONS]
    matches = [path for path in matches if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one mask for {image_path.name} under {mask_video_dir}, got {matches}"
        )
    return matches[0]


def discover_video_sequences(image_root, mask_root) -> List[VideoSequence]:
    image_root = Path(image_root).expanduser().resolve()
    mask_root = Path(mask_root).expanduser().resolve()
    if not image_root.is_dir() or not mask_root.is_dir():
        raise FileNotFoundError(f"Missing image/mask root: {image_root}, {mask_root}")

    groups = {}
    for path in image_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            relative_parent = path.parent.relative_to(image_root)
            groups.setdefault(relative_parent, []).append(path)
    if not groups:
        raise RuntimeError(f"No images found under {image_root}")

    sequences = []
    for relative_parent, image_paths in sorted(groups.items(), key=lambda item: str(item[0])):
        image_paths = tuple(sorted(image_paths))
        mask_video_dir = mask_root / relative_parent
        mask_paths = tuple(_find_mask(mask_video_dir, path) for path in image_paths)
        sequences.append(VideoSequence(str(relative_parent), image_paths, mask_paths))
    return sequences


def letterbox_geometry(width: int, height: int, size: int):
    scale = min(size / width, size / height)
    resized_w = max(1, min(size, int(round(width * scale))))
    resized_h = max(1, min(size, int(round(height * scale))))
    left = (size - resized_w) // 2
    top = (size - resized_h) // 2
    return resized_w, resized_h, (left, top, size - resized_w - left, size - resized_h - top)


class VideoTrainTransform:
    """Use identical geometry and color parameters for every frame in a clip."""

    def __init__(
        self,
        size=512,
        scale_range=(0.5, 2.0),
        hflip_probability=0.5,
        ignore_index=255,
    ):
        self.size = size
        self.scale_range = scale_range
        self.hflip_probability = hflip_probability
        self.ignore_index = ignore_index

    def __call__(self, images: Sequence[Image.Image], masks: Sequence[Image.Image]):
        source_w, source_h = images[0].size
        if any(image.size != (source_w, source_h) for image in images):
            raise ValueError("All frames in a clip must have identical resolution")

        scale = random.uniform(*self.scale_range)
        short_side = max(1, int(min(source_h, source_w) * scale))
        resize_scale = short_side / min(source_h, source_w)
        new_h = max(1, int(round(source_h * resize_scale)))
        new_w = max(1, int(round(source_w * resize_scale)))
        pad_right = max(0, self.size - new_w)
        pad_bottom = max(0, self.size - new_h)

        # RandomCrop only needs the resulting canvas geometry; sample once.
        canvas_h = new_h + pad_bottom
        canvas_w = new_w + pad_right
        top = random.randint(0, canvas_h - self.size)
        left = random.randint(0, canvas_w - self.size)
        do_flip = random.random() < self.hflip_probability

        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        saturation = random.uniform(0.8, 1.2)
        hue = random.uniform(-0.05, 0.05)
        color_ops = [
            lambda image: F.adjust_brightness(image, brightness),
            lambda image: F.adjust_contrast(image, contrast),
            lambda image: F.adjust_saturation(image, saturation),
            lambda image: F.adjust_hue(image, hue),
        ]
        random.shuffle(color_ops)

        output_images, output_masks = [], []
        for image, mask in zip(images, masks):
            image = F.resize(image, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)
            mask = F.resize(mask, [new_h, new_w], interpolation=InterpolationMode.NEAREST)
            if pad_right or pad_bottom:
                image = F.pad(image, [0, 0, pad_right, pad_bottom], fill=0)
                mask = F.pad(mask, [0, 0, pad_right, pad_bottom], fill=self.ignore_index)
            image = F.crop(image, top, left, self.size, self.size)
            mask = F.crop(mask, top, left, self.size, self.size)
            if do_flip:
                image, mask = F.hflip(image), F.hflip(mask)
            for operation in color_ops:
                image = operation(image)
            output_images.append(F.to_tensor(image))
            output_masks.append(torch.from_numpy(np.array(mask, dtype=np.int64, copy=True)))
        return torch.stack(output_images), torch.stack(output_masks)


class VideoValidTransform:
    def __init__(self, size=512, resize_mode="letterbox", ignore_index=255):
        if resize_mode not in ("letterbox", "stretch"):
            raise ValueError("resize_mode must be letterbox or stretch")
        self.size = size
        self.resize_mode = resize_mode
        self.ignore_index = ignore_index

    def __call__(self, images: Sequence[Image.Image], masks: Sequence[Image.Image]):
        output_images, output_masks = [], []
        for image, mask in zip(images, masks):
            if self.resize_mode == "stretch":
                image = F.resize(image, [self.size, self.size], interpolation=InterpolationMode.BILINEAR)
                mask = F.resize(mask, [self.size, self.size], interpolation=InterpolationMode.NEAREST)
            else:
                resized_w, resized_h, padding = letterbox_geometry(image.width, image.height, self.size)
                image = F.resize(image, [resized_h, resized_w], interpolation=InterpolationMode.BILINEAR)
                mask = F.resize(mask, [resized_h, resized_w], interpolation=InterpolationMode.NEAREST)
                image = F.pad(image, list(padding), fill=0)
                mask = F.pad(mask, list(padding), fill=self.ignore_index)
            output_images.append(F.to_tensor(image))
            output_masks.append(torch.from_numpy(np.array(mask, dtype=np.int64, copy=True)))
        return torch.stack(output_images), torch.stack(output_masks)


class VideoClipDataset(Dataset):
    """
    Return clips as ``images[T,C,H,W], masks[T,H,W]``.

    Windows never cross video boundaries. Short tails are padded with the last
    frame while the padded masks are set entirely to ``ignore_index`` so they do
    not affect loss or metrics.
    """

    def __init__(
        self,
        image_root,
        mask_root,
        num_classes=13,
        clip_length=5,
        frame_stride=1,
        clip_step=None,
        transform=None,
        ignore_index=255,
        temporal_reverse_probability=0.0,
        minimum_valid_frames=1,
    ):
        if clip_length < 1 or frame_stride < 1:
            raise ValueError("clip_length and frame_stride must be positive")
        self.sequences = discover_video_sequences(image_root, mask_root)
        self.num_classes = num_classes
        self.clip_length = clip_length
        self.frame_stride = frame_stride
        self.clip_step = clip_step or clip_length * frame_stride
        self.transform = transform
        self.ignore_index = ignore_index
        self.temporal_reverse_probability = temporal_reverse_probability
        self.windows = []
        for sequence_index, sequence in enumerate(self.sequences):
            for start in range(0, len(sequence.image_paths), self.clip_step):
                valid_count = sum(
                    start + offset * frame_stride < len(sequence.image_paths)
                    for offset in range(clip_length)
                )
                if valid_count >= minimum_valid_frames:
                    self.windows.append((sequence_index, start))
        if not self.windows:
            raise RuntimeError("No valid video clips were found")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        sequence_index, start = self.windows[index]
        sequence = self.sequences[sequence_index]
        indices, valid = [], []
        for offset in range(self.clip_length):
            frame_index = start + offset * self.frame_stride
            is_valid = frame_index < len(sequence.image_paths)
            indices.append(min(frame_index, len(sequence.image_paths) - 1))
            valid.append(is_valid)

        images, masks = [], []
        for frame_index in indices:
            with Image.open(sequence.image_paths[frame_index]) as image_file:
                image = image_file.convert("RGB")
            with Image.open(sequence.mask_paths[frame_index]) as mask_file:
                if mask_file.mode not in ("L", "P", "I", "I;16"):
                    raise ValueError(f"Mask must be one channel: {sequence.mask_paths[frame_index]}")
                mask = mask_file.copy()
            if image.size != mask.size:
                raise ValueError(f"Image/mask size mismatch in {sequence.name}")
            self._validate_mask(np.asarray(mask), sequence.mask_paths[frame_index])
            images.append(image)
            masks.append(mask)

        if all(valid) and random.random() < self.temporal_reverse_probability:
            images.reverse()
            masks.reverse()

        if self.transform:
            images, masks = self.transform(images, masks)
        else:
            images = torch.stack([F.to_tensor(image) for image in images])
            masks = torch.stack([
                torch.from_numpy(np.array(mask, dtype=np.int64, copy=True)) for mask in masks
            ])
        for time_index, is_valid in enumerate(valid):
            if not is_valid:
                masks[time_index].fill_(self.ignore_index)
        self._validate_mask(masks.numpy(), sequence.name)
        return images, masks

    def _validate_mask(self, mask, source):
        mask = np.asarray(mask)
        valid = mask != self.ignore_index
        if valid.any() and (mask[valid].min() < 0 or mask[valid].max() >= self.num_classes):
            values = np.unique(mask[valid]).tolist()
            raise ValueError(
                f"Mask {source} contains {values}; expected 0..{self.num_classes - 1} "
                f"or ignore_index={self.ignore_index}"
            )
