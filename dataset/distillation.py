"""Datasets applying identical random geometry to GT masks and cached KD maps."""

import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from distillation.cache import cache_path_for_image, load_cached_probabilities
from .static_semantic import discover_static_pairs
from .video_semantic import discover_video_sequences, normalize_spatial_size, split_video_sequences_on_gaps


def _resize_probability(probability, size):
    value = F.resize(
        probability.float(), list(size), interpolation=InterpolationMode.BILINEAR, antialias=True
    )
    return value.round().clamp(0, 255).to(torch.uint8)


class DistillationPreparedStaticTransform:
    def __init__(self, size, hflip_probability=0.0):
        self.height, self.width = normalize_spatial_size(size)
        self.hflip_probability = hflip_probability

    def __call__(self, images, masks, probabilities):
        output_images, output_masks, output_probabilities = [], [], []
        do_flip = random.random() < self.hflip_probability
        for image, mask, probability in zip(images, masks, probabilities):
            if image.size != (self.width, self.height) or mask.size != image.size:
                raise ValueError(f"Prepared static samples must already be {self.width}x{self.height}")
            probability = _resize_probability(probability, (self.height, self.width))
            if do_flip:
                image, mask, probability = F.hflip(image), F.hflip(mask), F.hflip(probability)
            output_images.append(F.to_tensor(image))
            output_masks.append(torch.from_numpy(np.array(mask, dtype=np.int64, copy=True)))
            output_probabilities.append(probability)
        return torch.stack(output_images), torch.stack(output_masks), torch.stack(output_probabilities)


class DistillationVideoTrainTransform:
    """Mirror VideoTrainTransform, using one shared random draw for each clip."""

    def __init__(self, size, scale_range=(0.9, 1.1), hflip_probability=0.5, ignore_index=255):
        self.height, self.width = normalize_spatial_size(size)
        self.scale_range = scale_range
        self.hflip_probability = hflip_probability
        self.ignore_index = ignore_index

    def __call__(self, images, masks, probabilities):
        source_w, source_h = images[0].size
        if any(image.size != (source_w, source_h) for image in images):
            raise ValueError("All frames in a clip must have identical resolution")
        scale = random.uniform(*self.scale_range)
        base_scale = min(self.width / source_w, self.height / source_h)
        new_h = max(1, int(round(source_h * base_scale * scale)))
        new_w = max(1, int(round(source_w * base_scale * scale)))
        pad_right, pad_bottom = max(0, self.width - new_w), max(0, self.height - new_h)
        top = random.randint(0, new_h + pad_bottom - self.height)
        left = random.randint(0, new_w + pad_right - self.width)
        do_flip = random.random() < self.hflip_probability
        brightness, contrast = random.uniform(0.8, 1.2), random.uniform(0.8, 1.2)
        saturation, hue = random.uniform(0.8, 1.2), random.uniform(-0.05, 0.05)
        color_ops = [
            lambda item: F.adjust_brightness(item, brightness),
            lambda item: F.adjust_contrast(item, contrast),
            lambda item: F.adjust_saturation(item, saturation),
            lambda item: F.adjust_hue(item, hue),
        ]
        random.shuffle(color_ops)
        output_images, output_masks, output_probabilities = [], [], []
        for image, mask, probability in zip(images, masks, probabilities):
            image = F.resize(image, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)
            mask = F.resize(mask, [new_h, new_w], interpolation=InterpolationMode.NEAREST)
            probability = _resize_probability(probability, (new_h, new_w))
            if pad_right or pad_bottom:
                image = F.pad(image, [0, 0, pad_right, pad_bottom], fill=0)
                mask = F.pad(mask, [0, 0, pad_right, pad_bottom], fill=self.ignore_index)
                probability = F.pad(probability, [0, 0, pad_right, pad_bottom], fill=0)
            image = F.crop(image, top, left, self.height, self.width)
            mask = F.crop(mask, top, left, self.height, self.width)
            probability = F.crop(probability, top, left, self.height, self.width)
            if do_flip:
                image, mask, probability = F.hflip(image), F.hflip(mask), F.hflip(probability)
            for operation in color_ops:
                image = operation(image)
            output_images.append(F.to_tensor(image))
            output_masks.append(torch.from_numpy(np.array(mask, dtype=np.int64, copy=True)))
            output_probabilities.append(probability)
        return torch.stack(output_images), torch.stack(output_masks), torch.stack(output_probabilities)


class DistillationStaticDataset(Dataset):
    def __init__(self, image_root, mask_root, cache_root, transform, num_classes=13,
                 ignore_index=255, max_samples=0):
        self.image_root, self.cache_root = image_root, cache_root
        self.pairs = discover_static_pairs(image_root, mask_root)
        if max_samples > 0:
            self.pairs = self.pairs[:max_samples]
        self.transform, self.num_classes, self.ignore_index = transform, num_classes, ignore_index

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
            raise ValueError(f"Static mask {mask_path} contains invalid class IDs")
        probability = load_cached_probabilities(
            cache_path_for_image(image_path, self.image_root, self.cache_root)
        )
        return self.transform([image], [mask], [probability])


class DistillationVideoClipDataset(Dataset):
    def __init__(self, image_root, mask_root, cache_root, *, num_classes=13, clip_length=5,
                 frame_stride=1, clip_step=None, transform=None, ignore_index=255,
                 temporal_reverse_probability=0.0, minimum_valid_frames=1, max_frame_gap=0):
        self.image_root, self.cache_root = image_root, cache_root
        self.sequences = split_video_sequences_on_gaps(
            discover_video_sequences(image_root, mask_root), max_frame_gap
        )
        self.num_classes, self.clip_length, self.frame_stride = num_classes, clip_length, frame_stride
        self.clip_step, self.transform = clip_step or clip_length * frame_stride, transform
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
            valid.append(frame_index < len(sequence.image_paths))
            indices.append(min(frame_index, len(sequence.image_paths) - 1))
        images, masks, probabilities = [], [], []
        for frame_index in indices:
            image_path, mask_path = sequence.image_paths[frame_index], sequence.mask_paths[frame_index]
            with Image.open(image_path) as handle:
                image = handle.convert("RGB")
            with Image.open(mask_path) as handle:
                if handle.mode not in ("L", "P", "I", "I;16"):
                    raise ValueError(f"Video mask must have one indexed channel: {mask_path}")
                mask = handle.copy()
            if image.size != mask.size:
                raise ValueError(f"Video image/mask sizes differ: {image_path}, {mask_path}")
            values = np.asarray(mask)
            usable = values != self.ignore_index
            if usable.any() and (values[usable].min() < 0 or values[usable].max() >= self.num_classes):
                raise ValueError(f"Video mask {mask_path} contains invalid class IDs")
            images.append(image)
            masks.append(mask)
            probabilities.append(load_cached_probabilities(
                cache_path_for_image(image_path, self.image_root, self.cache_root)
            ))
        if all(valid) and random.random() < self.temporal_reverse_probability:
            images.reverse()
            masks.reverse()
            probabilities.reverse()
        images, masks, probabilities = self.transform(images, masks, probabilities)
        for time_index, is_valid in enumerate(valid):
            if not is_valid:
                masks[time_index].fill_(self.ignore_index)
                probabilities[time_index].zero_()
        return images, masks, probabilities
