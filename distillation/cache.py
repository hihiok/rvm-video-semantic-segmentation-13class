"""Compact, deterministic cache helpers for 13-class teacher probabilities."""

import os
import tempfile
from pathlib import Path

import numpy as np
import torch


def cache_path_for_image(image_path, image_root, cache_root):
    image_path = Path(image_path).expanduser().resolve()
    image_root = Path(image_root).expanduser().resolve()
    try:
        relative = image_path.relative_to(image_root)
    except ValueError as error:
        raise ValueError(f"Image {image_path} is outside image root {image_root}") from error
    return Path(cache_root).expanduser().resolve() / relative.with_suffix(".npz")


def save_cached_probabilities(path, probabilities):
    """Quantize normalized [13,H,W] probabilities to uint8 and replace atomically."""
    path = Path(path)
    values = probabilities.detach().float().cpu().numpy() if torch.is_tensor(probabilities) else np.asarray(probabilities)
    if values.ndim != 3 or values.shape[0] != 13:
        raise ValueError(f"Expected [13,H,W] teacher probabilities, received {values.shape}")
    values = np.maximum(values, 0)
    values /= np.maximum(values.sum(axis=0, keepdims=True), 1e-8)
    quantized = np.rint(values * 255).clip(0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz", dir=path.parent)
    os.close(handle)
    try:
        with open(temporary, "wb") as stream:
            np.savez_compressed(stream, probabilities=quantized)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_cached_probabilities(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing OneFormer cache entry: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"probabilities"}:
            raise ValueError(f"Unexpected cache keys in {path}: {payload.files}")
        values = payload["probabilities"]
    if values.dtype != np.uint8 or values.ndim != 3 or values.shape[0] != 13:
        raise ValueError(f"Invalid cache tensor in {path}: dtype={values.dtype}, shape={values.shape}")
    return torch.from_numpy(values.copy())
