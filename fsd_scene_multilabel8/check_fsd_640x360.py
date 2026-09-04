#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def _to_tensor_image(value):
    if isinstance(value, torch.Tensor):
        x = value
    else:
        x = torch.from_numpy(np.asarray(value))
    if x.ndim == 2:
        x = x.unsqueeze(0)
    elif x.ndim == 3 and x.shape[0] not in (1, 3) and x.shape[-1] in (1, 3):
        x = x.permute(2, 0, 1)
    return x.float().contiguous()


def apply_transform(transform, image_bgr):
    try:
        out = transform(image_bgr)
    except Exception:
        out = transform(image_bgr, None, None)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return _to_tensor_image(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fsd-root', type=Path, required=True)
    args = p.parse_args()

    fsd_root = args.fsd_root.resolve()
    expected_model = fsd_root / 'vision' / 'ssd' / 'mb_tiny_RFB_fd_3.py'
    if not expected_model.exists():
        raise FileNotFoundError(f'Not an FSD repo: {fsd_root}; missing {expected_model}')
    sys.path.insert(0, str(fsd_root))

    # Bootstrap the existing FSD module using a supported UltraFace scalar size.
    # The actual scene tensor below is explicitly 360x640.
    from vision.ssd.config.fd_config import define_img_size
    define_img_size(640)
    from vision.ssd.data_preprocessing import YUV444TestTransform_scene
    from vision.ssd.mb_tiny_RFB_fd_3 import create_Mb_Tiny_RFB_fd_3_scene_noRFB

    transform = YUV444TestTransform_scene([360, 640])
    dummy_bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
    x = apply_transform(transform, dummy_bgr)
    print('TRANSFORMED_SHAPE=', tuple(x.shape))
    if tuple(x.shape) != (1, 360, 640):
        raise RuntimeError(f'Expected transformed scene tensor (1,360,640), got {tuple(x.shape)}')

    model = create_Mb_Tiny_RFB_fd_3_scene_noRFB(2, 8).eval()
    with torch.no_grad():
        y = model(x.unsqueeze(0))
    print('MODEL_OUTPUT_SHAPE=', tuple(y.shape))
    if tuple(y.shape) != (1, 8):
        raise RuntimeError(f'Expected FSD scene output (1,8), got {tuple(y.shape)}')

    params = sum(p.numel() for p in model.parameters())
    print('MODEL_PARAMS=', params)
    print('FSD_640X360_FACTORY_TEST=PASS')


if __name__ == '__main__':
    main()
