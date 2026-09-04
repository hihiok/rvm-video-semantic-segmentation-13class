#!/usr/bin/env python3
"""640x360 entrypoint using the exact FSD scene transform names from the reference trainer.

The uploaded FSD reference code uses `YUVTrainAugmentation_scene` and
`YUVTestTransform_scene` and produces one-channel scene tensors. The shared
trainer in this branch expects compatibility aliases so it can also run on FSD
forks that used YUV444-prefixed names. This entrypoint binds the exact reference
functions without modifying the existing FSD repository, then delegates to the
shared 8-label trainer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_bootstrap_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--fsd-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args, _ = p.parse_known_args()
    return args


def main():
    boot = parse_bootstrap_args()
    fsd_root = boot.fsd_root.resolve()
    if not (fsd_root / 'vision' / 'ssd' / 'data_preprocessing.py').exists():
        raise FileNotFoundError(f'Not an FSD repo: {fsd_root}')
    sys.path.insert(0, str(fsd_root))

    import vision.ssd.data_preprocessing as dp
    if not hasattr(dp, 'YUVTrainAugmentation_scene'):
        raise AttributeError('Reference FSD transform YUVTrainAugmentation_scene is missing')
    if not hasattr(dp, 'YUVTestTransform_scene'):
        raise AttributeError('Reference FSD transform YUVTestTransform_scene is missing')

    # Compatibility aliases consumed by the shared trainer. They point to the
    # exact functions used by the uploaded FSD reference script.
    dp.YUV444TrainAugmentation_scene = dp.YUVTrainAugmentation_scene
    dp.YUV444TestTransform_scene = dp.YUVTestTransform_scene

    import train_fsd_scene_multilabel8 as trainer
    trainer.main()

    meta_path = boot.output_dir / 'deployment_metadata.json'
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        meta['preprocess'] = 'existing FSD YUVTrainAugmentation_scene/YUVTestTransform_scene with explicit size [360,640]'
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
