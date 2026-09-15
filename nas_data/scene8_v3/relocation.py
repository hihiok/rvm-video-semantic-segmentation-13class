"""Explicit prefix relocation; never basename-search or rewrite source manifests."""
from pathlib import Path
from storage import Blocked

DATASET_NAMES = ('coco', 'places365', 'COCO_ADE_13cls_16x9_640x360', '10_scenes', 'AWB_10_scenes')


def migration_maps(root):
    # Known historical layouts only. Preserve the entire path inside each dataset.
    return [(Path(prefix) / name, Path(root) / name)
            for prefix in ('/data/pub1/z00919662/segmentation/datasets', '/data/pub1/z00919662/dataset')
            for name in DATASET_NAMES]


def normalized_maps(pairs):
    result = {}
    for old, new in pairs:
        old, new = Path(old), Path(new).resolve()
        if not old.is_absolute() or '..' in old.parts:
            raise Blocked('Relocation prefix must be absolute without traversal: ' + str(old))
        if old in result and result[old] != new:
            raise Blocked('Conflicting relocation targets: ' + str(old))
        result[old] = new
    return sorted(result.items(), key=lambda x: len(x[0].parts), reverse=True)


def relocate(image, mappings):
    original = Path(image)
    if not original.is_absolute() or '..' in original.parts:
        raise Blocked('Legacy image must be absolute without traversal: ' + str(image))
    for old, new in mappings:
        if original == old or old in original.parents:
            result = (new / original.relative_to(old)).resolve()
            if result != new and new not in result.parents:
                raise Blocked('Relocated image escapes target root: ' + str(image))
            return result
    return original.resolve()
