"""Five physical legacy roots, label provenance and path-boundary regressions."""
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from prepare_all import BASE, DEFAULT_SOURCE_ROOTS
from sources import legacy_rows
from storage import Blocked
from test_pipeline import make_image

HERE = Path(__file__).parent


def manifests(root, records):
    root.mkdir(exist_ok=True)
    for split in ('train', 'val', 'test'):
        (root / (split + '.jsonl')).write_text(
            ''.join(json.dumps(r) + '\n' for r in records) if split == 'train' else '')


class LegacyRoots(unittest.TestCase):
    def test_defaults_cover_five_specific_datasets(self):
        self.assertEqual(len(DEFAULT_SOURCE_ROOTS), 5)
        self.assertIn(BASE / 'dataset/AWB_10_scenes', DEFAULT_SOURCE_ROOTS)
        self.assertIn(BASE / 'dataset/10_scenes', DEFAULT_SOURCE_ROOTS)
        self.assertNotIn(BASE / 'dataset', DEFAULT_SOURCE_ROOTS)

    def test_five_roots_keep_night_and_synthetic_provenance(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            roots = [root / p.name for p in DEFAULT_SOURCE_ROOTS]
            specs = [('coco2017', 'a.jpg', ''), ('places365', 'office/b.jpg', 'office'),
                     ('seg13', 'c.jpg', ''), ('10_scenes', 'Computer_synthesized/d.jpg', 'Computer_synthesized->objective_image'),
                     ('10_scenes', 'Night/e.jpg', 'Night->night')]
            records = []
            for i, (source, name, detail) in enumerate(specs):
                image = roots[i] / name
                make_image(image, i)
                records.append(dict(image=str(image), source=source, detail=detail,
                                    labels={'night': 0, 'rain_snow': 0}))
            manifests(root / 'old', records)
            audit = {}
            rows = legacy_rows(root / 'old', HERE / 'places365_io.txt', roots, audit, [])
            self.assertEqual(len(rows), 5)
            self.assertEqual(audit['legacy_root_counts'], {str(p): 1 for p in roots})
            self.assertEqual(audit['legacy_source_counts']['10_scenes'], 2)
            self.assertEqual(rows[4]['source_dataset_root'], str(roots[4]))
            self.assertEqual(rows[4]['labels']['night'], 1)
            self.assertTrue(rows[4]['evidence']['night'].startswith('weak:'))
            for label in ('rain_snow', 'indoor', 'outdoor', 'sports'):
                self.assertEqual(rows[4]['labels'][label], -1)
            self.assertEqual(rows[3]['labels']['objective_image'], 1)
            self.assertEqual(rows[3]['labels']['night'], 0)

    def test_sibling_prefix_and_symlink_escape_still_blocked(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            allowed = root / 'AWB_10_scenes'
            allowed.mkdir()
            external = root / 'AWB_10_scenes_backup/Night/a.jpg'
            make_image(external, 1)
            link = allowed / 'Night'
            link.symlink_to(external.parent, target_is_directory=True)
            for image in (external, link / 'a.jpg'):
                with self.subTest(image=image):
                    manifests(root / 'old', [dict(image=str(image), source='10_scenes')])
                    with self.assertRaisesRegex(Blocked, 'outside allowed source roots'):
                        legacy_rows(root / 'old', HERE / 'places365_io.txt', [allowed], {}, [])

    def test_cli_five_roots_and_overlap_protection(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            cache = root / 'cache'
            (cache / 'nus').mkdir(parents=True)
            roots = [root / p.name for p in DEFAULT_SOURCE_ROOTS]
            for name in ('mir.zip', 'ann.zip'):
                with zipfile.ZipFile(root / name, 'w') as z:
                    z.writestr('README.txt', 'fixture')
            common = [sys.executable, str(HERE / 'prepare_all.py'), '--inspect-only',
                      '--mir-images', str(root / 'mir.zip'), '--mir-annotations', str(root / 'ann.zip'),
                      '--nus-root', str(cache / 'nus'), '--cache-root', str(cache),
                      '--old-root', str(root / 'old')]
            for out, allowed, valid in [(root / 'out', roots, True),
                                        (roots[4] / 'output', roots, False),
                                        (root / 'out_broad', [root], False)]:
                with self.subTest(out=out):
                    result = subprocess.run(common + ['--output-root', str(out), '--source-roots'] +
                                            [str(p) for p in allowed], capture_output=True, text=True)
                    if valid:
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual(json.loads((out / 'summary.json').read_text())['status'], 'INSPECTED_ONLY')
                    else:
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn('overlaps a source', result.stderr)
                        self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
