"""Migration checks plus miniature complete pipeline (native parsers tested separately)."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import prepare_all
from relocation import DATASET_NAMES, migration_maps, normalized_maps, relocate
from sources import legacy_rows
from storage import Blocked, sha
from test_legacy_roots import manifests
from test_pipeline import make_image, record

HERE = Path(__file__).parent
OLD = Path('/data/pub1/z00919662/segmentation/datasets')


class Relocation(unittest.TestCase):
    def test_prefix_only_nested_suffix_and_already_migrated(self):
        maps = normalized_maps(migration_maps('/mnt/new'))
        self.assertEqual(relocate(OLD/'places365/versions/1/train/office/a.jpg', maps),
                         Path('/mnt/new/places365/versions/1/train/office/a.jpg'))
        self.assertEqual(relocate('/mnt/new/coco/a.jpg', maps), Path('/mnt/new/coco/a.jpg'))
        self.assertEqual(relocate(OLD/'coco_backup/a.jpg', maps), OLD/'coco_backup/a.jpg')

    def test_traversal_and_conflicting_mappings_rejected(self):
        with self.assertRaises(Blocked):relocate('/old/coco/../a.jpg', [])
        with self.assertRaises(Blocked):relocate('relative/a.jpg', [])
        with self.assertRaises(Blocked):normalized_maps([('/old', '/one'), ('/old', '/two')])

    def test_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'coco').mkdir();(root/'outside').mkdir()
            (root/'coco/link').symlink_to(root/'outside', target_is_directory=True)
            with self.assertRaisesRegex(Blocked, 'escapes target'):
                relocate('/old/link/a.jpg', normalized_maps([('/old', root/'coco')]))

    def test_missing_images_audited_without_rewriting_old_labels(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);old=root/'old'
            manifests(old,[dict(image=str(OLD/'coco/missing.jpg'),source='coco',labels={'night':1})])
            before=sha(old/'train.jsonl');audit={}
            with self.assertRaisesRegex(Blocked, 'MIGRATION_IMAGES_MISSING'):
                legacy_rows(old,HERE/'places365_io.txt',[root/'coco'],audit,[],migration_maps(root),True)
            self.assertEqual(sha(old/'train.jsonl'),before)
            self.assertEqual(audit['path_relocation']['missing_images'],1)
            self.assertEqual(audit['path_relocation']['missing_examples'][0]['new'],str(root/'coco/missing.jpg'))

    def test_new_allowed_root_does_not_accept_unmapped_external_image(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);old=root/'old'
            manifests(old,[dict(image='/unknown/coco/a.jpg',source='coco')])
            with self.assertRaisesRegex(Blocked, 'outside allowed'):
                legacy_rows(old,HERE/'places365_io.txt',[root/'coco'],{},[],migration_maps(root),True)

    def test_early_missing_image_failure_keeps_diagnostics_and_skips_native_parsers(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);old=root/prepare_all.OLD.name
            manifests(old,[dict(image=str(OLD/'coco/missing.jpg'),source='coco')])
            args=['prepare_all.py','--relocated-datasets-root',str(root),'--mir-image-root',str(root/'mir'),
                  '--mir-annotation-root',str(root/'ann'),'--nus-root',str(root/'nus')]
            with patch('sys.argv',args), patch('prepare_all.mir_rows') as mir:
                with self.assertRaisesRegex(Blocked,'MIGRATION_IMAGES_MISSING'):prepare_all.main()
                mir.assert_not_called()
            out=root/'NAS8_multilabel_clean_v3_rev5'
            self.assertTrue((out/'BLOCKED.json').is_file())
            self.assertFalse((out/'train.jsonl').exists())

    def test_complete_migrated_pipeline_gt100_and_new_paths(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);old=root/prepare_all.OLD.name;records=[];index=0
            for name,source,count,folder in [('coco','coco',15,''),('places365','places365',20,'office'),
                (DATASET_NAMES[2],'seg13',15,''),('10_scenes','10_scenes',5,'Computer_synthesized'),
                ('AWB_10_scenes','10_scenes',5,'Night')]:
                for i in range(count):
                    suffix=Path(folder)/('im%d.jpg'%i);make_image(root/name/suffix,index);index+=1
                    records.append(dict(image=str(OLD/name/suffix),source=source,detail=folder,labels={'night':0}))
            manifests(old,records);before=sha(old/'train.jsonl')
            native={}
            for folder,source in [('mir','mirflickr'),('nus','nuswide')]:
                native[source]=[]
                for i in range(20):
                    path=root/folder/('im%d.jpg'%i);make_image(path,index);index+=1
                    native[source].append(record(path,source,i=i))
            (root/'ann').mkdir()
            out=root/'custom_out'
            args=['prepare_all.py','--relocated-datasets-root',str(root),'--output-root',str(out),
                  '--mir-image-root',str(root/'mir'),'--mir-annotation-root',str(root/'ann'),'--nus-root',str(root/'nus')]
            with patch('sys.argv',args), patch('prepare_all.mir_rows',return_value=native['mirflickr']), \
                 patch('prepare_all.nus_rows',return_value=native['nuswide']), \
                 patch('source_preflight.export_nus_diagnostics'), contextlib.redirect_stdout(io.StringIO()):
                prepare_all.main()
            summary=json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['status'],'PREPARED_REVIEW_REQUIRED')
            self.assertEqual(summary['gt_visualizations'],100)
            self.assertFalse(summary['training_started'])
            self.assertEqual(len(list((out/'gt100').glob('0*.jpg')))+len(list((out/'gt100').glob('100_*.jpg'))),100)
            self.assertTrue((out/'gt100/index.html').is_file())
            self.assertTrue((out/'resolution_audit.json').is_file())
            self.assertEqual(sha(old/'train.jsonl'),before)
            audit=json.loads((out/'source_audit.json').read_text())
            self.assertEqual(audit['path_relocation']['changed_rows'],60)
            self.assertEqual(audit['path_relocation']['missing_images'],0)
            self.assertEqual(audit['legacy_root_counts'][str(root/'AWB_10_scenes')],5)
            for r in map(json.loads,(out/'source_records.jsonl').read_text().splitlines()):
                self.assertTrue(Path(r['image']).is_file())
                self.assertTrue(str(r['image']).startswith(str(root)))
