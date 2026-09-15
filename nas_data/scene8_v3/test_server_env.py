"""User-approved environment and AWB -> 10_scenes replacement."""
import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import prepare_all
from storage import Blocked,sha
from test_legacy_roots import manifests
from test_pipeline import make_image

HERE=Path(__file__).parent

class ServerEnvironment(unittest.TestCase):
    def test_wrapper_accepts_ultraface_new_and_rejects_unrelated_env(self):
        for name,ok in [('ultraface_new',True),('unrelated_environment',False)]:
            with self.subTest(name=name):
                env=dict(os.environ,CONDA_DEFAULT_ENV=name)
                r=subprocess.run(['bash',str(HERE/'run_prepare.sh'),'--help'],env=env,capture_output=True,text=True)
                self.assertEqual(r.returncode==0,ok,r.stderr)
                if ok:self.assertIn('--legacy-path-map',r.stdout)
                else:self.assertIn('HUMAN_ACTION_REQUIRED',r.stderr)

    def test_explicit_awb_replacement_reaches_mir_with_four_roots(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);old=root/prepare_all.OLD.name;out=root/'out'
            image=root/'10_scenes/Night/a.jpg';make_image(image,1)
            prefixes=['/data/pub1/z00919662/dataset/AWB_10_scenes',
                      '/data/pub1/z00919662/segmentation/datasets/AWB_10_scenes',
                      str(root/'AWB_10_scenes')]
            manifests(old,[dict(image=p+'/Night/a.jpg',source='10_scenes',detail='Night->night') for p in prefixes])
            before=sha(old/'train.jsonl')
            for d in ['mir','ann','nus']:(root/d).mkdir()
            args=['prepare_all.py','--relocated-datasets-root',str(root),'--output-root',str(out),
                  '--mir-image-root',str(root/'mir'),'--mir-annotation-root',str(root/'ann'),
                  '--nus-root',str(root/'nus'),'--source-roots']
            args += [str(root/d) for d in ['coco','places365','COCO_ADE_13cls_16x9_640x360','10_scenes']]
            for p in prefixes:args += ['--legacy-path-map',p,str(root/'10_scenes')]
            captured=[]
            original=prepare_all.legacy_rows
            def capture(*a,**kw):
                rows=original(*a,**kw);captured.extend(rows);return rows
            with patch('sys.argv',args),patch('prepare_all.legacy_rows',side_effect=capture), \
                 patch('prepare_all.mir_rows',side_effect=Blocked('TEST_NATIVE_STAGE_REACHED')), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(Blocked,'TEST_NATIVE_STAGE_REACHED'):prepare_all.main()
            audit=json.loads((out/'source_audit.json').read_text())
            self.assertEqual(audit['legacy_root_counts'][str(root/'10_scenes')],3)
            self.assertNotIn(str(root/'AWB_10_scenes'),audit['legacy_root_counts'])
            self.assertEqual(audit['path_relocation']['missing_images'],0)
            self.assertEqual(sha(old/'train.jsonl'),before)
            self.assertFalse((root/'AWB_10_scenes').exists())
            self.assertTrue(all(r['image']==str(image) and r['labels']['night']==1 and
                                r['labels']['rain_snow']==-1 for r in captured))
