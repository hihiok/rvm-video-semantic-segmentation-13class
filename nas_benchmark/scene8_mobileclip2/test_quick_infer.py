import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
from quick_infer import select_images, render


class QuickTests(unittest.TestCase):
    def test_sample_excludes_masks_and_needs_no_manifests(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'images').mkdir();(root/'masks').mkdir()
            for i in range(6):
                Image.new('RGB',(64,36),(i*20,40,50)).save(root/'images'/f'{i}.jpg')
                Image.new('P',(64,36)).save(root/'masks'/f'{i}.png')
            a,n=select_images(root,4,20260914)
            b,_=select_images(root,4,20260914)
            self.assertEqual(n,6);self.assertEqual(len(a),4);self.assertEqual(a,b)
            self.assertTrue(all('/images/' in r['image'] for r in a))
            out=root.parent/(root.name+'_unused')
            # Separate temporary output root, no torch or model download.
            with tempfile.TemporaryDirectory() as od:
                out=Path(od)/'preflight'
                p=subprocess.run([sys.executable,str(Path(__file__).with_name('quick_infer.py')),
                    '--input-dir',str(root),'--output-dir',str(out),'--count','4','--preflight-only'],
                    text=True,capture_output=True)
                self.assertEqual(p.returncode,0,p.stderr)
                self.assertEqual(json.loads((out/'STATUS.json').read_text())['gpu_tested'],False)
                self.assertFalse((out/'metrics.json').exists())
                self.assertFalse((out/'predictions.csv').exists())

    def test_palette_sample_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);Image.new('P',(64,36)).save(root/'suspect.png')
            with self.assertRaisesRegex(ValueError,'possible mask'):
                select_images(root,1,0)

    def test_render_has_eight_scores_without_gt(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'photo.jpg'
            Image.new('RGB',(640,360),(30,50,70)).save(src)
            out=root/'out';out.mkdir()
            scores=np.array([[.9,.1,.2,.3,.8,.7,.6,.4]])
            render([{'image':str(src)}],scores,out)
            with (out/'predictions.csv').open(encoding='utf-8-sig') as f:
                row=list(csv.DictReader(f))[0]
            self.assertEqual(sum(k.startswith('score_') for k in row),8)
            self.assertEqual(row['pred_night'],'1')
            self.assertEqual(row['pred_indoor'],'0')
            self.assertFalse(any(k.startswith('gt_') for k in row))
            self.assertTrue((out/'index.html').is_file())
            self.assertTrue((out/'contact_01.jpg').is_file())
            self.assertFalse((out/'metrics.json').exists())


if __name__=='__main__':
    unittest.main()
