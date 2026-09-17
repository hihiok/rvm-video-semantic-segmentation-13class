import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from rev10_data import LABELS,sha
from decision_policy import decide
from model import create_ultraface_slim_scene8
from test_visualize_test_rev10 import fixture
from infer_fixed099 import run

class FixedTests(unittest.TestCase):
 def test_fixed_boundary_fallback_and_exclusion(self):
  s=np.full((4,8),.1);s[0,0]=.99;s[0,1]=.9899
  s[1,3]=.5;s[2,1]=.995;s[2,4]=.999;s[3,1]=s[3,4]=.99
  pred,notes=decide(s,np.full(8,.99))
  self.assertEqual([np.flatnonzero(r).tolist() for r in pred],[[0],[3],[4],[1]])
  self.assertTrue(notes[1]['fallback_top1']);self.assertFalse((pred[:,1]&pred[:,4]).any())
 def test_test_only_fresh_inference_without_tuning(self):
  with tempfile.TemporaryDirectory() as d:
   model,root,rows=fixture(d);torch.manual_seed(7)
   m=create_ultraface_slim_scene8()
   torch.save({'model':m.state_dict(),'labels':LABELS,'input_hw':[180,320],'thresholds':dict.fromkeys(LABELS,.5)},model/'best_ultraface_scene8.pth')
   before=sha(root/'test.jsonl');out=Path(d)/'fixed'
   # There are no VAL files in this fixture; threshold tuner must not be called.
   with patch('infer_precision_rules.tune_thresholds',side_effect=AssertionError('Must not tune')):
    a=run(model,root,out,device='cpu',batch=12,workers=0,n=1)
   self.assertEqual(a['status'],'INFERENCE_COMPLETE');self.assertEqual(sha(root/'test.jsonl'),before)
   self.assertEqual(set(a['policy']['thresholds'].values()),{.99})
   self.assertEqual(a['summaries']['fixed_099_context']['empty_prediction_images'],0)
   self.assertEqual(a['summaries']['fixed_099_context']['indoor_outdoor_overlap'],0)
   self.assertTrue(out.with_suffix('.zip').is_file())
   self.assertIn('未经validation选择',(out/'gallery/index.html').read_text())
   self.assertTrue(any(out.glob('gallery/*/*/*.jpg')))
if __name__=='__main__':unittest.main()
