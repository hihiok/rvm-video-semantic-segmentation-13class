import json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from decision_policy import decide,tune_thresholds,top1_099
from model import create_ultraface_slim_scene8
from rev10_data import LABELS,dump,sha
from train_rev10 import metrics
from test_visualize_test_rev10 import fixture
from visualize_test_rev10 import select_cases,read_inputs
from infer_precision_rules import run

class DecisionTests(unittest.TestCase):
 def test_fallback_only_when_no_class_qualifies(self):
  s=np.full((2,8),.1);s[0,0]=.7;s[0,1]=.8;s[1,0]=.2
  th=np.full(8,.9);th[0]=.6
  out,notes=decide(s,th)
  self.assertEqual(np.flatnonzero(out[0]).tolist(),[0]) # Raw top1 indoor doesn't pass; do not force it.
  self.assertFalse(notes[0]['fallback_top1']);self.assertTrue(notes[1]['fallback_top1'])
  self.assertEqual(np.flatnonzero(out[1]).tolist(),[0])
 def test_indoor_outdoor_never_coexist_even_tie(self):
  s=np.full((3,8),.1);s[:,1]=[.8,.81,.9];s[:,4]=[.8,.83,.6]
  out,notes=decide(s,np.full(8,.5));self.assertFalse((out[:,1]&out[:,4]).any())
  self.assertEqual(out[:,1].tolist(),[True,False,True]);self.assertTrue(all(out.sum(1)>=1))
  self.assertIn('indoor_outdoor_close_scores_winner_kept',notes[0]['ambiguities'])
 def test_indoor_weather_night_landscape_not_forbidden(self):
  s=np.full((1,8),.8);s[0,4]=.1
  out,notes=decide(s,np.full(8,.7));self.assertTrue(out[0,[0,1,2,3,5,6,7]].all())
  np.testing.assert_equal(s,np.array([[.8,.8,.8,.8,.1,.8,.8,.8]]))
 def test_target_99_differs_from_score_99_unknown_ignored(self):
  gt=np.full((81,8),-1);scores=np.full((81,8),.999)
  gt[:40]=1;gt[40:80]=0;scores[:40]=.8;scores[40:80]=.6
  th,a=tune_thresholds(gt,scores,np.full(8,.5))
  np.testing.assert_allclose(th,.8)
  self.assertTrue(a['office']['target_met']);self.assertEqual(a['office']['predicted_known'],40)
  self.assertEqual(a['office']['unknown'],1)
 def test_unmet_or_insufficient_does_not_lower_target(self):
  gt=np.tile(np.r_[np.ones(40),np.zeros(40)][:,None],(1,8));s=np.full((80,8),.9)
  th,a=tune_thresholds(gt,s,np.full(8,.5));self.assertTrue((th>1).all())
  self.assertEqual(a['night']['status'],'target_unmet_no_threshold_emission')
  out,notes=decide(s,th);self.assertTrue((out.sum(1)==1).all());self.assertTrue(notes[0]['fallback_top1'])
  th2,a2=tune_thresholds(np.full((80,8),-1),s,np.full(8,.5));self.assertTrue((th2>1).all())
 def test_threshold_floor_and_invalid_inputs(self):
  gt=np.tile(np.r_[np.ones(40),np.zeros(40)][:,None],(1,8));s=np.where(gt==1,.8,.2)
  th,a=tune_thresholds(gt,s,np.full(8,.9));self.assertTrue((th>1).all())
  for bad in [np.full((1,8),np.nan),np.full((1,8),1.01),np.zeros((1,7))]:
   with self.assertRaises(ValueError):decide(bad,np.full(8,.5))
 def test_all_099_baseline_and_actual_decisions_scored(self):
  gt=np.tile(np.array([1,0])[:,None],(1,8));s=np.full((2,8),.6);th=np.full(8,.5)
  base,base_sum=metrics(gt,s,th);chosen=np.zeros((2,8),bool)
  per,summary=metrics(gt,s,th,decisions=chosen)
  self.assertEqual(per[0]['fn'],1);self.assertEqual(per[0]['fp'],0)
  self.assertEqual(base[0]['ap'],per[0]['ap'])
  cases,_=select_cases(gt,s,th,10,5,decisions=chosen)
  self.assertEqual(cases['night']['correct'],[1]);self.assertEqual(cases['night']['error'],[0])
  self.assertTrue((top1_099(s).sum(1)==1).all())
 def test_cpu_fresh_val_test_inference_and_gallery(self):
  with tempfile.TemporaryDirectory() as d:
   model,root,rows=fixture(d)
   val=[]
   for r in rows:
    rr=dict(r);rr['split']='val'
    src=Path(r['image']);dst=root/('val_'+src.name);dst.write_bytes(src.read_bytes());rr['image']=str(dst);val.append(rr)
   manifest=root/'val.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in val))
   cfg=json.loads((model/'config.json').read_text());cfg['data_sha256']['val.jsonl']=sha(manifest);dump(model/'config.json',cfg)
   with np.load(model/'test_predictions.npz') as z:
    np.savez(model/'val_predictions.npz',image=np.array([r['image'] for r in val][::-1]),gt=z['gt'],scores=z['scores'])
   torch.manual_seed(3);m=create_ultraface_slim_scene8()
   torch.save({'model':m.state_dict(),'labels':LABELS,'input_hw':[180,320],'thresholds':dict.fromkeys(LABELS,.5)},model/'best_ultraface_scene8.pth')
   before=sha(root/'test.jsonl');out=Path(d)/'output'
   audit=run(model,root,out,device='cpu',batch=12,workers=0,n=1,min_predictions=3)
   self.assertEqual(audit['status'],'INFERENCE_COMPLETE');self.assertEqual(sha(root/'test.jsonl'),before)
   self.assertEqual(audit['indoor_outdoor_output_overlap'],0)
   self.assertEqual(audit['summaries']['precision_context']['empty_prediction_images'],0)
   self.assertTrue((out/'gallery/index.html').is_file());self.assertTrue(out.with_suffix('.zip').is_file())
   with np.load(out/'test_inference.npz') as z:
    self.assertTrue((z['precision_context'].sum(1)>=1).all());self.assertEqual(z['scores'].shape,(36,8))
   self.assertEqual(sha(out/'decision_policy.json'),audit['decision_policy_sha256'])
   with self.assertRaises(ValueError):run(model,root,out,device='cpu',workers=0)
if __name__=='__main__':unittest.main()
