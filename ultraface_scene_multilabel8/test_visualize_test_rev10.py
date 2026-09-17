import json,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
from rev10_data import LABELS,sha,dump
from visualize_test_rev10 import read_inputs,select_cases,result,run,tier

def fixture(base):
 base=Path(base);root=base/'data';root.mkdir();model=base/'model';model.mkdir();rows=[]
 for i in range(36):
  im=root/('%03d.jpg'%i);Image.new('RGB',(120,80),(i*7%255,70,100)).save(im)
  labels=dict.fromkeys(LABELS,[-1,0,1][i%3]);ev=dict.fromkeys(LABELS,'manual:test')
  ev['rain_snow']='weak_user_rule:example' if labels['rain_snow']==0 else 'manual:test'
  rows.append({'image':str(im),'labels':labels,'evidence':ev,'split':'test','source':'fixture'})
 (root/'test.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 dump(model/'config.json',{'labels':LABELS,'data_sha256':{'test.jsonl':sha(root/'test.jsonl')},'input_wh':[320,180]})
 dump(model/'COMPLETE.json',{'status':'TRAINING_COMPLETE'});dump(model/'thresholds.json',dict.fromkeys(LABELS,.5))
 images=np.array([r['image'] for r in rows]);gt=np.array([[r['labels'][k] for k in LABELS] for r in rows]);scores=np.array([[.9 if (i//3+j)%2 else .1 for j in range(8)] for i in range(len(rows))])
 np.savez(model/'test_predictions.npz',image=images[::-1],gt=gt[::-1],scores=scores[::-1])
 return model,root,rows

class VisualTests(unittest.TestCase):
 def test_outcomes_and_unknown(self):
  self.assertEqual([result(y,p) for y,p in [(-1,True),(0,True),(0,False),(1,True),(1,False)]],['UNKNOWN','FP','TN','TP','FN'])
  gt=np.full((4,8),-1);gt[0]=1;gt[1]=0;scores=np.full((4,8),.5)
  chosen,stats=select_cases(gt,scores,np.full(8,.5),10,7)
  self.assertEqual(chosen['night']['correct'],[0]);self.assertEqual(chosen['night']['error'],[1]);self.assertEqual(stats['night']['error']['shortfall'],9)
 def test_deterministic_and_no_replacement(self):
  gt=np.tile(np.array([0,1]),(40,4));scores=np.random.default_rng(5).random((40,8));th=np.full(8,.5)
  a,_=select_cases(gt,scores,th,10,99);b,_=select_cases(gt,scores,th,10,99);self.assertEqual(a,b)
  for c in a.values():
   self.assertEqual(len(c['correct']),len(set(c['correct'])));self.assertFalse(set(c['correct'])&set(c['error']))
 def test_path_alignment_and_gt_mismatch(self):
  with tempfile.TemporaryDirectory() as d:
   model,root,rows=fixture(d);ordered,gt,_,_,_,_,_=read_inputs(model,root)
   self.assertEqual(ordered[0]['image'],rows[0]['image']);np.testing.assert_equal(gt[0],-1)
   p=model/'test_predictions.npz'
   with np.load(p) as z:data={k:z[k].copy() for k in z.files}
   data['gt'][0,0]=-1;np.savez(p,**data)
   with self.assertRaisesRegex(ValueError,'disagree'):read_inputs(model,root)
 def test_full_gallery_and_input_immutability(self):
  with tempfile.TemporaryDirectory() as d:
   model,root,rows=fixture(d);before=sha(root/'test.jsonl');out=Path(d)/'gallery';summary=run(model,root,out)
   self.assertEqual(summary['display_records'],160);self.assertEqual(sha(root/'test.jsonl'),before)
   self.assertEqual(len(list(out.glob('*/*/contact.jpg'))),16)
   self.assertTrue(out.with_suffix('.zip').is_file())
   records=[json.loads(x) for x in (out/'selected_cases.jsonl').read_text().splitlines()]
   self.assertTrue(any(r['focus_class']=='rain_snow' and r['focus_evidence_tier']=='WEAK/RULE' for r in records))
   for r in records:
    self.assertIn(r['focus_gt'],[0,1]);self.assertTrue((out/r['card']).exists())
   with self.assertRaises(ValueError):run(model,root,out)
if __name__=='__main__':unittest.main()
