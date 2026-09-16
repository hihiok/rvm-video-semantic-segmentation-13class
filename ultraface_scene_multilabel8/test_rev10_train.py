import json,os,tempfile,unittest
from pathlib import Path
import numpy as np,torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from rev10_data import LABELS,validate,resolution,dump
from train_rev10 import masked_loss,metrics,strict_gt,calibrate,export
from model import create_ultraface_slim_scene8

def fixture(root):
 import cv2
 root=Path(root);root.mkdir(parents=True);counts={}
 for split,n in [('train',16),('val',12),('test',12)]:
  rows=[]
  for i in range(n):
   p=root/(split+str(i)+'.jpg');cv2.imwrite(str(p),np.full((64,80,3),30+i*10,dtype=np.uint8))
   labels={k:(1 if i%2==0 else 0) for k in LABELS}
   if i%2:labels['night']=1 if i%4==1 else 0
   r={'image':str(p),'group_id':split+str(i),'source':'fixture','split':split,'use_for_training_manifest':True,'width':80,'height':64,'labels':labels,'evidence':{k:'manual:fixture' for k in LABELS}}
   if i%4==1:r['evidence']['rain_snow']='weak_user_rule:night_indoor_office_no_sports_weather_negative'
   rows.append(r)
  (root/(split+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows));counts[split]={'records':len(rows)}
 dump(root/'summary.json',{'schema':'nas8_weather_rev10','status':'PREPARED_REVIEW_REQUIRED','splits':counts});dump(root/'PREPARED.json',{});dump(root/'source_audit.json',{})
 return root

def ddp_worker(rank,path,out):
 torch.set_num_threads(1)
 dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=2)
 net=torch.nn.Linear(1,8,bias=False);net.weight.data.fill_(.1);ddp=DDP(net)
 x=torch.tensor([[1.],[2.]]) if rank==0 else torch.tensor([[3.],[4.]])
 y=torch.zeros(2,8) if rank==0 else torch.full((2,8),-1.)
 if rank==1:y[0,0]=1
 loss,_,_=masked_loss(ddp(x),y,torch.ones(8));loss.backward()
 if rank==0:np.save(out,net.weight.grad.numpy())
 dist.destroy_process_group()

class TrainingTests(unittest.TestCase):
 def test_unknown_zero_gradient(self):
  z=torch.zeros(1,8,requires_grad=True);y=torch.tensor([[1.,0.,-1.,-1.,-1.,-1.,-1.,-1.]])
  masked_loss(z,y,torch.ones(8))[0].backward();self.assertEqual(z.grad[0,2:].abs().sum(),0)
 @unittest.skipUnless(os.environ.get('NAS8_RUN_DDP_TEST')=='1','Distributed sockets must be tested on the training server')
 def test_ddp_global_known_count_gradient(self):
  with tempfile.TemporaryDirectory() as d:
   mp.spawn(ddp_worker,args=(d+'/init',d+'/grad.npy'),nprocs=2,join=True)
   net=torch.nn.Linear(1,8,bias=False);net.weight.data.fill_(.1)
   y=torch.cat([torch.zeros(2,8),torch.full((2,8),-1.)]);y[2,0]=1
   masked_loss(net(torch.tensor([[1.],[2.],[3.],[4.]])),y,torch.ones(8))[0].backward()
   np.testing.assert_allclose(np.load(d+'/grad.npy'),net.weight.grad.numpy(),rtol=1e-6,atol=1e-7)
 def test_global_mask_normalization_without_sockets(self):
  from unittest.mock import patch
  net=torch.nn.Linear(1,8,bias=False);net.weight.data.fill_(.1)
  x=torch.tensor([[1.],[2.],[3.],[4.]])
  y=torch.cat([torch.zeros(2,8),torch.full((2,8),-1.)]);y[2,0]=1
  masked_loss(net(x),y,torch.ones(8))[0].backward();expected=net.weight.grad.clone();grads=[]
  for start in (0,2):
   net.zero_grad()
   with patch('train_rev10.dist.is_initialized',return_value=True),patch('train_rev10.dist.get_world_size',return_value=2),patch('train_rev10.dist.all_reduce',side_effect=lambda t:t.fill_(17)):
    masked_loss(net(x[start:start+2]),y[start:start+2],torch.ones(8))[0].backward()
   grads.append(net.weight.grad.clone())
  torch.testing.assert_close((grads[0]+grads[1])/2,expected)
 def test_metrics_one_sided_and_ties(self):
  gt=np.ones((2,8));sc=np.full((2,8),.5);per,sm=metrics(gt,sc,np.full(8,.5))
  self.assertIsNone(per[0]['f1']);self.assertIsNone(per[0]['ap']);self.assertIsNone(sm['macro_ap'])
  gt[1,:]=0;per,_=metrics(gt,sc,np.full(8,.5));self.assertAlmostEqual(per[0]['ap'],.5)
  np.testing.assert_allclose(calibrate(np.ones((2,8)),sc),.5)
 def test_preflight_and_strict(self):
  with tempfile.TemporaryDirectory() as d:
   p=fixture(Path(d)/'data')
   with self.assertRaises(ValueError):validate(p,False)
   data,audit=validate(p,True);self.assertEqual(resolution(data['train'])['input_wh'],[320,180])
   self.assertEqual(strict_gt(data['train'])[1,2],-1)
   vv=data['val'];vv[0]['group_id']=data['train'][0]['group_id'];(p/'val.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in vv))
   with self.assertRaises(ValueError):validate(p,True)
 def test_export_dimensions_and_numerics(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);data,_=validate(fixture(root/'data'),True);out=root/'export';out.mkdir()
   export(create_ultraface_slim_scene8(),{'thresholds':dict.fromkeys(LABELS,.5)},out,[320,180],data['test'])
   meta=json.loads((out/'model.json').read_text());self.assertEqual(meta['input_shape'],[1,3,180,320]);self.assertEqual(json.loads((out/'onnx_check.json').read_text())['checker'],'PASS')
if __name__=='__main__':
 torch.set_num_threads(1);unittest.main()
