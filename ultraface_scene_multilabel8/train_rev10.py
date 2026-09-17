#!/usr/bin/env python3
"""Rev10 NAS8 training: unchanged Mb_Tiny, DDP, masked BCE, calibrated evaluation."""
import argparse,datetime,inspect,json,os,random,time
from pathlib import Path
import cv2,numpy as np,torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset,DataLoader,DistributedSampler
from model import create_ultraface_slim_scene8
from train import pos_weight,save_csv
from rev10_data import LABELS,validate,resolution,dump,sha

def parse():
 p=argparse.ArgumentParser()
 p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
 p.add_argument('--accept-weak-weather',action='store_true');p.add_argument('--epochs',type=int,default=60)
 p.add_argument('--global-batch-size',type=int,default=128);p.add_argument('--workers',type=int,default=2)
 p.add_argument('--lr',type=float,default=.01);p.add_argument('--milestones',default='35,50')
 p.add_argument('--amp',action='store_true');p.add_argument('--resume',type=Path)
 p.add_argument('--seed',type=int,default=20260916);p.add_argument('--smoke',action='store_true')
 p.add_argument('--max-train-steps',type=int,default=0);p.add_argument('--cpu-test',action='store_true',help=argparse.SUPPRESS)
 return p.parse_args()
class Scenes(Dataset):
 def __init__(self,rows,wh,train=False):self.rows=rows;self.wh=tuple(wh);self.train=train
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i];im=cv2.imread(r['image'])
  if im is None:raise RuntimeError('Cannot decode '+r['image'])
  im=cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),self.wh,interpolation=cv2.INTER_LINEAR)
  if self.train and random.random()<.5:im=im[:,::-1]
  im=im.astype(np.float32)
  if self.train:im=np.clip(im*random.uniform(.9,1.1)+random.uniform(-10,10),0,255)
  return torch.from_numpy(np.ascontiguousarray(((im-127)/128).transpose(2,0,1))),torch.tensor([r['labels'][k] for k in LABELS],dtype=torch.float32)
def worker_init(_):
 seed=torch.initial_seed()%2**32;np.random.seed(seed);random.seed(seed);cv2.setNumThreads(0)
def masked_loss(z,y,w):
 mask=y>=0;num=(F.binary_cross_entropy_with_logits(z,y.clamp(0,1),pos_weight=w,reduction='none')*mask).sum()
 denominator=mask.sum().float()
 world=dist.get_world_size() if dist.is_initialized() else 1
 if world>1:dist.all_reduce(denominator)
 # DDP averages gradients, so compensate before dividing by GLOBAL known labels.
 return num*world/denominator.clamp_min(1),num.detach(),mask.sum().detach()
def average_precision(y,s):
 order=np.argsort(-s,kind='stable');yy=y[order];ss=s[order];tp=np.cumsum(yy==1)
 ends=np.r_[np.where(np.diff(ss)!=0)[0],len(ss)-1]
 recall=tp[ends]/sum(y==1);precision=tp[ends]/(ends+1)
 return float(np.sum(np.diff(np.r_[0,recall])*precision))
def metrics(gt,sc,threshold,decisions=None):
 if decisions is not None and np.asarray(decisions).shape!=gt.shape:raise ValueError("Decision shape mismatch")
 result=[]
 for j,label in enumerate(LABELS):
  m=gt[:,j]>=0;y=gt[m,j];s=sc[m,j];p=s>=threshold[j] if decisions is None else np.asarray(decisions,dtype=bool)[m,j]
  pos=int(sum(y==1));neg=int(sum(y==0));tp=int(sum(p&(y==1)));fp=int(sum(p&(y==0)));fn=pos-tp;tn=neg-fp
  supported=pos>0 and neg>0
  result.append({'label':label,'threshold':float(threshold[j]),'positive':pos,'negative':neg,'unknown':int(sum(~m)),
   'precision':tp/(tp+fp) if supported and tp+fp else (0. if supported else None),
   'recall':tp/pos if pos else None,'specificity':tn/neg if neg else None,
   'f1':2*tp/max(2*tp+fp+fn,1) if supported else None,
   'accuracy':(tp+tn)/(pos+neg) if supported else None,
   'balanced_accuracy':.5*(tp/pos+tn/neg) if supported else None,
   'ap':average_precision(y,s) if supported else None,'tp':tp,'fp':fp,'fn':fn,'tn':tn,'two_sided_support':supported})
 summary={}
 for key in ('f1','balanced_accuracy','ap'):
  values=[r[key] for r in result if r[key] is not None];summary['macro_'+key]=float(np.mean(values)) if values else None
 summary['supported_classes']=[r['label'] for r in result if r['two_sided_support']]
 return result,summary
def calibrate(gt,sc):
 th=np.full(8,.5)
 for j in range(8):
  m=gt[:,j]>=0;y=gt[m,j];s=sc[m,j]
  if not (np.any(y==1) and np.any(y==0)):continue
  best=(-1.,-1.)
  for t in np.linspace(.05,.95,91):
   pred=s>=t;tp=sum(pred&(y==1));fp=sum(pred&(y==0));fn=sum((~pred)&(y==1));f=2*tp/max(2*tp+fp+fn,1)
   # Prefer threshold closest to .5 for ties.
   score=(f,-abs(t-.5))
   if score>best:best=score;th[j]=t
 return th
def strict_gt(rows):
 return np.array([[r['labels'][k] if r.get('evidence',{}).get(k,'').startswith(('manual:','human_review:','user_review:')) else -1 for k in LABELS] for r in rows],dtype=np.float32)
def evaluate(model,loader,device,amp,max_batches=0):
 model.eval();gs=[];ss=[]
 with torch.inference_mode():
  for i,(x,y) in enumerate(loader):
   if max_batches and i>=max_batches:break
   with torch.cuda.amp.autocast(enabled=amp):z=model(x.to(device,non_blocking=True))
   gs.append(y.numpy());ss.append(z.float().sigmoid().cpu().numpy())
 return np.concatenate(gs),np.concatenate(ss)
def atomic_save(obj,path):
 tmp=path.with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,path)
def torch_load(path):
 return torch.load(path,map_location='cpu',**({'weights_only':False} if 'weights_only' in inspect.signature(torch.load).parameters else {}))
def barrier():
 if dist.is_initialized():dist.barrier()
def export(model,ck,out,wh,test_rows):
 import onnx,onnxruntime as ort
 model=model.cpu().eval();w,h=wh
 generator=torch.Generator().manual_seed(99);x=torch.randn(2,3,h,w,generator=generator)
 path=out/'model.onnx';kw={'dynamo':False} if 'dynamo' in inspect.signature(torch.onnx.export).parameters else {}
 torch.onnx.export(model,x,str(path),input_names=['image'],output_names=['logits'],opset_version=13,dynamic_axes={'image':{0:'batch'},'logits':{0:'batch'}},**kw)
 onnx.checker.check_model(onnx.load(str(path)));options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
 session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
 errors=[]
 real=Scenes(test_rows[:2],wh)
 for batch in (x,torch.stack([real[i][0] for i in range(len(real))])):
  with torch.inference_mode():expected=model(batch).numpy()
  actual=session.run(None,{'image':batch.numpy()})[0]
  np.testing.assert_allclose(actual,expected,rtol=1e-3,atol=1e-4);errors.append(float(np.max(np.abs(actual-expected))))
 dump(out/'onnx_check.json',{'checker':'PASS','pytorch_ort_max_abs_errors':errors})
 dump(out/'model.json',{'input_shape':[1,3,h,w],'output':'logits; apply sigmoid then per-class thresholds','labels':LABELS,'thresholds':ck['thresholds'],'preprocess':'RGB; direct resize WxH; (pixel-127)/128; CHW float32','input_wh':wh,'weak_weather_notice':'Rain/snow threshold calibrated using weak-rule negatives; not an individually verified weather operating point.'})

def main():
 a=parse();world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'));local=int(os.environ.get('LOCAL_RANK','0'))
 if a.cpu_test and not a.smoke:raise ValueError('CPU test only allowed for smoke')
 if a.max_train_steps and not a.smoke:raise ValueError('Step cap only allowed for smoke')
 if a.epochs<1 or a.global_batch_size<world*2 or a.global_batch_size%world:raise ValueError('Invalid epochs or divisible global batch')
 if not a.cpu_test and not torch.cuda.is_available():raise RuntimeError('CUDA required')
 if not a.cpu_test:torch.cuda.set_device(local)
 device=torch.device('cpu' if a.cpu_test else 'cuda:'+str(local));amp=a.amp and device.type=='cuda'
 torch.set_num_threads(2);cv2.setNumThreads(0)
 if world>1:dist.init_process_group('gloo' if a.cpu_test else 'nccl',timeout=datetime.timedelta(hours=2))
 random.seed(a.seed+rank);np.random.seed(a.seed+rank);torch.manual_seed(a.seed+rank)
 if device.type=='cuda':torch.cuda.manual_seed_all(a.seed+rank);torch.backends.cudnn.benchmark=True
 # Rank zero audits files; other ranks load only after a successful audit.
 if rank==0:
  if a.output_dir.exists() and not a.resume:raise ValueError('New run requires a new output directory')
  if a.resume and a.resume.resolve().parent!=a.output_dir.resolve():raise ValueError('Resume checkpoint must belong to output directory')
  root=a.data_root.resolve();out=a.output_dir.resolve()
  if root==out or root in out.parents:raise ValueError('Output cannot be inside label root')
  data,audit=validate(a.data_root,a.accept_weak_weather);res=resolution(data['train']);wh=res['input_wh']
  a.output_dir.mkdir(parents=True,exist_ok=True)
  config={'labels':LABELS,'input_wh':wh,'data_sha256':audit['sha256'],'global_batch_size':a.global_batch_size,'lr':a.lr,'milestones':a.milestones,'seed':a.seed,'amp':amp,'epochs':a.epochs,'world_size':world,'smoke':a.smoke}
  if a.resume:
   previous=json.loads((a.output_dir/'config.json').read_text())
   for k in ('labels','input_wh','data_sha256','global_batch_size','lr','milestones','seed','amp','smoke'):
    if previous[k]!=config[k]:raise ValueError('Resume config changed: '+k)
  dump(a.output_dir/'config.json',config);dump(a.output_dir/'data_audit.json',audit);dump(a.output_dir/'resolution_audit.json',res)
  print('PREFLIGHT_PASS',json.dumps(config),flush=True)
 else:data=audit=res=wh=None
 payload=[wh]
 if world>1:dist.broadcast_object_list(payload,src=0)
 wh=payload[0];barrier()
 if rank!=0:
  from rev10_data import load
  data={s:load(a.data_root/(s+'.jsonl')) for s in ('train','val','test')}
 tr=data['train'];sampler=DistributedSampler(tr,num_replicas=world,rank=rank,shuffle=True,seed=a.seed,drop_last=False) if world>1 else None
 generator=torch.Generator().manual_seed(a.seed+rank)
 loader_context={'multiprocessing_context':'spawn'} if a.workers>0 else {}
 trainloader=DataLoader(Scenes(tr,wh,True),batch_size=a.global_batch_size//world,shuffle=sampler is None,sampler=sampler,num_workers=a.workers,pin_memory=device.type=='cuda',persistent_workers=a.workers>0,worker_init_fn=worker_init,generator=generator,**loader_context)
 valloader=DataLoader(Scenes(data['val'],wh),batch_size=a.global_batch_size,num_workers=a.workers,pin_memory=device.type=='cuda',persistent_workers=a.workers>0,worker_init_fn=worker_init,**loader_context) if rank==0 else None
 base=create_ultraface_slim_scene8().to(device);model=DDP(base,device_ids=[local] if device.type=='cuda' else None) if world>1 else base
 weights,pc,nc=pos_weight(tr);weights=weights.to(device)
 opt=torch.optim.SGD(model.parameters(),lr=a.lr,momentum=.9,weight_decay=1e-4)
 scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,[int(v) for v in a.milestones.split(',')],gamma=.1)
 scaler=torch.cuda.amp.GradScaler(enabled=amp);start=0;best=-1.
 if a.resume:
  ck=torch_load(a.resume);base.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);scaler.load_state_dict(ck['scaler']);start=ck['epoch']+1;best=ck['best_macro_ap']
 if rank==0:dump(a.output_dir/'loss_weights.json',dict(zip(LABELS,map(float,weights.cpu()))))
 for epoch in range(start,a.epochs):
  if sampler:sampler.set_epoch(epoch)
  model.train();numerator=torch.zeros((),device=device);denominator=torch.zeros((),device=device);t0=time.time();steps=0
  for step,(x,y) in enumerate(trainloader):
   if a.max_train_steps and step>=a.max_train_steps:break
   opt.zero_grad(set_to_none=True);x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
   with torch.cuda.amp.autocast(enabled=amp):loss,num,den=masked_loss(model(x),y,weights)
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
   scaler.scale(loss).backward();scaler.step(opt);scaler.update();numerator+=num;denominator+=den;steps+=1
   if rank==0 and step%100==0:print('TRAIN',epoch+1,step,float(loss.detach()),flush=True)
  if world>1:dist.all_reduce(numerator);dist.all_reduce(denominator)
  scheduler.step();barrier()
  if rank==0:
   # Unwrapped model: other ranks do not enter evaluation collectives.
   g,s=evaluate(base,valloader,device,amp,10 if a.smoke else 0);per,sm=metrics(g,s,np.full(8,.5))
   score=sm['macro_ap'];score=-1. if score is None else score;improved=score>best or not (a.output_dir/'best_train_state.pth').exists()
   if improved:best=score
   state={'epoch':epoch,'model':base.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'best_macro_ap':best,'labels':LABELS,'input_hw':[wh[1],wh[0]]}
   if improved:atomic_save(state,a.output_dir/'best_train_state.pth');save_csv(a.output_dir/'best_val_0p5.csv',per)
   atomic_save(state,a.output_dir/'last_train_state.pth')
   seconds=time.time()-t0;rec={'epoch':epoch+1,'train_loss':float(numerator/denominator.clamp_min(1)),'seconds':seconds,'estimated_remaining_hours':seconds*(a.epochs-epoch-1)/3600,'steps':steps,**sm}
   with (a.output_dir/'metrics.jsonl').open('a') as f:f.write(json.dumps(rec,allow_nan=False)+'\n')
   print('VAL',json.dumps(rec),flush=True)
  barrier()
 if rank==0:
  if a.smoke:dump(a.output_dir/'SMOKE_PASS.json',{'status':'PASS','epochs':a.epochs,'world_size':world,'input_wh':wh,'cpu_test_only':a.cpu_test})
  else:
   ck=torch_load(a.output_dir/'best_train_state.pth');base.load_state_dict(ck['model'])
   gv,sv=evaluate(base,valloader,device,amp);thresholds=calibrate(gv,sv)
   testloader=DataLoader(Scenes(data['test'],wh),batch_size=a.global_batch_size,num_workers=a.workers,pin_memory=True,worker_init_fn=worker_init,**loader_context)
   gt,st=evaluate(base,testloader,device,amp)
   for split,g,s in [('val',gv,sv),('test',gt,st)]:
    per,sm=metrics(g,s,thresholds);save_csv(a.output_dir/(split+'_per_class_calibrated.csv'),per);dump(a.output_dir/(split+'_summary.json'),sm)
    per,sm=metrics(strict_gt(data[split]),s,thresholds);save_csv(a.output_dir/(split+'_strict_per_class.csv'),per);dump(a.output_dir/(split+'_strict_summary.json'),sm)
    np.savez_compressed(a.output_dir/(split+'_predictions.npz'),gt=g,scores=s,image=np.array([r['image'] for r in data[split]]))
   deployment={'model':base.state_dict(),'labels':LABELS,'thresholds':dict(zip(LABELS,map(float,thresholds))),'input_hw':[wh[1],wh[0]],'architecture':'original UltraFace Mb_Tiny backbone + GAP + Linear(256,8)'}
   atomic_save(deployment,a.output_dir/'best_ultraface_scene8.pth');dump(a.output_dir/'thresholds.json',deployment['thresholds'])
   export(base,deployment,a.output_dir,wh,data['test'])
   for name,expected in audit['sha256'].items():
    if sha(a.data_root/name)!=expected:raise RuntimeError('Dataset changed during training')
   dump(a.output_dir/'COMPLETE.json',{'status':'TRAINING_COMPLETE','epochs':a.epochs,'best_epoch':ck['epoch']+1,'input_wh':wh,'world_size':world,'data_root':str(a.data_root),'weak_weather_accepted':True,'strict_weather_negative_evidence':'unavailable; unsupported metrics null','test_selection':'test not used for checkpoint or thresholds','onnx_checked':True})
 barrier()
 if dist.is_initialized():dist.destroy_process_group()
if __name__=='__main__':main()
