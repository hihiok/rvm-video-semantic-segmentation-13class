#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,os,random,time
from pathlib import Path
import cv2,numpy as np,torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader,Dataset
from model import create_ultraface_slim_scene8

LABELS=['night','indoor','rain_snow','office','outdoor','landscape','sports','objective_image']
DISPLAY={'night':'夜景','indoor':'室内','rain_snow':'雨/雪','office':'办公场景','outdoor':'户外','landscape':'风景','sports':'运动','objective_image':'客观图'}

def parse_args():
 p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True)
 p.add_argument('--epochs',type=int,default=30); p.add_argument('--batch-size',type=int,default=128); p.add_argument('--workers',type=int,default=16); p.add_argument('--prefetch-factor',type=int,default=2)
 p.add_argument('--lr',type=float,default=1e-2); p.add_argument('--momentum',type=float,default=.9); p.add_argument('--weight-decay',type=float,default=1e-4); p.add_argument('--milestones',default='20,27'); p.add_argument('--gamma',type=float,default=.1)
 p.add_argument('--dropout',type=float,default=.1); p.add_argument('--seed',type=int,default=20260907); p.add_argument('--cpu-threads',type=int,default=4); p.add_argument('--amp',action='store_true'); p.add_argument('--resume',type=Path)
 p.add_argument('--log-every',type=int,default=50); p.add_argument('--max-train-steps',type=int,default=0); p.add_argument('--max-eval-batches',type=int,default=0); return p.parse_args()

def load(p): return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def worker_init(_): cv2.setNumThreads(0); torch.set_num_threads(1)

class DS(Dataset):
 def __init__(self,rows,train=False): self.rows=rows; self.train=train
 def __len__(self): return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]; im=cv2.imread(r['image'],cv2.IMREAD_COLOR)
  if im is None: raise RuntimeError('cannot read '+r['image'])
  if im.shape[:2]!=(360,640): im=cv2.resize(im,(640,360),interpolation=cv2.INTER_LINEAR)
  im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
  if self.train and random.random()<.5: im=np.ascontiguousarray(im[:,::-1])
  if self.train: im=cv2.convertScaleAbs(im,alpha=random.uniform(.9,1.1),beta=random.uniform(-10,10))
  x=torch.from_numpy(np.ascontiguousarray(im.transpose(2,0,1)))  # uint8: 4x smaller IPC/pinned-memory payload
  y=torch.tensor([float(r['labels'][k]) for k in LABELS],dtype=torch.float32); return x,y

def loader(ds,batch,shuffle,workers,prefetch,drop=False):
 kw=dict(batch_size=batch,shuffle=shuffle,num_workers=workers,pin_memory=True,drop_last=drop,worker_init_fn=worker_init)
 if workers>0: kw.update(persistent_workers=True,prefetch_factor=prefetch)
 return DataLoader(ds,**kw)

def to_gpu_image(x,dev): return x.to(dev,non_blocking=True).float().sub_(127.0).div_(128.0)
def pweight(rows):
 p=np.zeros(8); n=np.zeros(8)
 for r in rows:
  for j,k in enumerate(LABELS):
   v=int(r['labels'][k]); p[j]+=v==1; n[j]+=v==0
 return torch.tensor(np.clip(n/np.maximum(p,1),.5,8),dtype=torch.float32),p.astype(int),n.astype(int)
def lossfn(z,y,w):
 m=y>=0; raw=F.binary_cross_entropy_with_logits(z,y.clamp(0,1),reduction='none',pos_weight=w); return (raw*m).sum()/m.sum().clamp_min(1)
def AP(y,s):
 if (y==1).sum()==0:return float('nan')
 o=np.argsort(-s); q=y[o]; tp=np.cumsum(q==1); fp=np.cumsum(q==0); pr=tp/np.maximum(tp+fp,1); return float(pr[q==1].sum()/max((q==1).sum(),1))
def metric(gt,sc,th):
 rows=[]
 for j,k in enumerate(LABELS):
  m=gt[:,j]>=0; y=gt[m,j].astype(int); s=sc[m,j]; p=(s>=th[j]).astype(int); tp=((p==1)&(y==1)).sum(); fp=((p==1)&(y==0)).sum(); fn=((p==0)&(y==1)).sum(); tn=((p==0)&(y==0)).sum(); pr=tp/max(tp+fp,1); rc=tp/max(tp+fn,1); f=2*pr*rc/max(pr+rc,1e-12); ba=.5*(tp/max(tp+fn,1)+tn/max(tn+fp,1))
  rows.append({'label':k,'display_name':DISPLAY[k],'threshold':float(th[j]),'known':int(m.sum()),'positive':int((y==1).sum()),'negative':int((y==0).sum()),'precision':float(pr),'recall':float(rc),'f1':float(f),'accuracy':float((tp+tn)/max(len(y),1)),'balanced_accuracy':float(ba),'ap':AP(y,s)})
 return rows,{'macro_f1':float(np.mean([r['f1'] for r in rows])),'macro_balanced_accuracy':float(np.mean([r['balanced_accuracy'] for r in rows])),'macro_ap':float(np.nanmean([r['ap'] for r in rows]))}
def evaluate(model,dl,dev,amp,maxb=0):
 model.eval(); G=[];S=[]
 with torch.no_grad():
  for bi,(x,y) in enumerate(dl):
   if maxb and bi>=maxb: break
   x=to_gpu_image(x,dev)
   with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=amp): z=model(x)
   G.append(y.numpy()); S.append(torch.sigmoid(z).float().cpu().numpy())
 return np.concatenate(G),np.concatenate(S)
def calibrate(gt,sc):
 out=np.full(8,.5,dtype=np.float32)
 for j in range(8):
  m=gt[:,j]>=0;y=gt[m,j].astype(int);s=sc[m,j];best=(-1,.5)
  for t in np.linspace(.05,.95,91):
   p=(s>=t).astype(int);tp=((p==1)&(y==1)).sum();fp=((p==1)&(y==0)).sum();fn=((p==0)&(y==1)).sum();pr=tp/max(tp+fp,1);rc=tp/max(tp+fn,1);f=2*pr*rc/max(pr+rc,1e-12)
   if f>best[0]:best=(f,float(t))
  out[j]=best[1]
 return out
def savecsv(p,rows):
 with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 a=parse_args(); cv2.setNumThreads(0); os.environ.setdefault('OMP_NUM_THREADS',str(a.cpu_threads)); os.environ.setdefault('MKL_NUM_THREADS',str(a.cpu_threads)); torch.set_num_threads(a.cpu_threads); random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 dev=torch.device('cuda:0');torch.backends.cudnn.benchmark=True;a.output_dir.mkdir(parents=True,exist_ok=True);tr=load(a.data_root/'train.jsonl');va=load(a.data_root/'val.jsonl');te=load(a.data_root/'test.jsonl');trl=loader(DS(tr,True),a.batch_size,True,a.workers,a.prefetch_factor,True);vdl=loader(DS(va),a.batch_size,False,a.workers,a.prefetch_factor);tdl=loader(DS(te),a.batch_size,False,a.workers,a.prefetch_factor)
 print(f'DATASET train={len(tr)} val={len(va)} test={len(te)} steps_per_epoch={len(trl)} batch={a.batch_size} workers={a.workers} prefetch={a.prefetch_factor} cpu_payload=uint8',flush=True);w,pc,nc=pweight(tr);w=w.to(dev);print('TRAIN_SUPERVISION',dict(zip(LABELS,[{'pos':int(pc[i]),'neg':int(nc[i]),'pos_weight':float(w[i])} for i in range(8)])),flush=True)
 model=create_ultraface_slim_scene8(a.dropout).to(dev);opt=torch.optim.SGD(model.parameters(),lr=a.lr,momentum=a.momentum,weight_decay=a.weight_decay);sch=MultiStepLR(opt,[int(x) for x in a.milestones.split(',')],gamma=a.gamma);scaler=torch.amp.GradScaler('cuda',enabled=a.amp);start=0;best=-1.
 if a.resume:
  ck=torch.load(a.resume,map_location='cpu');model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);sch.load_state_dict(ck['scheduler']);scaler.load_state_dict(ck['scaler']);start=ck['epoch']+1;best=ck.get('best_macro_f1',-1.)
 hist=a.output_dir/'metrics.jsonl'
 for ep in range(start,a.epochs):
  model.train();ls=0.;steps=0;samples=0;t0=time.perf_counter();last=t0;datawait=0.;int0=t0;intsamp=0
  for bi,(x,y) in enumerate(trl):
   if a.max_train_steps and bi>=a.max_train_steps:break
   ready=time.perf_counter();datawait+=ready-last;profile=(bi==0 or (bi+1)%a.log_every==0)
   if profile:torch.cuda.synchronize();gp0=time.perf_counter()
   x=to_gpu_image(x,dev);y=y.to(dev,non_blocking=True);opt.zero_grad(set_to_none=True)
   with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=a.amp):loss=lossfn(model(x),y,w)
   if not torch.isfinite(loss):raise RuntimeError(f'non-finite loss ep={ep} step={bi}')
   scaler.scale(loss).backward();scaler.step(opt);scaler.update()
   if profile:torch.cuda.synchronize();gpms=(time.perf_counter()-gp0)*1000
   else:gpms=float('nan')
   bs=x.shape[0];ls+=float(loss.item());steps+=1;samples+=bs;intsamp+=bs;last=time.perf_counter()
   if profile:
    now=time.perf_counter();rate=intsamp/max(now-int0,1e-9);eta=max(len(trl)-bi-1,0)*(now-t0)/max(bi+1,1)/60;print(f'EPOCH {ep:02d}/{a.epochs-1:02d} STEP {bi+1}/{len(trl)} loss={ls/steps:.5f} samples_s={rate:.1f} data_wait_ms_avg={1000*datawait/steps:.1f} gpu_compute_ms_profiled={gpms:.1f} eta_epoch_min={eta:.1f} lr={opt.param_groups[0]["lr"]:.3e}',flush=True);int0=now;intsamp=0
  sch.step();g,s=evaluate(model,vdl,dev,a.amp,a.max_eval_batches);rows,sm=metric(g,s,np.full(8,.5));sec=time.perf_counter()-t0;rec={'epoch':ep,'train_loss':ls/max(steps,1),'seconds':sec,'samples':samples,'samples_per_sec':samples/max(sec,1e-9),'avg_data_wait_ms':1000*datawait/max(steps,1),**sm};print('VAL',json.dumps(rec),flush=True)
  with hist.open('a',encoding='utf-8') as f:f.write(json.dumps(rec)+'\n')
  state={'epoch':ep,'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sch.state_dict(),'scaler':scaler.state_dict(),'best_macro_f1':best,'labels':LABELS,'input_hw':[360,640],'fast_v2':True};torch.save(state,a.output_dir/'last_train_state.pth')
  if sm['macro_f1']>best:best=sm['macro_f1'];state['best_macro_f1']=best;torch.save(state,a.output_dir/'best_train_state.pth');savecsv(a.output_dir/'best_val_per_class_0p5.csv',rows)
 ck=torch.load(a.output_dir/'best_train_state.pth',map_location='cpu');model.load_state_dict(ck['model']);gv,sv=evaluate(model,vdl,dev,a.amp,a.max_eval_batches);th=calibrate(gv,sv);gt,st=evaluate(model,tdl,dev,a.amp,a.max_eval_batches);rows,sm=metric(gt,st,th);savecsv(a.output_dir/'test_per_class_calibrated.csv',rows);(a.output_dir/'test_summary.json').write_text(json.dumps(sm,indent=2));(a.output_dir/'thresholds.json').write_text(json.dumps(dict(zip(LABELS,map(float,th))),indent=2));torch.save({'model':model.state_dict(),'labels':LABELS,'thresholds':dict(zip(LABELS,map(float,th))),'input_hw':[360,640],'architecture':'original UltraFace Mb_Tiny backbone + AdaptiveAvgPool2d + Linear(256,8)','fast_v2':True},a.output_dir/'best_ultraface_slim_multilabel8_640x360_fast_v2.pth');print('TEST',json.dumps(sm),flush=True)
if __name__=='__main__':main()
