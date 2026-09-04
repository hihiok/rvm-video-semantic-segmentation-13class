#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math, os, random, time
from pathlib import Path
import cv2, numpy as np, torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset
from model import create_ultraface_slim_scene8

LABELS=["night","indoor","rain_snow","office","outdoor","landscape","sports","objective_image"]
DISPLAY={"night":"夜景","indoor":"室内","rain_snow":"雨/雪","office":"办公场景","outdoor":"户外","landscape":"风景","sports":"运动","objective_image":"客观图"}


def args_parse():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=200); p.add_argument('--batch-size',type=int,default=24); p.add_argument('--workers',type=int,default=4)
    p.add_argument('--lr',type=float,default=1e-2); p.add_argument('--momentum',type=float,default=.9); p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--milestones',default='95,150'); p.add_argument('--gamma',type=float,default=.1); p.add_argument('--dropout',type=float,default=.1)
    p.add_argument('--seed',type=int,default=20260904); p.add_argument('--cpu-threads',type=int,default=4); p.add_argument('--amp',action='store_true')
    p.add_argument('--resume',type=Path,default=None); p.add_argument('--max-train-steps',type=int,default=0); p.add_argument('--max-eval-batches',type=int,default=0)
    return p.parse_args()


def load_jsonl(p): return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]

class SceneDataset(Dataset):
    def __init__(self, rows, train=False): self.rows=rows; self.train=train
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]; im=cv2.imread(r['image'],cv2.IMREAD_COLOR)
        if im is None: raise RuntimeError('cannot read '+r['image'])
        im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        im=cv2.resize(im,(640,360),interpolation=cv2.INTER_LINEAR)
        if self.train and random.random()<.5: im=np.ascontiguousarray(im[:,::-1])
        if self.train:
            alpha=random.uniform(.9,1.1); beta=random.uniform(-10,10)
            im=np.clip(im.astype(np.float32)*alpha+beta,0,255)
        else: im=im.astype(np.float32)
        im=(im-127.0)/128.0
        x=torch.from_numpy(im.transpose(2,0,1)).float()
        y=torch.tensor([float(r['labels'][k]) for k in LABELS])
        return x,y


def pos_weight(rows):
    p=np.zeros(8); n=np.zeros(8)
    for r in rows:
        for j,k in enumerate(LABELS):
            v=int(r['labels'][k]); p[j]+=v==1; n[j]+=v==0
    w=np.clip(n/np.maximum(p,1),.5,8.0)
    return torch.tensor(w,dtype=torch.float32),p.astype(int),n.astype(int)

def loss_fn(logits,y,w):
    m=y>=0; t=y.clamp(0,1)
    raw=F.binary_cross_entropy_with_logits(logits,t,reduction='none',pos_weight=w)
    return (raw*m).sum()/m.sum().clamp_min(1)

def ap(y,s):
    if (y==1).sum()==0:return float('nan')
    o=np.argsort(-s); yy=y[o]; tp=np.cumsum(yy==1); fp=np.cumsum(yy==0); pr=tp/np.maximum(tp+fp,1)
    return float(pr[yy==1].sum()/max((yy==1).sum(),1))
def metrics(gt,sc,th):
    rows=[]
    for j,k in enumerate(LABELS):
        m=gt[:,j]>=0; y=gt[m,j].astype(int); s=sc[m,j]; pred=(s>=th[j]).astype(int)
        tp=((pred==1)&(y==1)).sum(); fp=((pred==1)&(y==0)).sum(); fn=((pred==0)&(y==1)).sum(); tn=((pred==0)&(y==0)).sum()
        pr=tp/max(tp+fp,1); rc=tp/max(tp+fn,1); f1=2*pr*rc/max(pr+rc,1e-12); ba=.5*(tp/max(tp+fn,1)+tn/max(tn+fp,1))
        rows.append({'label':k,'display_name':DISPLAY[k],'threshold':float(th[j]),'known':int(m.sum()),'positive':int((y==1).sum()),'negative':int((y==0).sum()),'precision':float(pr),'recall':float(rc),'f1':float(f1),'accuracy':float((tp+tn)/max(len(y),1)),'balanced_accuracy':float(ba),'ap':ap(y,s)})
    sm={'macro_f1':float(np.mean([r['f1'] for r in rows])),'macro_balanced_accuracy':float(np.mean([r['balanced_accuracy'] for r in rows])),'macro_ap':float(np.nanmean([r['ap'] for r in rows]))}
    return rows,sm

def evaluate(model,loader,device,amp,max_batches=0):
    model.eval(); G=[]; S=[]
    with torch.no_grad():
        for bi,(x,y) in enumerate(loader):
            if max_batches and bi>=max_batches: break
            x=x.to(device,non_blocking=True)
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=amp): z=model(x)
            G.append(y.numpy()); S.append(torch.sigmoid(z).float().cpu().numpy())
    return np.concatenate(G),np.concatenate(S)
def calibrate(gt,sc):
    out=np.full(8,.5,dtype=np.float32)
    for j in range(8):
        m=gt[:,j]>=0; y=gt[m,j].astype(int); s=sc[m,j]; best=(-1,.5)
        for t in np.linspace(.05,.95,91):
            p=(s>=t).astype(int); tp=((p==1)&(y==1)).sum(); fp=((p==1)&(y==0)).sum(); fn=((p==0)&(y==1)).sum(); pr=tp/max(tp+fp,1); rc=tp/max(tp+fn,1); f=2*pr*rc/max(pr+rc,1e-12)
            if f>best[0]: best=(f,float(t))
        out[j]=best[1]
    return out
def save_csv(p,rows):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    a=args_parse(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    os.environ.setdefault('OMP_NUM_THREADS',str(a.cpu_threads)); os.environ.setdefault('MKL_NUM_THREADS',str(a.cpu_threads)); torch.set_num_threads(a.cpu_threads)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    dev=torch.device('cuda:0'); a.output_dir.mkdir(parents=True,exist_ok=True)
    tr=load_jsonl(a.data_root/'train.jsonl'); va=load_jsonl(a.data_root/'val.jsonl'); te=load_jsonl(a.data_root/'test.jsonl')
    dlkw=dict(num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0)
    trl=DataLoader(SceneDataset(tr,True),a.batch_size,shuffle=True,drop_last=True,**dlkw); val=DataLoader(SceneDataset(va),a.batch_size,shuffle=False,**dlkw); test=DataLoader(SceneDataset(te),a.batch_size,shuffle=False,**dlkw)
    w,pc,nc=pos_weight(tr); w=w.to(dev); print('TRAIN_SUPERVISION',dict(zip(LABELS,[{'pos':int(pc[i]),'neg':int(nc[i]),'pos_weight':float(w[i])} for i in range(8)])))
    model=create_ultraface_slim_scene8(a.dropout).to(dev); opt=torch.optim.SGD(model.parameters(),lr=a.lr,momentum=a.momentum,weight_decay=a.weight_decay); sch=MultiStepLR(opt,[int(x) for x in a.milestones.split(',')],gamma=a.gamma); scaler=torch.amp.GradScaler('cuda',enabled=a.amp)
    start=0; best=-1.0
    if a.resume:
        ck=torch.load(a.resume,map_location='cpu'); model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); sch.load_state_dict(ck['scheduler']); scaler.load_state_dict(ck['scaler']); start=ck['epoch']+1; best=ck.get('best_macro_f1',-1.0)
    hist=a.output_dir/'metrics.jsonl'
    for ep in range(start,a.epochs):
        model.train(); ls=0.; ns=0; t0=time.time()
        for bi,(x,y) in enumerate(trl):
            if a.max_train_steps and bi>=a.max_train_steps: break
            x=x.to(dev,non_blocking=True); y=y.to(dev,non_blocking=True); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=a.amp): loss=loss_fn(model(x),y,w)
            if not torch.isfinite(loss): raise RuntimeError(f'non-finite loss ep={ep} step={bi}')
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); ls+=float(loss); ns+=1
        sch.step(); g,s=evaluate(model,val,dev,a.amp,a.max_eval_batches); rows,sm=metrics(g,s,np.full(8,.5)); rec={'epoch':ep,'train_loss':ls/max(ns,1),'seconds':time.time()-t0,**sm}; print('VAL',rec)
        with hist.open('a') as f:f.write(json.dumps(rec)+'\n')
        st={'epoch':ep,'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sch.state_dict(),'scaler':scaler.state_dict(),'best_macro_f1':best,'labels':LABELS,'input_hw':[360,640]}; torch.save(st,a.output_dir/'last_train_state.pth')
        if sm['macro_f1']>best: best=sm['macro_f1']; st['best_macro_f1']=best; torch.save(st,a.output_dir/'best_train_state.pth'); save_csv(a.output_dir/'best_val_per_class_0p5.csv',rows)
    ck=torch.load(a.output_dir/'best_train_state.pth',map_location='cpu'); model.load_state_dict(ck['model'])
    gv,sv=evaluate(model,val,dev,a.amp,a.max_eval_batches); th=calibrate(gv,sv); gt,st=evaluate(model,test,dev,a.amp,a.max_eval_batches); rows,sm=metrics(gt,st,th)
    save_csv(a.output_dir/'test_per_class_calibrated.csv',rows); (a.output_dir/'test_summary.json').write_text(json.dumps(sm,indent=2)); (a.output_dir/'thresholds.json').write_text(json.dumps(dict(zip(LABELS,map(float,th))),indent=2))
    deploy={'model':model.state_dict(),'labels':LABELS,'thresholds':dict(zip(LABELS,map(float,th))),'input_hw':[360,640],'architecture':'original UltraFace Mb_Tiny backbone + AdaptiveAvgPool2d + Linear(256,8)'}; torch.save(deploy,a.output_dir/'best_ultraface_slim_multilabel8_640x360.pth')
    print('TEST',sm)
if __name__=='__main__': main()
