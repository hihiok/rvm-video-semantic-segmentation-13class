#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, random, shutil
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

ADE_HILL_ID = 69
TARGET_MOUNTAIN_ID = 12
TARGET_CLASSES = {0:'background',1:'sky',2:'person',3:'plant',4:'building',5:'flower',6:'food',7:'water',8:'desert',9:'ice_or_snow',10:'text',11:'ball',12:'mountain'}


def args():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--ade-root',type=Path,required=True)
    p.add_argument('--train-ratio',type=float,default=0.9)
    p.add_argument('--seed',type=int,default=20260730)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--recover-missing',action='store_true')
    p.add_argument('--backup-dir',type=Path,default=None)
    return p.parse_args()


def load(p):
    a=np.asarray(Image.open(p))
    if a.ndim!=2: raise ValueError(f'non-single-channel mask: {p} {a.shape}')
    return a.astype(np.uint8,copy=False)


def mask_root(root):
    for n in ('masks','annotations'):
        x=root/n
        if (x/'train').is_dir() and (x/'val').is_dir(): return x
    raise FileNotFoundError('need masks|annotations/train,val')


def split_map(ade_root, ratio, seed):
    fs=sorted(p for p in (ade_root/'images'/'training').iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'})
    idx=list(range(len(fs))); random.Random(seed).shuffle(idx)
    n=int(round(len(idx)*ratio)); tr=set(idx[:n])
    return {p.stem:('train' if i in tr else 'val') for i,p in enumerate(fs)}


def find_img(d, stem):
    xs=[p for p in d.glob(stem+'.*') if p.suffix.lower() in {'.jpg','.jpeg','.png'}]
    if len(xs)!=1: raise RuntimeError(f'image lookup {stem}: {xs}')
    return xs[0]


def backup(src, backup_root, dataset_root):
    if not backup_root or (not src.exists() and not src.is_symlink()): return
    dst=backup_root/src.relative_to(dataset_root); dst.parent.mkdir(parents=True,exist_ok=True)
    if src.is_symlink():
        if dst.exists() or dst.is_symlink(): dst.unlink()
        dst.symlink_to(src.resolve())
    else: shutil.copy2(src,dst)


def read_mapping(root):
    p=root/'class_mapping.json'
    d=json.loads(p.read_text()) if p.is_file() else {}
    raw=d.get('active_ade_id_to_target_id',{})
    m={int(k):int(v) for k,v in raw.items()}
    if not m: raise RuntimeError('class_mapping.json active mapping required')
    if m.get(17)!=12: raise RuntimeError(f'expected ADE mountain 17->12, got {m.get(17)}')
    return d,m


def lut_from(mapping):
    lut=np.zeros(256,dtype=np.uint8)
    for s,t in mapping.items(): lut[s]=t
    lut[ADE_HILL_ID]=TARGET_MOUNTAIN_ID
    return lut


def update_stats(root,mroot):
    out={'splits':{}}
    for split in ('train','val'):
        pix=Counter(); imgs=Counter(); n=0
        for p in tqdm(sorted((mroot/split).glob('*.png')),desc='stats '+split):
            a=load(p); u,c=np.unique(a,return_counts=True)
            for k,v in zip(u,c): pix[int(k)]+=int(v); imgs[int(k)]+=1
            n+=1
        tot=sum(pix.values())
        out['splits'][split]={'kept_images':n,'classes':{str(i):{'name':TARGET_CLASSES[i],'pixels':int(pix[i]),'pixel_ratio':pix[i]/tot if tot else 0.0,'images':int(imgs[i]),'image_ratio':imgs[i]/n if n else 0.0} for i in TARGET_CLASSES}}
    return out


def main():
    a=args(); root=a.dataset_root.resolve(); ade=a.ade_root.resolve(); mr=mask_root(root)
    bd=a.backup_dir.resolve() if a.backup_dir else None
    if bd and not a.dry_run: bd.mkdir(parents=True,exist_ok=True)
    meta,mapping=read_mapping(root); lut=lut_from(mapping); smap=split_map(ade,a.train_ratio,a.seed)
    existing={s:{p.stem for p in (mr/s).glob('*.png')} for s in ('train','val')}
    report={'ade_hill_id':69,'target_mountain_id':12,'dry_run':a.dry_run,'source_hill_images':{'training':0,'validation':0},'existing_patched':{'train':0,'val':0},'missing':{'train':0,'val':0},'recovered':{'train':0,'val':0},'hill_pixels_added':{'train':0,'val':0}}

    sources=[]
    for ade_split in ('training','validation'):
        for p in tqdm(sorted((ade/'annotations'/ade_split).glob('*.png')),desc='scan '+ade_split):
            src=load(p)
            if np.any(src==ADE_HILL_ID):
                report['source_hill_images'][ade_split]+=1; sources.append((ade_split,p,src))

    for ade_split,p,src in tqdm(sources,desc='patch/recover hill'):
        split=smap[p.stem] if ade_split=='training' else 'val'
        hill=(src==ADE_HILL_ID); hp=int(hill.sum()); dstm=mr/split/p.name
        if p.stem in existing[split]:
            cur=load(dstm)
            if cur.shape!=src.shape: raise RuntimeError(f'shape mismatch {dstm}')
            conflict=hill & ~np.isin(cur,[0,12])
            if conflict.any(): raise RuntimeError(f'hill overlaps existing class 1..11: {dstm}')
            new=cur.copy(); new[hill]=12
            if not np.array_equal(cur[~hill],new[~hill]): raise RuntimeError('non-hill changed')
            if not np.array_equal(cur,new):
                report['existing_patched'][split]+=1; report['hill_pixels_added'][split]+=hp
                if not a.dry_run:
                    backup(dstm,bd,root); Image.fromarray(new,mode='L').save(dstm)
        else:
            report['missing'][split]+=1
            if a.recover_missing:
                new=lut[src]
                if not np.any(new==12): raise RuntimeError('recovered mask lost hill')
                report['recovered'][split]+=1; report['hill_pixels_added'][split]+=hp
                if not a.dry_run:
                    dstm.parent.mkdir(parents=True,exist_ok=True); Image.fromarray(new,mode='L').save(dstm)
                    src_img=find_img(ade/'images'/ade_split,p.stem); dsti=root/'images'/split/src_img.name; dsti.parent.mkdir(parents=True,exist_ok=True)
                    if dsti.exists() or dsti.is_symlink(): dsti.unlink()
                    dsti.symlink_to(src_img.resolve())

    if not a.dry_run:
        mapping[69]=12
        meta['num_classes_including_background']=13
        meta['target_classes']={str(k):v for k,v in TARGET_CLASSES.items()}
        meta['active_ade_id_to_target_id']={str(k):int(v) for k,v in sorted(mapping.items())}
        sg=meta.setdefault('source_groups',{})
        sg['12']={'target_name':'mountain','source_ids':[17,69],'source_names':['mountain/mount','hill']}
        meta['mountain_definition']='ADE IDs 17 mountain + 69 hill; rock/stone excluded'
        mp=root/'class_mapping.json'; backup(mp,bd,root); mp.write_text(json.dumps(meta,ensure_ascii=False,indent=2))
        cp=root/'classes.txt'; backup(cp,bd,root); cp.write_text(''.join(f'{i}\t{TARGET_CLASSES[i]}\n' for i in TARGET_CLASSES))
        st=update_stats(root,mr); (root/'split_stats_13class_mountain_hill.json').write_text(json.dumps(st,ensure_ascii=False,indent=2))
        with (root/'mountain_candidates.txt').open('w') as f:
            f.write('split,mask\n')
            for s in ('train','val'):
                for q in sorted((mr/s).glob('*.png')):
                    if np.any(load(q)==12): f.write(f'{s},{q}\n')

    rp=root/('hill_dry_run.json' if a.dry_run else 'hill_report.json'); rp.write_text(json.dumps(report,ensure_ascii=False,indent=2)); print(json.dumps(report,ensure_ascii=False,indent=2)); print(rp)

if __name__=='__main__': main()
