#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, shutil
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

TARGET_CLASSES={0:'background',1:'sky',2:'person',3:'plant',4:'building',5:'flower',6:'food',7:'water',8:'desert',9:'ice_or_snow',10:'text',11:'ball',12:'mountain'}
NEED_CLASS={
1:['sky-other','clouds'],
2:['person'],
3:['plant-other','straw','moss','branch','leaves','bush','tree','grass'],
4:['building-other','roof','tent','bridge','skyscraper','house'],
5:['flower'],
6:['sandwich','hot dog','pizza','donut','cake'],
7:['water-other','waterdrops','sea','river'],
8:['sand'],
9:['snow'],
10:['street sign','stop sign'],
11:['sports ball'],
12:['mountain','hill'],
}


def parse():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset-root',type=Path,required=True,help='already copied custom13 working dataset')
    p.add_argument('--raw-images-root',type=Path,required=True,help='contains train2017/ val2017')
    p.add_argument('--raw-annotations-root',type=Path,required=True,help='contains train2017/ val2017 stuffthingmaps')
    p.add_argument('--labels-file',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--recover-missing',action='store_true')
    return p.parse_args()


def load(p):
    a=np.asarray(Image.open(p))
    if a.ndim!=2: raise ValueError(f'non-single-channel: {p} {a.shape}')
    return a.astype(np.uint8,copy=False)


def labels(p):
    names=[]
    for line in p.read_text(encoding='utf-8').splitlines():
        line=line.strip()
        if not line: continue
        if ':' in line: names.append(line.split(':',1)[1].strip())
        else:
            parts=line.split(None,1); names.append(parts[1].strip() if len(parts)==2 and parts[0].isdigit() else line)
    return names


def build_mapping(names):
    m={}
    for tid,group in NEED_CLASS.items():
        for name in group:
            if name not in names: raise RuntimeError(f'missing COCO-Stuff label name: {name}')
            pix=names.index(name)-1
            if pix<0: raise RuntimeError(f'invalid pixel id for {name}: {pix}')
            if pix in m and m[pix]!=tid: raise RuntimeError(f'conflicting source pixel {pix}')
            m[pix]=tid
    return m


def maskroot(root):
    for n in ('masks','annotations'):
        x=root/n
        if (x/'train').is_dir() and (x/'val').is_dir(): return x
    raise FileNotFoundError('need masks|annotations/train,val')


def lut(mapping):
    x=np.zeros(256,dtype=np.uint8)
    for s,t in mapping.items(): x[s]=t
    return x


def find_img(root,rawsplit,stem):
    xs=[p for p in (root/rawsplit).glob(stem+'.*') if p.suffix.lower() in {'.jpg','.jpeg','.png'}]
    if len(xs)!=1: raise RuntimeError(f'image lookup {rawsplit}/{stem}: {xs}')
    return xs[0]


def stats(mr):
    out={'splits':{}}
    for split in ('train','val'):
        pix=Counter(); imgs=Counter(); n=0
        for p in tqdm(sorted((mr/split).glob('*.png')),desc='stats '+split):
            a=load(p); u,c=np.unique(a,return_counts=True)
            for k,v in zip(u,c): pix[int(k)]+=int(v); imgs[int(k)]+=1
            n+=1
        tot=sum(pix.values())
        out['splits'][split]={'kept_images':n,'classes':{str(i):{'name':TARGET_CLASSES[i],'pixels':int(pix[i]),'pixel_ratio':pix[i]/tot if tot else 0.0,'images':int(imgs[i]),'image_ratio':imgs[i]/n if n else 0.0} for i in TARGET_CLASSES}}
    return out


def main():
    a=parse(); root=a.dataset_root.resolve(); mr=maskroot(root); names=labels(a.labels_file.resolve()); mapping=build_mapping(names); L=lut(mapping)
    mountain_pix={names.index('mountain')-1,names.index('hill')-1}
    print('source pixel ids:',{'mountain':names.index('mountain')-1,'hill':names.index('hill')-1})
    existing={s:{p.stem for p in (mr/s).glob('*.png')} for s in ('train','val')}
    report={'dry_run':a.dry_run,'source_pixel_ids':{'mountain':names.index('mountain')-1,'hill':names.index('hill')-1},'source_images':{'train2017':0,'val2017':0},'existing_patched':{'train':0,'val':0},'missing':{'train':0,'val':0},'missing_zero_old_foreground':{'train':0,'val':0},'recovered':{'train':0,'val':0},'mountain_hill_pixels_added':{'train':0,'val':0}}
    oldmap={k:v for k,v in mapping.items() if v!=12}; oldL=lut(oldmap)

    for rawsplit,split in (('train2017','train'),('val2017','val')):
        adir=a.raw_annotations_root.resolve()/rawsplit
        for p in tqdm(sorted(adir.glob('*.png')),desc='scan '+rawsplit):
            src=load(p); mh=np.isin(src,list(mountain_pix))
            if not mh.any(): continue
            report['source_images'][rawsplit]+=1; hp=int(mh.sum()); dst=mr/split/p.name
            if p.stem in existing[split]:
                cur=load(dst)
                if cur.shape!=src.shape: raise RuntimeError(f'shape mismatch {dst}')
                conflict=mh & (cur!=0)
                if conflict.any():
                    vals,cnt=np.unique(cur[conflict],return_counts=True); raise RuntimeError(f'mountain/hill overlaps old target at {dst}: {dict(zip(vals.tolist(),cnt.tolist()))}')
                new=cur.copy(); new[mh]=12
                if not np.array_equal(cur[~mh],new[~mh]): raise RuntimeError('non mountain/hill pixels changed')
                if not np.array_equal(cur,new):
                    report['existing_patched'][split]+=1; report['mountain_hill_pixels_added'][split]+=hp
                    if not a.dry_run: Image.fromarray(new,mode='L').save(dst)
            else:
                report['missing'][split]+=1
                old=oldL[src]
                if not np.any(old): report['missing_zero_old_foreground'][split]+=1
                if a.recover_missing:
                    new=L[src]
                    if not np.any(new==12): raise RuntimeError('recovered mask lost mountain/hill')
                    report['recovered'][split]+=1; report['mountain_hill_pixels_added'][split]+=hp
                    if not a.dry_run:
                        dst.parent.mkdir(parents=True,exist_ok=True); Image.fromarray(new,mode='L').save(dst)
                        si=find_img(a.raw_images_root.resolve(),rawsplit,p.stem); di=root/'images'/split/si.name; di.parent.mkdir(parents=True,exist_ok=True)
                        if di.exists() or di.is_symlink(): di.unlink()
                        di.symlink_to(si.resolve())

    if not a.dry_run:
        (root/'classes.txt').write_text(''.join(f'{i}\t{TARGET_CLASSES[i]}\n' for i in TARGET_CLASSES),encoding='utf-8')
        meta={'num_classes_including_background':13,'target_classes':{str(k):v for k,v in TARGET_CLASSES.items()},'source':'COCO-Stuff stuffthingmaps','source_pixel_mapping':{str(k):int(v) for k,v in sorted(mapping.items())},'target_groups':{str(k):v for k,v in NEED_CLASS.items()},'mountain_definition':'COCO-Stuff mountain + hill; rock/gravel/stone excluded'}
        (root/'class_mapping.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
        st=stats(mr); (root/'split_stats_13class_mountain_hill.json').write_text(json.dumps(st,ensure_ascii=False,indent=2),encoding='utf-8')
        with (root/'mountain_candidates.txt').open('w') as f:
            f.write('split,mask\n')
            for s in ('train','val'):
                for q in sorted((mr/s).glob('*.png')):
                    if np.any(load(q)==12): f.write(f'{s},{q}\n')

    rp=root/('mountain_hill_dry_run.json' if a.dry_run else 'mountain_hill_report.json'); rp.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2)); print(rp)

if __name__=='__main__': main()
