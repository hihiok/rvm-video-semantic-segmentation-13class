#!/usr/bin/env python3
"""Create a smaller, deduplicated training manifest without touching source data.

Policy for TRAIN only:
- keep all 10_scenes records;
- keep all sampled COCO objective-image negatives already present;
- cap Places365 to N images per canonical scene category;
- keep every SEG13 record with rain_snow=1 or landscape=1;
- sample low-value SEG13 negative-only records to max(min_neg, ratio * positive_records);
- merge records that identify the same underlying image and union compatible partial labels.

VAL/TEST are not distribution-subsampled; only duplicate underlying images are merged.
Conflicting known labels for the same underlying image are changed to unknown (-1) and audited.
"""
from __future__ import annotations
import argparse, hashlib, json, random, re
from collections import defaultdict
from pathlib import Path

LABELS = ["night","indoor","rain_snow","office","outdoor","landscape","sports","objective_image"]


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--input-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--places-cap-per-category',type=int,default=250)
    p.add_argument('--seg-negative-ratio',type=float,default=4.0)
    p.add_argument('--seg-negative-min',type=int,default=5000)
    p.add_argument('--seed',type=int,default=20260907)
    return p.parse_args()


def load(p):
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]


def underlying_id(r):
    p=Path(r['image']); stem=p.stem.lower()
    m=re.search(r'(?<!\d)(\d{12})(?!\d)',stem)
    if m: return 'coco:'+m.group(1)
    return 'path:'+str(p.resolve())


def source_priority(r):
    s=r.get('source','')
    # Prefer already prepared 640x360 SEG13 derivatives when duplicate paths exist.
    return {'seg13':0,'10_scenes':1,'places365':2,'coco2017':3}.get(s,9)


def sample_train(rows,rng,a):
    places=defaultdict(list); seg_pos=[]; seg_neg=[]; other=[]
    for r in rows:
        s=r.get('source','')
        if s=='places365': places[r.get('detail','unknown')].append(r)
        elif s=='seg13':
            y=r['labels']
            if int(y.get('rain_snow',-1))==1 or int(y.get('landscape',-1))==1: seg_pos.append(r)
            else: seg_neg.append(r)
        else: other.append(r)
    kept_places=[]
    for _,items in sorted(places.items()):
        if len(items)>a.places_cap_per_category: items=rng.sample(items,a.places_cap_per_category)
        kept_places.extend(items)
    neg_cap=max(a.seg_negative_min,int(round(a.seg_negative_ratio*max(len(seg_pos),1))))
    if len(seg_neg)>neg_cap: seg_neg=rng.sample(seg_neg,neg_cap)
    kept=other+kept_places+seg_pos+seg_neg
    rng.shuffle(kept)
    stats={'input':len(rows),'places_kept':len(kept_places),'seg_positive_kept':len(seg_pos),'seg_negative_kept':len(seg_neg),'other_kept':len(other),'before_merge':len(kept)}
    return kept,stats


def merge_rows(rows):
    groups=defaultdict(list)
    for r in rows: groups[underlying_id(r)].append(r)
    out=[]; conflicts=defaultdict(int); merged_groups=0
    for uid,items in groups.items():
        if len(items)>1: merged_groups+=1
        items=sorted(items,key=source_priority)
        base=dict(items[0]); labels={k:-1 for k in LABELS}; sources=[]; details=[]
        for r in items:
            sources.append(r.get('source','unknown')); details.append(r.get('detail',''))
            for k in LABELS:
                v=int(r['labels'].get(k,-1))
                if v<0: continue
                if labels[k]<0: labels[k]=v
                elif labels[k]!=v:
                    labels[k]=-1; conflicts[k]+=1
        base['labels']=labels
        base['source']='+'.join(sorted(set(sources)))
        base['detail']='MERGED:'+uid+';sources='+','.join(sorted(set(sources)))
        out.append(base)
    return out,dict(conflicts),merged_groups


def summarize(rows):
    d={'records':len(rows),'labels':{k:{'pos':0,'neg':0,'unknown':0} for k in LABELS},'sources':defaultdict(int)}
    for r in rows:
        d['sources'][r.get('source','unknown')]+=1
        for k in LABELS:
            v=int(r['labels'][k]); d['labels'][k]['pos' if v==1 else 'neg' if v==0 else 'unknown']+=1
    d['sources']=dict(d['sources']); return d


def main():
    a=parse_args(); rng=random.Random(a.seed); a.output_root.mkdir(parents=True,exist_ok=True)
    report={'policy':{'places_cap_per_category':a.places_cap_per_category,'seg_negative_ratio':a.seg_negative_ratio,'seg_negative_min':a.seg_negative_min},'splits':{}}
    for split in ('train','val','test'):
        rows=load(a.input_root/f'{split}.jsonl'); sample_stats=None
        if split=='train': rows,sample_stats=sample_train(rows,rng,a)
        merged,conflicts,ng=merge_rows(rows); rng.shuffle(merged)
        with (a.output_root/f'{split}.jsonl').open('w',encoding='utf-8') as f:
            for r in merged: f.write(json.dumps(r,ensure_ascii=False)+'\n')
        report['splits'][split]={'sampling':sample_stats,'merged_duplicate_groups':ng,'label_conflicts_to_unknown':conflicts,'summary':summarize(merged)}
    (a.output_root/'fast_v2_summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
