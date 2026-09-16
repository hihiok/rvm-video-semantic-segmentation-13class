"""Review real-photo weather negatives; never infer weather from scene labels."""
import argparse,csv,hashlib,html,json,traceback
from pathlib import Path
from collections import Counter
from refine_labels import load_completed,strict_rows,SPLITS
from policy import LABELS
from prepare_all import write_manifest
from storage import dump,jsonl,sha,Blocked
from curate import quality_gates

GROUPS=('night','indoor','office','sports')
def select(rows,limit):
    chosen=[]
    for split in SPLITS:
        pool=[r for r in rows if r['split']==split and r['labels']['rain_snow']==-1 and r['labels']['objective_image']!=1 and any(r['labels'][g]==1 for g in GROUPS)]
        pool.sort(key=lambda r:hashlib.sha256(r['image'].encode()).hexdigest())
        seen=set(); buckets=[[r for r in pool if r['labels'][g]==1] for g in GROUPS]
        while len(seen)<limit and any(buckets):
            for b in buckets:
                while b and b[-1]['image'] in seen:b.pop()
                if b and len(seen)<limit:
                    r=b.pop();seen.add(r['image']);chosen.append(r)
    return chosen

def apply(rows,reviews):
    lookup={(r['split'],r['image']):r for r in rows};seen=set();changes=[]
    for v in reviews:
        key=(v['split'],v['image'])
        if key in seen:raise Blocked('Duplicate review row')
        seen.add(key)
        if key not in lookup:raise Blocked('Review image/split not in input')
        if v.get('reviewed','') not in ('','0','1'):raise Blocked('Invalid reviewed value')
        if v.get('reviewed')!='1':continue
        r=lookup[key]
        if v.get('rain_absent')!='1' or v.get('snow_absent')!='1' or not v.get('reviewer','').strip():raise Blocked('Reviewed negative requires rain_absent=1 AND snow_absent=1 and reviewer')
        if sha(Path(r['image']))!=v.get('image_sha256'):raise Blocked('Reviewed image hash mismatch')
        if r['labels']['rain_snow']!=-1 or r['labels']['objective_image']==1 or not any(r['labels'][g]==1 for g in GROUPS):raise Blocked('Ineligible review; cannot overwrite weather positives or existing labels')
        r['labels']['rain_snow']=0;r['evidence']['rain_snow']='human_review:rain_absent_and_snow_absent'
        changes.append(dict(v))
    return changes

def run(root,out,review=None,limit=200):
    root=root.resolve();out=out.resolve()
    if out.exists() or out.parent!=root.parent:raise Blocked('Output must be a NEW sibling directory')
    rows,hashes,parent=load_completed(root)
    baseline={r['image']:dict(r['labels']) for r in rows}
    out.mkdir(mode=0o700)
    try:
        removed=[]
        for r in rows:
            if r['labels']['objective_image']==1 and r['labels']['rain_snow']==0:
                removed.append({'image':r['image'],'split':r['split'],'before':0,'after':-1,'previous_evidence':r['evidence'].get('rain_snow')})
                r['labels']['rain_snow']=-1;r['evidence']['rain_snow']='sampling:exclude_synthetic_weather_negative'
        changes=[]
        if review:
            with review.open(encoding='utf-8-sig',newline='') as f:changes=apply(rows,list(csv.DictReader(f)))
        for r in rows:
            old=baseline[r['image']]
            for label in LABELS:
                if (label!='rain_snow' or old[label]==1) and r['labels'][label]!=old[label]:raise Blocked('Protected label changed')
        counts,blocks,overlap=quality_gates(rows)
        for split in SPLITS:
            selected=[r for r in rows if r['split']==split]
            write_manifest(out/(split+'.jsonl'),selected)
            if split!='train':write_manifest(out/(split+'_strict.jsonl'),strict_rows(selected))
        candidates=select(rows,limit)
        fields=['split','image','image_sha256','candidate_classes','reviewed','rain_absent','snow_absent','reviewer','notes']
        with (out/'weather_review_template.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for r in candidates:w.writerow({'split':r['split'],'image':r['image'],'image_sha256':sha(Path(r['image'])),'candidate_classes':','.join(g for g in GROUPS if r['labels'][g]==1)})
        from PIL import Image,ImageOps
        thumbs=out/'weather_review';thumbs.mkdir()
        cards=[]
        for i,r in enumerate(candidates):
            with Image.open(r['image']) as im:
                im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((640,640));im.save(thumbs/('%04d.jpg'%i))
            cards.append('<div><img loading="lazy" src="%04d.jpg"><p>%s</p><p>%s</p></div>'%(i,html.escape(r['split']),html.escape(r['image'])))
        (thumbs/'index.html').write_text('<meta charset="utf-8"><h1>Weather review: confirm BOTH no rain and no snow</h1>'+''.join(cards),encoding='utf-8')
        jsonl(out/'source_records.jsonl',rows);jsonl(out/'weather_removed_synthetic.jsonl',removed);jsonl(out/'weather_applied_reviews.jsonl',changes)
        audit={'parent_root':str(root),'input_sha256':hashes,'synthetic_negative_labels_removed':len(removed),'real_negative_reviews_applied':len(changes),'candidate_counts':dict(Counter(r['split'] for r in candidates)),'all_records_and_positive_labels_preserved':True,'other_labels_unchanged':True,'review_file_sha256':sha(review) if review else None}
        dump(out/'weather_audit.json',audit)
        source=json.loads((root/'source_audit.json').read_text());source['weather_refinement']=audit;dump(out/'source_audit.json',source)
        summary=dict(parent);summary.update(schema='nas8_weather_rev9',parent_root=str(root),splits=counts,new_split_group_overlap=overlap,training_quality_blockers=blocks,READY_FOR_TRAINING=False,HUMAN_ACTION_REQUIRED=True,training_started=False,gt_visualizations=0,human_action='Review weather_review/index.html and fill CSV; run again with --reviews into a new sibling. No training.')
        summary.pop('negative_cap_passed',None);summary.pop('max_negative_to_positive',None)
        summary['weather_refinement']=audit
        for name,h in hashes.items():
            if sha(root/name)!=h:raise Blocked('Input changed')
        dump(out/'summary.json',summary);dump(out/'PREPARED.json',audit)
        print(json.dumps(audit,ensure_ascii=False,indent=2))
    except Exception as e:
        dump(out/'BLOCKED.json',{'error':str(e),'traceback':traceback.format_exc()});raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--reviews',type=Path);p.add_argument('--candidates-per-split',type=int,default=200)
    a=p.parse_args()
    if a.candidates_per_split<1:p.error('candidate count must be positive')
    run(a.input_root,a.output_root,a.reviews,a.candidates_per_split)
