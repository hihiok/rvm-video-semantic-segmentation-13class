#!/usr/bin/env python3
"""Refine completed NAS8 manifests without rebuilding sources or changing splits."""
from __future__ import annotations
import argparse
import csv
import json
import traceback
from collections import Counter
from pathlib import Path
from policy import LABELS, ALIASES, OBJECT_HARD, USER_FIXES, norm
from storage import Blocked, read_jsonl, sha, dump, jsonl
from curate import count_labels, quality_gates, resolution, priority
from prepare_all import write_manifest
from visualize import export_review, review_template

SPLITS=('train','val','test')
DEFAULT_INPUT=Path('/mnt/ssd1/z00919662/datasets/NAS8_multilabel_clean_v3_rev7_20260916_103040')


def set_reviewed(r,label,value,reason,changes):
    before=r['labels'][label]
    if before==value:return
    previous=r['evidence'].get(label,'')
    r.setdefault('refinement_original',{})[label]={'value':before,'evidence':previous}
    r['labels'][label]=value;r['evidence'][label]=reason
    changes.append({'image':r['image'],'source':r['source'],'split':r['split'],
                    'label':label,'before':before,'after':value,'reason':reason,'previous_evidence':previous})


def apply_confirmed_fixes(rows):
    changes=[]
    for r in rows:
        if r['source']=='10_scenes' and norm(str(r['detail']).split('->')[0]) in ALIASES['night']:
            set_reviewed(r,'outdoor',1,'user_dataset:10_scenes_Night_dashcam_outdoor',changes)
        if r['source']=='mirflickr':
            for label,value in USER_FIXES.get(('mirflickr',Path(r['image']).name),{}).items():
                set_reviewed(r,label,value,'user_review:reported_specific_image_outdoor',changes)
    return changes


def active_counts(rows):
    return {s:count_labels([r for r in rows if r['split']==s and r['use_for_training_manifest']]) for s in SPLITS}


def cap_negatives(rows,seed):
    """Remove pure-negative records first; preserve every remaining positive label."""
    report={}
    for split in SPLITS:
        active=[r for r in rows if r['split']==split and r['use_for_training_manifest']]
        before=count_labels(active)
        protected=[(r,[l for l in LABELS if r['labels'][l]==1]) for r in active if 1 in r['labels'].values()]
        positives={l:before[l]['positive'] for l in LABELS}
        negative={l:before[l]['negative'] for l in LABELS}
        pure=[r for r in active if 1 not in r['labels'].values()]
        # Single-label pure negatives first, then multi-negative-only records.
        # Keep office/screen hard negatives longer within the same tier.
        pure.sort(key=lambda r:(sum(v==0 for v in r['labels'].values()),
                               r['source']=='places365' and r['detail'] in OBJECT_HARD,
                               priority(r,seed),r['sample_id']))
        removed=0;removed_by_label=Counter();masked=Counter()
        for r in pure:
            zeros=[l for l in LABELS if r['labels'][l]==0]
            if not zeros:
                r['use_for_training_manifest']=False;r['exclusion_reason']='no_reliable_supervision';continue
            if not any(negative[l]>positives[l] for l in zeros):continue
            r['use_for_training_manifest']=False
            r['exclusion_reason']='pure_negative_excess_removed'
            for l in zeros:negative[l]-=1;removed_by_label[l]+=1
            removed+=1
        # Remaining excess may be on useful positive multi-label images.
        # Mask ONLY the excessive zero; do not delete those images or flip zero to one.
        for label in LABELS:
            candidates=[r for r in active if r['use_for_training_manifest'] and r['labels'][label]==0]
            def retain_key(r):
                hard=label=='objective_image' and r['source']=='places365' and r['detail'] in OBJECT_HARD
                reviewed=r['evidence'].get(label,'').startswith(('human_review:','user_review:'))
                return (not reviewed,not hard,priority(r,seed),r['sample_id'])
            candidates.sort(key=retain_key)
            for r in candidates[positives[label]:]:
                r.setdefault('negative_sampling_original',{})[label]={'value':0,'evidence':r['evidence'].get(label,'')}
                r['labels'][label]=-1
                r['evidence'][label]='sampling:negative_cap_1_to_1;not_supervised_this_version'
                masked[label]+=1
        for r,labels in protected:
            if not r['use_for_training_manifest'] or any(r['labels'][l]!=1 for l in labels):
                raise Blocked('Sampling changed a positive record: '+r['image'])
        remaining=[r for r in active if r['use_for_training_manifest']]
        after=count_labels(remaining)
        for l in LABELS:
            if after[l]['positive']!=positives[l]:raise Blocked('Sampling lost a positive: '+split+':'+l)
            if after[l]['negative']>after[l]['positive']:raise Blocked('Negative cap failed: '+split+':'+l)
        report[split]={'before':before,'after':after,'pure_negative_rows_removed':removed,
                       'negative_labels_removed_with_rows':dict(removed_by_label),
                       'negative_labels_masked_on_retained_rows':dict(masked),
                       'records_before':len(active),'records_after':len(remaining),'positive_records_preserved':len(protected)}
    return report


def load_completed(root):
    if (root/'BLOCKED.json').exists():raise Blocked('Input has BLOCKED.json; use a completed version')
    required=['summary.json','PREPARED.json','source_audit.json']+[s+'.jsonl' for s in SPLITS]
    hashes={name:sha(root/name) for name in required}
    summary=json.loads((root/'summary.json').read_text())
    if summary.get('status')!='PREPARED_REVIEW_REQUIRED':raise Blocked('Input not PREPARED_REVIEW_REQUIRED')
    rows=[];groups=set();paths=set()
    for split in SPLITS:
        count=0
        for r in read_jsonl(root/(split+'.jsonl')):
            needed={'image','source','split','group_id','sample_id','labels','evidence','width','height','detail'}
            if not needed<=r.keys():raise Blocked('Missing row fields: '+str(needed-r.keys()))
            if r['split']!=split or not r.get('use_for_training_manifest',False):raise Blocked('Invalid active input row')
            if set(r['labels'])!=set(LABELS) or any(type(v)!=int or v not in (-1,0,1) for v in r['labels'].values()):
                raise Blocked('Invalid eight-label schema')
            p=Path(r['image']).resolve()
            if not p.is_file():raise Blocked('Missing input image: '+str(p))
            if r['group_id'] in groups or str(p) in paths:raise Blocked('Input duplicate group/image across manifests')
            groups.add(r['group_id']);paths.add(str(p))
            rows.append(r);count+=1
            if len(rows)%20000==0:print('INPUT_ROWS',len(rows),flush=True)
        if count!=summary['splits'][split]['records']:raise Blocked('Input manifest count disagrees with summary: '+split)
    return rows,hashes,summary


def strict_rows(rows):
    result=[]
    for r in rows:
        copy=dict(r);copy['labels']=r['labels'].copy()
        for l in LABELS:
            if not r['evidence'].get(l,'').startswith(('manual:','human_review:','user_review:')):copy['labels'][l]=-1
        if any(v!=-1 for v in copy['labels'].values()):result.append(copy)
    return result


def refine(root,out,seed):
    root=Path(root).resolve();out=Path(out).resolve()
    # New sibling only: no writes into input, source-image trees or old partial outputs.
    if out.parent!=root.parent or out==root or out.exists():raise Blocked('Choose a NEW sibling output directory')
    out.mkdir(mode=0o700,parents=False)
    audit={}
    try:
        rows,hashes,parent_summary=load_completed(root)
        before=active_counts(rows)
        changes=apply_confirmed_fixes(rows)
        after_corrections=active_counts(rows)
        sampling=cap_negatives(rows,seed)
        counts,blocks,overlap=quality_gates(rows)
        audit=json.loads((root/'source_audit.json').read_text())
        if 'objective_sampling' in audit:audit['parent_objective_sampling']=audit.pop('objective_sampling')
        audit['refinement']={'input_root':str(root),'input_sha256':hashes,'seed':seed,
            'scope':'only previously active manifests; previously excluded images are not reintroduced',
            'original_splits_preserved':True,'max_negative_to_positive':1,'by_split':sampling,
            'correction_count':len(changes),'input_counts':before,'after_corrections':after_corrections,
            'image_quality':'Reuse parent decoding/deduplication audit; check paths, do not rescan source archives',
            'evaluation_note':'Sampled labels; class ratios are not natural prevalence. Strict subset ratios may differ.'}
        for split in SPLITS:
            selected=[r for r in rows if r['split']==split and r['use_for_training_manifest']]
            write_manifest(out/(split+'.jsonl'),selected)
            if split!='train':write_manifest(out/(split+'_strict.jsonl'),strict_rows(selected))
        jsonl(out/'source_records.jsonl',rows)
        jsonl(out/'excluded.jsonl',({'image':r['image'],'source':r['source'],'split':r['split'],
             'reason':r['exclusion_reason'],'labels':r['labels']} for r in rows if not r['use_for_training_manifest']))
        jsonl(out/'label_corrections.jsonl',changes)
        jsonl(out/'negative_masks.jsonl',({'image':r['image'],'source':r['source'],'split':r['split'],
             'masked_labels':r['negative_sampling_original']} for r in rows if r.get('negative_sampling_original')))
        dump(out/'balance_audit.json',audit['refinement'])
        dump(out/'source_audit.json',audit)
        dump(out/'resolution_audit.json',resolution(rows))
        with (out/'class_balance.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.writer(f);writer.writerow(['split','label','input_positive','input_negative','corrected_positive',
                'corrected_negative','final_positive','final_negative','final_unknown','negative_to_positive'])
            for split in SPLITS:
                for l in LABELS:
                    b=before[split][l];c=after_corrections[split][l];a=counts[split]['labels'][l]
                    writer.writerow([split,l,b['positive'],b['negative'],c['positive'],c['negative'],
                        a['positive'],a['negative'],a['unknown'],a['negative']/a['positive'] if a['positive'] else 'NA'])
        print('STAGE: GT100 (includes marked excluded examples for source coverage)',flush=True)
        chosen=export_review(rows,out,seed);review_template(chosen,out/'review100_template.csv')
        weather=[]
        for split in SPLITS:
            candidates=[r for r in rows if r['split']==split and r['use_for_training_manifest']
                        and r['labels']['rain_snow']==-1 and r['source']!='10_scenes']
            weather+=sorted(candidates,key=lambda r:priority(r,seed))[:100]
        review_template(weather,out/'weather_review_template.csv')
        for name,h in hashes.items():
            if sha(root/name)!=h:raise Blocked('Input changed during refinement: '+name)
        summary={'status':'PREPARED_REVIEW_REQUIRED','schema':'nas8_refined_rev8','labels':LABELS,
                 'parent_root':str(root),'splits':counts,'new_split_group_overlap':overlap,
                 'negative_cap_passed':True,'max_negative_to_positive':1,'sampling_scope':'each of train/val/test, all eight labels',
                 'training_quality_blockers':blocks,'READY_FOR_TRAINING':False,'HUMAN_ACTION_REQUIRED':True,
                 'training_started':False,'gt_visualizations':100,
                 'human_action':'Review refreshed GT100; real-photo rain/snow negatives still need image review',
                 'source_observations':len(rows),'source_counts':dict(Counter(r['source'] for r in rows)),
                 'excluded_reasons':dict(Counter(r['exclusion_reason'] for r in rows if not r['use_for_training_manifest'])),
                 'label_correction_count':len(changes),'nus_format':parent_summary.get('nus_format'),
                 'evaluation_note':audit['refinement']['evaluation_note']}
        dump(out/'summary.json',summary)
        dump(out/'PREPARED.json',{'input_sha256':hashes,'seed':seed,'negative_cap_passed':True})
        print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
        return summary
    except Exception as e:
        dump(out/'BLOCKED.json',{'status':'BLOCKED','error':str(e),'traceback':traceback.format_exc()})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--input-root',type=Path,default=DEFAULT_INPUT)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20260910)
    a=parser.parse_args();refine(a.input_root,a.output_root,a.seed)

if __name__=='__main__':main()
