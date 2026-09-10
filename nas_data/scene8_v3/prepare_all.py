#!/usr/bin/env python3
"""Offline NAS8 source curation v3. No model inference or training, no source writes."""
from __future__ import annotations
import argparse,csv,json,os,sys,traceback
from collections import Counter,defaultdict
from pathlib import Path
from policy import *
from storage import *
from sources import legacy_rows,mir_rows,nus_rows
from curate import *
from visualize import export_review,review_template

BASE=Path('/data/pub1/z00919662')
OLD=BASE/'dataset/UltraFaceSlim_8scene_multilabel_manifests_640x360_v1'


def apply_reviews(rows,path,audit):
    if path is None:return
    lookup=defaultdict(list)
    for r in rows:lookup[r['image']].append(r)
    count=0;seen=set()
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r.get('reviewed')!='1':continue
            if not r.get('reviewer','').strip():raise Blocked('reviewer missing in approved override')
            image=str(Path(r['image']).resolve())
            if image in seen:raise Blocked('Duplicate approved review row: '+image)
            seen.add(image)
            if image not in lookup:raise Blocked('Approved review image not in source inventory: '+image)
            for dst in lookup[image]:
                changed=False
                for l in LABELS:
                    value=r.get(l,'').strip()
                    if not value:continue
                    if value not in ('-1','0','1'):raise Blocked('Invalid reviewed label '+value)
                    assign(dst['labels'],dst['evidence'],l,int(value),'human_review:'+r['reviewer']+':'+r.get('note',''))
                    changed=True
                if changed:dst.pop('force_review',None)
            count+=1
    audit['human_overrides']={'path':str(path),'sha256':sha(path),'approved_images':count}


def write_manifest(p,rows):
    jsonl(p,rows)
    with p.with_suffix('.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['image','source','split','group_id','width','height']+LABELS)
        for r in rows:w.writerow([r[k] for k in ['image','source','split','group_id','width','height']]+[r['labels'][l] for l in LABELS])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-root',type=Path,default=OLD)
    p.add_argument('--mir-images',type=Path,default=BASE/'dataset/mirflickr25k.zip')
    p.add_argument('--mir-annotations',type=Path,default=BASE/'dataset/mirflickr25k_annotations_v080.zip')
    p.add_argument('--nus-archive',type=Path,default=BASE/'dataset/archive.zip')
    p.add_argument('--nus-root',type=Path,help='Reuse already extracted NUS root instead of archive; read-only')
    p.add_argument('--nus-metadata-root',type=Path,help='Separate official ImageList, Concepts81 and Groundtruth root; never retrieval index guesses')
    p.add_argument('--cache-root',type=Path,default=BASE/'dataset/NAS8_new_sources_raw_v3')
    p.add_argument('--output-root',type=Path,default=BASE/'dataset/NAS8_multilabel_clean_v3')
    p.add_argument('--overrides',type=Path,help='Human-reviewed CSV, generated template supported')
    p.add_argument('--seed',type=int,default=20260910)
    p.add_argument('--inspect-only',action='store_true',help='Inventory local archives and legacy manifests only; no extraction')
    p.add_argument('--source-roots',nargs=4,type=Path,default=[BASE/'segmentation/datasets/coco',BASE/'segmentation/datasets/places365',BASE/'segmentation/datasets/COCO_ADE_13cls_16x9_640x360',BASE/'dataset/10_scenes'])
    a=p.parse_args();out=a.output_root.resolve();cache=a.cache_root.resolve()
    protected=[a.old_root.resolve()]+[x.resolve() for x in a.source_roots]
    # A reused NUS source may be cache/nus. The cache can contain that source,
    # but must not be created inside the read-only NUS source itself.
    for extra in (a.nus_root,a.nus_metadata_root):
        if extra is None:continue
        v=extra.resolve()
        if out==v or v in out.parents or out in v.parents or cache==v or v in cache.parents:
            raise Blocked('Output/cache overlaps a read-only NUS input')
    if any(out==v or v in out.parents or out in v.parents or cache==v or v in cache.parents or cache in v.parents for v in protected):raise Blocked('Output/cache overlaps a source or old manifest root')
    if out==cache or out in cache.parents or cache in out.parents:raise Blocked('Keep cache and derived output as sibling directories')
    if out.exists():raise Blocked('Output already exists, do not overwrite. Choose a new --output-root: '+str(out))
    out.mkdir(parents=True);inventory=out/'archive_inventory';inventory.mkdir();audit={};excluded=[]
    try:
        # Resolve the user's basename-only image ZIP without broad or ambiguous guessing.
        if not a.mir_images.is_file():
            fallback=BASE/'segmentation/datasets/MIRFLICKR25K/downloads/mirflickr25k.zip'
            if fallback.is_file():a.mir_images=fallback
            else:raise Blocked('MIR image ZIP missing; supply actual --mir-images path (do not redownload automatically)')
        for name,path in [('mir_images',a.mir_images),('mir_annotations',a.mir_annotations)]+([] if a.nus_root else [('nus',a.nus_archive)]):
            if not path.is_file():raise Blocked('Missing local ZIP: '+str(path))
            with zipfile.ZipFile(path) as z:
                members=safe_members(z)
                dump(inventory/(name+'_preflight.json'),{'path':str(path),'file_bytes':path.stat().st_size,'count':len(members),
                    'images':sum(PurePosixPath(i.filename).suffix.lower() in IMG for i in members),
                    'selected_expanded_bytes':sum(i.file_size for i in members if selected(i)),
                    'first_members':[i.filename for i in members[:100]],
                    'label_and_list_members':[i.filename for i in members if PurePosixPath(i.filename).suffix.lower() in ('.txt','.csv','.json','.zip')][:300]})
        if a.inspect_only:
            dump(out/'summary.json',{'status':'INSPECTED_ONLY','next':'run preparation with a NEW output root; no files extracted'});return
        mirimg=extract(a.mir_images,cache/'mir_images',inventory,'a23d0a8564ee84cda5622a6c2f947785')
        mirann=extract(a.mir_annotations,cache/'mir_annotations',inventory)
        nus=a.nus_root.resolve() if a.nus_root else extract(a.nus_archive,cache/'nus',inventory)
        if not a.nus_root:expand_nested(nus,inventory)
        print('STAGE: parse MIRFLICKR manual labels',flush=True)
        rows=mir_rows(mirimg,mirann,a.seed,audit,excluded)
        print('STAGE: align NUS manual labels with image lists',flush=True)
        from source_preflight import export_nus_diagnostics
        export_nus_diagnostics(nus,out/'nus_diagnostics')
        rows+=nus_rows(nus,a.seed,audit,excluded,metadata_root=a.nus_metadata_root)
        print('STAGE: rebuild legacy labels (not copying old labels)',flush=True)
        rows+=legacy_rows(a.old_root,Path(__file__).with_name('places365_io.txt'),a.source_roots,audit,excluded)
        for s in sorted({r['source'] for r in rows}):print('SOURCE',s,sum(r['source']==s for r in rows),flush=True)
        apply_reviews(rows,a.overrides,audit)
        before=len(rows);rows=cap_candidates(rows,a.seed,excluded)
        print('STAGE: audit/decode selected sources',len(rows),'from',before,flush=True)
        rows=quality_scan(rows,excluded)
        if not rows:raise Blocked('No usable images')
        rows=group_and_split(rows,audit);balance_objective(rows,a.seed,audit)
        # Potential near-duplicate dHash groups: report, do NOT invent label merges.
        dh=defaultdict(list)
        for r in rows:dh[(r['dhash64'],r['aspect_bucket'])].append(r)
        suspects=[]
        for key,rs in dh.items():
            if len({r['split'] for r in rs})>1 and len({r['rgb_sha256'] for r in rs})>1:
                suspects.append({'signature':str(key),'examples':[{'image':r['image'],'split':r['split']} for r in rs[:10]],'count':len(rs)})
        dump(out/'near_duplicate_review.json',{'method':'same dHash64 + rounded aspect, candidate only','groups':suspects,'limits':'not exhaustive near duplicate search; false matches possible'})
        # Conservative protection: candidate near duplicate train groups are excluded, not relabeled.
        suspect_train={r['group_id'] for key,rs in dh.items() if len({r['split'] for r in rs})>1 and len({r['rgb_sha256'] for r in rs})>1 for r in rs if r['split']=='train'}
        for r in rows:
            if r['group_id'] in suspect_train and r['split']=='train':r['use_for_training_manifest']=False;r['exclusion_reason']='possible_cross_split_near_duplicate'
        counts,blocks,overlap=quality_gates(rows)
        for split in ('train','val','test'):
            selected_rows=[r for r in rows if r['split']==split and r['use_for_training_manifest']]
            write_manifest(out/(split+'.jsonl'),selected_rows)
            # Evaluation on native/reviewed labels only; category weak rules excluded.
            strict=[]
            for r in selected_rows:
                copy=dict(r);copy['labels']=r['labels'].copy()
                for l in LABELS:
                    ev=r['evidence'].get(l,'')
                    if not any(tag in ev for tag in ('manual:','human_review:','user_review:')):copy['labels'][l]=-1
                if any(v!=-1 for v in copy['labels'].values()):strict.append(copy)
            if split!='train':write_manifest(out/(split+'_strict.jsonl'),strict)
        # Keep all source observations, including rejected representatives, for source coverage/provenance.
        jsonl(out/'source_records.jsonl',rows);jsonl(out/'excluded.jsonl',excluded)
        for r in rows:
            if not r['use_for_training_manifest']:
                excluded.append({'source':r['source'],'image':r['image'],'reason':r.get('exclusion_reason','review')})
        jsonl(out/'excluded.jsonl',excluded)
        legacy_changes=[]
        for r in rows:
            if 'legacy_labels' in r:
                delta={l:{'old':r['legacy_labels'].get(l,-1),'new':r['labels'][l]} for l in LABELS if r['legacy_labels'].get(l,-1)!=r['labels'][l]}
                if delta:legacy_changes.append({'image':r['image'],'source':r['source'],'changes':delta,'evidence':r['evidence']})
        jsonl(out/'legacy_label_changes.jsonl',legacy_changes)
        dump(out/'reported_examples.json',{name:[r for r in rows if Path(r['image']).name==name] for name in REPORTED})
        dump(out/'resolution_audit.json',resolution(rows))
        print('STAGE: render source-balanced GT100',flush=True)
        chosen=export_review(rows,out,a.seed);review_template(chosen,out/'review100_template.csv')
        weather=sorted([r for r in rows if r['labels']['rain_snow']==-1 and r['source'] in ('mirflickr','nuswide','places365','coco')],key=lambda r:priority(r,a.seed))
        # Candidate queue is balanced by split; these are NOT automatically labeled dry-weather images.
        queue=[]
        for s in ('train','val','test'):queue += [r for r in weather if r['split']==s][:100]
        review_template(queue,out/'weather_review_template.csv')
        dump(out/'source_audit.json',audit)
        for split,meta in audit['old_manifest_files'].items():
            if sha(meta['path'])!=meta['sha256']:raise Blocked('Old manifest changed externally during run: '+meta['path'])
        summary={'status':'PREPARED_REVIEW_REQUIRED','schema':'nas8_source_curation_v3_rev3','labels':LABELS,
            'nus_format':audit.get('NUS_WIDE',{}).get('format','native_official'),
            'nus_target_supervision':audit.get('NUS_WIDE',{}).get('target_supervision','native nighttime/sports/snow'),
            'source_observations':len(rows),'source_counts':dict(Counter(r['source'] for r in rows)),
            'splits':counts,'excluded_reasons':dict(Counter(r['reason'] for r in excluded)),
            'new_split_group_overlap':overlap,'training_quality_blockers':blocks,
            'READY_FOR_TRAINING':False,'HUMAN_ACTION_REQUIRED':True,
            'human_action':'Review GT100 (weak mappings remain) and fill weather_review_template with real rain/snow absence/presence. Do not set unknowns to zero. No training is started by this task.',
            'native_evaluation_warning':'val_strict/test_strict only score native or human/user-reviewed labels; absence of evaluation coverage must be reported, not set to 0.',
            'source_integrity':'Source images and old labels never opened for writing. Old manifest SHA256 rechecked. New archives only extracted to separate cache.',
            'old_checkpoint_warning':'New splits do not make a checkpoint trained on old data independently valid. Train a NEW model for new validation/test claims.',
            'resolution':'See resolution_audit.json; 640x360 not fixed; no image overwritten',
            'network_access':'none','training_started':False,'gt_visualizations':100}
        dump(out/'summary.json',summary);dump(out/'PREPARED.json',{'seed':a.seed,'source_inputs':audit.get('old_manifest_files',{})})
        print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    except Exception as e:
        dump(out/'source_audit.json',audit)
        dump(out/'BLOCKED.json',{'status':'BLOCKED','HUMAN_ACTION_REQUIRED':True,'error':str(e),'traceback':traceback.format_exc(),
            'action':'Return this report plus archive_inventory. Do not rewrite source files or invent NUS image/GT ordering. No network downloads or training permitted.'})
        raise

if __name__=='__main__':main()
