"""Parse native manual GT and rebuild legacy source labels without trusting old y."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
from collections import Counter
from policy import *
from storage import *

def split_hash(key,seed):
    n=int(hashlib.sha256((str(seed)+':'+key).encode()).hexdigest()[:8],16)%100
    return 'train' if n<80 else ('val' if n<90 else 'test')

def mir_rows(image_root,ann_root,seed,audit,excluded,expected=25000):
    images={}
    for p in Path(image_root).rglob('*'):
        m=re.fullmatch(r'im([1-9][0-9]*)\.jpg',p.name,re.I)
        if p.is_file() and m:
            i=int(m[1])
            if i in images:raise Blocked('MIR duplicate im ID: '+str(i))
            images[i]=p
    if set(images)!=set(range(1,expected+1)):raise Blocked('MIR im1..im%d incomplete; found=%d'%(expected,len(images)))
    night=unique_file(ann_root,'night.txt');indoor=unique_file(ann_root,'indoor.txt')
    if night.parent!=indoor.parent:raise Blocked('MIR manual label directory mismatch')
    readmes=[p for p in Path(ann_root).rglob('*') if p.is_file() and p.name.lower().startswith('readme')]
    if not readmes:raise Blocked('MIR manual README missing')
    sets={};meta={}
    for p in sorted(night.parent.glob('*.txt')):
        if p.name.lower().startswith('readme'):continue
        ids=set()
        for n,s in enumerate(p.read_text(encoding='utf-8-sig').splitlines(),1):
            if not s.strip():continue
            if not s.strip().isdigit():raise Blocked('MIR format error %s:%d'%(p,n))
            v=int(s)
            if v not in images or v in ids:raise Blocked('MIR ID invalid or duplicate: '+str(p))
            ids.add(v)
        sets[p.stem.lower()]=ids;meta[p.name]={'path':str(p),'sha256':sha(p),'positives':len(ids)}
    differences={}
    for k,v in sets.items():
        if not k.endswith('_r1'):continue
        if k[:-3] not in sets:raise Blocked('MIR relevant list without potential list: '+k)
        extra=sorted(v-sets[k[:-3]])
        if extra:
            differences[k]={'count':len(extra),'ids':extra,'used_for_nas8':k[:-3] in ('night','indoor')}
            print('MIR WARNING: relevant IDs outside potential:',k,len(extra),'(audited; original lists retained)',flush=True)
    audit['MIRFLICKR']={'annotation_files':meta,'readme_files':[str(p) for p in readmes],
        'relevant_outside_potential':differences,
        'negative_policy':'outside BOTH native potential and relevant lists; explicit relevant positive takes precedence',
        'image_count':len(images)}
    rows=[]
    for i,p in sorted(images.items()):
        y,ev=map_mir(i,sets)
        rows.append(new_record(p,'mirflickr',split_hash('mir:'+str(i),seed),'manual_v080',y,ev,
            original_labels={k:int(i in s) for k,s in sets.items()},sample_id='mir:'+str(i)))
    return rows

def nus_rows(root,seed,audit,excluded,metadata_root=None):
    root=Path(root);index=ImageIndex(root)
    # Manual labels/image lists may be supplied separately from the photo mirror.
    # Scan metadata once, avoiding repeated 269k-image walks for each concept.
    metadata_root=Path(metadata_root) if metadata_root is not None else root
    files={}
    for path in metadata_root.rglob('*'):
        if path.is_file() and path.suffix.lower()=='.txt':files.setdefault(path.name.lower(),[]).append(path)
    def lookup(name,optional=False):
        candidates=files.get(name.lower(),[])
        if not candidates:
            if optional:return None
            raise Blocked('NUS missing '+name+' under '+str(metadata_root))
        if len(candidates)>1 and len({sha(x) for x in candidates})!=1:
            raise Blocked('NUS conflicting metadata copies: '+name)
        return sorted(candidates,key=lambda x:(len(x.parts),str(x)))[0]
    concepts=lookup('Concepts81.txt',True)
    if concepts is None:
        retrieval=any(k in files for k in ('database_label.txt','test_label.txt','targets_tc10.txt'))
        if 'database_label.txt' in files:
            from nus21 import top21_rows
            return top21_rows(index,lookup,seed,audit,excluded,split_hash)
        raise Blocked('NUS_RETRIEVAL_MAPPING_UNVERIFIED: numeric retrieval labels are not the official 81-concept GT. Return nus_diagnostics.zip for ChatGPT to verify README/class mapping; do not invent indices or negatives.' if retrieval else 'NUS official Concepts81.txt missing; return nus_diagnostics.zip')
    names=[s.strip() for s in concepts.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    if len(names)!=81 or len(set(names))!=81 or not {'nighttime','sports','snow'}<=set(names):raise Blocked('NUS Concepts81 invalid or wrong dataset')
    tr=lookup('TrainImagelist.txt',True);te=lookup('TestImagelist.txt',True)
    tasks=[]
    if tr is not None and te is not None:
        tasks=[(tr,'Train'),(te,'Test')]
    else:
        allp=lookup('Imagelist.txt',True)
        if allp is None:raise Blocked('NUS needs official TrainImagelist/TestImagelist or Imagelist plus Groundtruth. CSV tags or sorted JPG order are NOT accepted. See archive inventory.')
        tasks=[(allp,None)]
    rows=[];manifest_audit=[]
    for imgfile,split in tasks:
        paths=image_lines(imgfile);arrays={};meta={}
        # sports subclasses are diagnostic contradictions, not a negative-complete sports taxonomy.
        for k in ('nighttime','sports','snow','soccer','running','swimmers','surf'):
            fname='Labels_%s%s.txt'%(k,'_'+split if split else '')
            label=lookup(fname,k not in ('nighttime','sports','snow'))
            if label is None:continue
            vals=binary_lines(label)
            if len(vals)!=len(paths):raise Blocked('NUS row count mismatch %s=%d vs %s=%d'%(imgfile,len(paths),label,len(vals)))
            arrays[k]=vals;meta[k]={'path':str(label),'sha256':sha(label),'rows':len(vals)}
        found=0
        for i,name in enumerate(paths):
            p=index.resolve(name)
            if p is None:
                excluded.append({'source':'nuswide','image':name,'reason':'image_missing_keep_annotation_row_index','native_row':i});continue
            found+=1;gt={k:v[i] for k,v in arrays.items()};y,ev=map_nus(gt)
            if y['sports']==0 and any(gt.get(k)==1 for k in ('soccer','running','swimmers','surf')):
                y['sports']=-1;ev['sports']='native sports/subclass conflict;review'
            out='test' if split=='Test' else split_hash('nus:'+name,seed)
            if split=='Train' and out=='test':out='train' # native Train -> custom 90/10 Train/Val
            rows.append(new_record(p,'nuswide',out,'native_manual_labels',y,ev,original_labels=gt,
                sample_id='nus:'+name.replace('\\','/'),native_image_list=str(imgfile),native_row=i,native_name=name))
        if not found:raise Blocked('NUS no image paths resolved; archive may contain ONLY labels, not photos')
        manifest_audit.append({'image_list':str(imgfile),'sha256':sha(imgfile),'rows':len(paths),'available':found,'labels':meta})
    audit['NUS_WIDE']={'image_index_count':index.count,'lists':manifest_audit,'concepts_file':str(concepts),'concepts_sha256':sha(concepts),'order':'EXACT native image-list row, never sorted image directory','missing_rows_excluded':sum(x['source']=='nuswide' for x in excluded)}
    return rows

def places_map(path):
    io={};flat={}
    for s in Path(path).read_text().splitlines():
        if not s.strip() or s.startswith('#'):continue
        raw,value=s.rsplit(None,1);parts=raw.strip('/').split('/')
        cat='/'.join(parts[1:]) if len(parts[0])==1 else '/'.join(parts)
        if cat in io:raise Blocked('Duplicate Places category '+cat)
        io[cat]=int(value)
        if norm(cat) in flat and flat[norm(cat)]!=cat:raise Blocked('Places flat key collision')
        flat[norm(cat)]=cat
    if len(io)!=365:raise Blocked('Official Places IO map must contain 365 categories')
    return io,flat

def identify_places(r,io,flat):
    detail=str(r.get('detail','')).replace('Places365:','')
    candidates=[]
    parts=Path(r['image']).parts[:-1]
    for k in (1,2,3):
        key=norm('/'.join(parts[-k:]))
        if key in flat:candidates.append(flat[key])
    cat=max(candidates,key=len) if candidates else None
    from_detail=flat.get(norm(detail))
    if cat and from_detail and cat!=from_detail:raise Blocked('Places folder/detail disagreement: '+str(r['image']))
    cat=cat or from_detail
    if cat not in io:raise Blocked('Unknown Places category; do not guess: '+str(r))
    return cat

def legacy_rows(root,io_file,allowed_roots,audit,excluded):
    io,flat=places_map(io_file);rows=[];oldcounts=Counter();unmapped=Counter()
    audit['old_manifest_files']={}
    roots=[Path(p).resolve() for p in allowed_roots]
    audit['legacy_root_counts']={str(p):0 for p in roots}
    for split in ('train','val','test'):
        file=Path(root)/(split+'.jsonl')
        audit['old_manifest_files'][split]={'path':str(file),'sha256':sha(file)}
        for old in read_jsonl(file):
            p=Path(old['image']).resolve()
            matches=[a for a in roots if p==a or a in p.parents]
            if not matches:raise Blocked('Legacy image outside allowed source roots: '+str(p))
            dataset_root=max(matches,key=lambda a:len(a.parts))
            audit['legacy_root_counts'][str(dataset_root)]+=1
            src={'coco2017':'coco','coco':'coco','seg13':'seg13','places365':'places365','10_scenes':'10_scenes'}.get(old.get('source'))
            if src is None:raise Blocked('Unknown legacy source: '+str(old.get('source')))
            oldcounts[src]+=1;y=unknown();ev={};detail=str(old.get('detail',''));reason=None
            if src=='places365':
                detail=identify_places(old,io,flat);y,ev=map_places(detail,io[detail])
            elif src=='10_scenes':
                detail=detail.split('->')[0] if '->' in detail else p.parent.name
                y,ev,reason=map_ten(detail)
                if reason:unmapped[detail]+=1
            else:
                # Old pixel-area GT is discarded entirely, including ice_or_snow=0.
                assign(y,ev,'objective_image',0,'weak:COCO_ADE_photograph_negative')
            if src=='places365' and p.name=='Places365_val_00027091.jpg':
                # User observed cabin/cloud image. Do not invent weather or IO from its category.
                for k in ('night','indoor','outdoor','rain_snow','landscape','sports','office'):y[k]=-1;ev[k]='user_reported_bad_GT:requires_image_review'
            for k,v in USER_FIXES.get((src,p.name),{}).items():
                assign(y,ev,k,v,'user_review:reported_specific_image_content')
            if reason:
                excluded.append({'source':src,'image':str(p),'reason':reason,'detail':detail})
            row=new_record(p,src,split,detail,y,ev,legacy_labels=old.get('labels',{}),sample_id=src+':'+str(p),source_dataset_root=str(dataset_root))
            if reason:row['force_review']=True
            rows.append(row)
    audit['legacy_source_counts']=dict(oldcounts);audit['ten_scenes_unmapped_or_ambiguous']=dict(unmapped)
    return rows
