"""Conservative exclusions, split-safe duplicate grouping and auditable dimensions."""
from __future__ import annotations
import hashlib,random
from collections import Counter,defaultdict
from pathlib import Path
from PIL import Image,ImageOps
from policy import LABELS,OFFICE,OBJECT_HARD,REPORTED
from storage import Blocked

def digest_text(s):return hashlib.sha256(s.encode()).hexdigest()

def priority(r,seed):return digest_text(str(seed)+r['sample_id'])

def cap_candidates(rows,seed,excluded):
    kept=[];buckets=defaultdict(list)
    for r in rows:
        if r['preferred_split']!='train' or Path(r['image']).name in REPORTED or r.get('force_review') or any(v.startswith('human_review:') for v in r['evidence'].values()):
            kept.append(r);continue
        s=r['source']
        if s=='places365':key=(s,r['detail']);cap=600 if r['detail'] in OBJECT_HARD else 250
        elif s=='seg13':key=(s,'photo_negatives');cap=2500
        elif s=='coco':key=(s,'photo_negatives');cap=5000
        elif s=='nuswide' and not any(v==1 for v in r['labels'].values()):key=(s,'negative_only');cap=30000
        else:kept.append(r);continue
        buckets[(key,cap)].append(r)
    for (key,cap),items in sorted(buckets.items()):
        items.sort(key=lambda r:priority(r,seed));kept.extend(items[:cap])
        for r in items[cap:]:excluded.append({'source':r['source'],'image':r['image'],'reason':'training_source_cap','detail':str(key)})
    return kept

def quality_scan(rows,excluded):
    cache={};good=[]
    for n,r in enumerate(rows,1):
        path=r['image']
        if path not in cache:
            try:
                with Image.open(path) as im:
                    orig=im.size;x=ImageOps.exif_transpose(im).convert('RGB');x.load();w,h=x.size
                    if min(w,h)<64:raise ValueError('short_edge_below_64')
                    pixels=x.tobytes();rgb=hashlib.sha256(('%dx%d:'%(w,h)).encode()+pixels).hexdigest()
                    # Exact 64-bit difference-hash matches are only a warning, not authoritative identity.
                    small=list(x.convert('L').resize((9,8)).getdata());bits=0
                    for yy in range(8):
                        for xx in range(8):bits=(bits<<1)|int(small[yy*9+xx]>small[yy*9+xx+1])
                    cache[path]={'width':w,'height':h,'raw_width':orig[0],'raw_height':orig[1],
                                 'rgb_sha256':rgb,'dhash64':'%016x'%bits,'aspect_bucket':round(w/h,2)}
            except (OSError,ValueError,Image.DecompressionBombError) as e:cache[path]={'error':str(e)}
        info=cache[path]
        if 'error' in info:
            excluded.append({'image':path,'source':r['source'],'reason':'unreadable_or_too_small','detail':info['error']});continue
        r.update(info);good.append(r)
        if n%2000==0:print('QUALITY',n,'/',len(rows),'unique_decoded',len(cache),flush=True)
    return good

class DSU:
    def __init__(self,n):self.p=list(range(n))
    def root(self,x):
        while self.p[x]!=x:self.p[x]=self.p[self.p[x]];x=self.p[x]
        return x
    def union(self,a,b):
        a=self.root(a);b=self.root(b)
        if a!=b:self.p[max(a,b)]=min(a,b)

def coco_id(r):
    # Never interpret NUS/Flickr/10_scenes numeric names as COCO IDs.
    if r['source'] not in ('coco','seg13'):return None
    import re
    m=re.fullmatch(r'(?:COCO_(?:train|val)201[47]_)?([0-9]{12})',Path(r['image']).stem)
    return 'coco:'+m[1] if m else None

def group_and_split(rows,audit):
    d=DSU(len(rows));keys={}
    for i,r in enumerate(rows):
        for key in ('pixel:'+r['rgb_sha256'],'path:'+r['image'],coco_id(r)):
            if key is None:continue
            if key in keys:d.union(i,keys[key])
            else:keys[key]=i
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[d.root(i)].append(r)
    reassigned=0;conflicts=[];duplicates=0
    for members in groups.values():
        # Evaluation wins. Never send an existing eval image into new training.
        dest=max((r['preferred_split'] for r in members),key=lambda s:{'train':0,'val':1,'test':2}[s])
        gid='group:'+min(r['rgb_sha256'] for r in members)
        pixel_groups=defaultdict(list)
        for r in members:
            reassigned+=r['preferred_split']!=dest;r['split']=dest;r['group_id']=gid
            pixel_groups[r['rgb_sha256']].append(r)
        # Merge annotations only when pixels are IDENTICAL, not merely COCO derivatives.
        for same in pixel_groups.values():
            y={};ev={}
            for l in LABELS:
                supplied=[r for r in same if r['labels'][l]!=-1]
                reviewed=[r for r in supplied if r['evidence'].get(l,'').startswith(('user_review:','human_review:'))]
                use=reviewed or supplied;values={r['labels'][l] for r in use}
                if len(values)>1:
                    y[l]=-1;ev[l]='conflict:identical_pixels_disagree'
                    conflicts.append({'group':gid,'label':l,'images':[r['image'] for r in same]})
                elif values:
                    y[l]=next(iter(values));ev[l]=' | '.join(sorted({r['evidence'].get(l,'') for r in use}))
                else:y[l]=-1
            for r in same:r['labels']=y.copy();r['evidence']=ev.copy()
        # One training record per known underlying group. Keep all source records for provenance/GT review.
        suitable=[r for r in members if not r.get('force_review') and any(v!=-1 for v in r['labels'].values())]
        representative=min(suitable,key=lambda r:(-sum(v==1 for v in r['labels'].values()),-sum(v!=-1 for v in r['labels'].values()),r['source']=='seg13',r['sample_id'])) if suitable else None
        for r in members:
            r['use_for_training_manifest']=r is representative
            if r is not representative:r['exclusion_reason']='duplicate_group_representative' if suitable else 'no_reliable_supervision'
        duplicates+=max(len(members)-1,0)
    audit['grouping']={'groups':len(groups),'duplicate_records_not_repeated':duplicates,'split_reassignments':reassigned,
        'policy':'exact pixels/path and source-qualified COCO ID; eval precedence test>val>train; labels only merged for identical pixels',
        'label_conflicts':conflicts,'limits':'Near duplicates and renamed/cropped photos without shared IDs are not guaranteed absent.'}
    return rows

def balance_objective(rows,seed,audit):
    active=[r for r in rows if r['use_for_training_manifest'] and r['split']=='train']
    p=[r for r in active if r['labels']['objective_image']==1];n=[r for r in active if r['labels']['objective_image']==0]
    target=min(len(n),max(1500,5*len(p)))
    hard=[r for r in n if r['source']=='places365' and r['detail'] in OBJECT_HARD]
    other=[r for r in n if r not in hard] if len(n)<500 else [r for r in n if not(r['source']=='places365' and r['detail'] in OBJECT_HARD)]
    hard.sort(key=lambda r:priority(r,seed));other.sort(key=lambda r:priority(r,seed))
    chosen=hard[:min(len(hard),target//2)];need=target-len(chosen)
    remaining=other+hard[len(chosen):];chosen+=remaining[:need];ids={id(r) for r in chosen}
    for r in n:
        if id(r) not in ids:
            r['labels']['objective_image']=-1;r['evidence']['objective_image']='sampling:photo_negative_not_supervised_this_version'
            if all(v==-1 for v in r['labels'].values()):r['use_for_training_manifest']=False;r['exclusion_reason']='objective_negative_only_not_selected'
    audit['objective_sampling']={'train_positive':len(p),'negative_before':len(n),'negative_kept':len(chosen),
        'hard_office_screen_negative_kept':sum(r['source']=='places365' and r['detail'] in OBJECT_HARD for r in chosen),'max_negative_to_positive':5,'minimum_photo_negatives':1500}

def count_labels(rows):
    return {l:{k:sum(r['labels'][l]==v for r in rows) for k,v in [('positive',1),('negative',0),('unknown',-1)]} for l in LABELS}

def quality_gates(rows):
    used=[r for r in rows if r['use_for_training_manifest']];result={};block=[]
    for s in ('train','val','test'):
        rr=[r for r in used if r['split']==s];cnt=count_labels(rr)
        result[s]={'records':len(rr),'by_source':dict(Counter(r['source'] for r in rr)),'labels':cnt}
        for l in LABELS:
            if not cnt[l]['positive'] or not cnt[l]['negative']:block.append('%s:%s missing positive/negative'%(s,l))
        for l in ('night','rain_snow'):
            pos=sum(r['labels'][l]==1 and r['source']!='10_scenes' for r in rr)
            # Exclude synthetic test-pattern negatives: old 100% night metric must not recur.
            neg=sum(r['labels'][l]==0 and 'test_pattern' not in r['evidence'].get(l,'') for r in rr)
            result[s][l+'_real_negative_count']=neg
            if neg<30:block.append('%s:%s fewer than 30 confirmed real-photo negatives'%(s,l))
    ids={s:{r['group_id'] for r in used if r['split']==s} for s in ('train','val','test')}
    overlap={a+'_'+b:len(ids[a]&ids[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
    if any(overlap.values()):raise Blocked('Internal error: new split group overlap '+str(overlap))
    return result,block,overlap

def resolution(rows):
    tr=[r for r in rows if r['use_for_training_manifest'] and r['split']=='train']
    def q(vals,p):
        a=sorted(vals)
        return a[min(len(a)-1,int((len(a)-1)*p))] if a else None
    groups={'all_training':tr}
    for s in sorted({r['source'] for r in tr}):groups['source:'+s]=[r for r in tr if r['source']==s]
    for l in LABELS:groups['positive:'+l]=[r for r in tr if r['labels'][l]==1]
    report={}
    for key,rs in groups.items():
        report[key]={'count':len(rs),'width':{str(p):q([r['width'] for r in rs],p) for p in (.1,.5,.9)},
            'height':{str(p):q([r['height'] for r in rs],p) for p in (.1,.5,.9)}}
    candidates=[]
    for w,h in [(256,144),(320,180),(384,216),(512,288),(640,360)]:
        rates={k:sum(min(w/r['width'],h/r['height'])>1.00001 for r in rs)/len(rs) for k,rs in groups.items() if len(rs)>=50}
        # SEG13 stored dimensions are processed, not native detail; do not let them force resolution.
        native=[v for k,v in rates.items() if k.startswith('source:') and k!='source:seg13']
        candidates.append({'width':w,'height':h,'letterbox_upscale_rates':rates,'native_source_gate':bool(native) and max(native)<=.1})
    options=[c for c in candidates if c['native_source_gate'] and c['width']<=384]
    suggestion=[options[-1]['width'],options[-1]['height']] if options else None
    return {'scope':'selected TRAIN only','stats':report,'candidates':candidates,'first_baseline_suggested_wh':suggestion,
        'final_training_resolution':'not accuracy-validated; no training launched','note':'Resolution heuristic assumes aspect-preserving letterbox. SEG13 dimensions are derivative, not proof of native detail. Compare 320x180/384x216 where supported; do not upscale original files.'}
