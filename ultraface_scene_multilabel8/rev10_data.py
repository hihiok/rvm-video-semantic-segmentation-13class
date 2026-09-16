"""Read-only validation and input-size selection for the user-approved Rev10."""
import hashlib,json,statistics
from pathlib import Path
from collections import Counter
LABELS=['night','indoor','rain_snow','office','outdoor','landscape','sports','objective_image']
SPLITS=('train','val','test')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def load(p):
 with Path(p).open(encoding='utf-8') as f:return [json.loads(line) for line in f if line.strip()]
def counts(rows):
 return {k:{str(v):sum(r['labels'][k]==v for r in rows) for v in (1,0,-1)} for k in LABELS}
def validate(root,accept_weak):
 root=Path(root)
 if not accept_weak:raise ValueError('User-authorized weak-weather training requires --accept-weak-weather')
 if (root/'BLOCKED.json').exists():raise ValueError('BLOCKED input')
 files=['PREPARED.json','summary.json','source_audit.json']+[s+'.jsonl' for s in SPLITS]
 hashes={n:sha(root/n) for n in files}
 summary=json.loads((root/'summary.json').read_text())
 if summary.get('schema')!='nas8_weather_rev10':raise ValueError('Expected completed Rev10 schema')
 if summary.get('status')!='PREPARED_REVIEW_REQUIRED':raise ValueError('Unexpected dataset status')
 unaccepted=[x for x in summary.get('training_quality_blockers',[]) if 'rain_snow' not in str(x)]
 if unaccepted:raise ValueError('Unresolved non-weather blockers: '+str(unaccepted))
 data={s:load(root/(s+'.jsonl')) for s in SPLITS};paths=set();groups=set();pixels=set();report={}
 for split,rows in data.items():
  if not rows or len(rows)!=summary['splits'][split]['records']:raise ValueError('Missing/inconsistent split '+split)
  weak=0
  for r in rows:
   if r.get('split')!=split or not r.get('use_for_training_manifest'):raise ValueError('Invalid split/active row')
   if set(r['labels'])!=set(LABELS) or any(type(v)!=int or v not in (-1,0,1) for v in r['labels'].values()):raise ValueError('Invalid NAS8 labels')
   path=str(Path(r['image']).resolve());group=r['group_id'];pixel=r.get('rgb_sha256')
   if path in paths or group in groups or (pixel and pixel in pixels):raise ValueError('Duplicate path/group/pixels, including cross-split leakage')
   if not Path(path).is_file():raise ValueError('Missing image '+path)
   if min(r['width'],r['height'])<=0:raise ValueError('Bad dimensions')
   paths.add(path);groups.add(group)
   if pixel:pixels.add(pixel)
   y=r['labels'];ev=r.get('evidence',{}).get('rain_snow','')
   if y['rain_snow']==0 and (y['sports']==1 or y['objective_image']==1):raise ValueError('Rev10 excludes sports/synthetic weather negatives')
   if ev.startswith('weak_user_rule:'):
    if y['rain_snow']!=0 or not any(y[g]==1 for g in ('night','indoor','office')):raise ValueError('Invalid weak-weather provenance')
    weak+=1
  c=counts(rows)
  if split=='train' and any(c[k]['1']==0 or c[k]['0']==0 for k in LABELS):raise ValueError('Train needs known positives and negatives for every class')
  report[split]={'records':len(rows),'labels':c,'weak_weather_negatives':weak,'sources':dict(Counter(r['source'] for r in rows))}
 return data,{'root':str(root.resolve()),'sha256':hashes,'splits':report,'weak_weather_accepted_by_user':True,'dataset_READY_FOR_TRAINING_not_modified':True,'evaluation_limit':'Main metrics use sampled partial labels incl. weak weather. Strict weather specificity/F1/AP unavailable without verified negatives.'}
def resolution(rows):
 candidates=[(256,144),(320,180),(384,216),(512,288),(640,360)]
 long_median=statistics.median(max(r['width'],r['height']) for r in rows)
 # A transparent speed-first heuristic, NOT a measured accuracy optimum.
 chosen=(320,180) if long_median<=500 else (384,216)
 sources={}
 for s in sorted({r['source'] for r in rows}):
  rr=[r for r in rows if r['source']==s]
  sources[s]={'count':len(rr),'median_width':statistics.median(r['width'] for r in rr),'median_height':statistics.median(r['height'] for r in rr)}
 return {'input_wh':list(chosen),'median_long_edge':long_median,'sources':sources,'candidates':[{'wh':[w,h],'axis_upscale_fraction':sum(w>r['width'] or h>r['height'] for r in rows)/len(rows)} for w,h in candidates], 'selection_reason':'Speed-first 16:9 baseline: median native/recorded long edge <=500 ->320x180, else384x216. Existing derivatives may be upscaled; no claim of original native detail or validated optimum.'}
def dump(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
