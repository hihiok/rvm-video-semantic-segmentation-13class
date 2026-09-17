#!/usr/bin/env python3
"""Visualize saved Rev10 test predictions, with per-class seeded correct/error sampling."""
import argparse,csv,hashlib,html,json,random,textwrap,zipfile
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from rev10_data import LABELS,sha,load,dump
NAMES=dict(zip(LABELS,['夜景','室内','雨雪','办公','户外','自然风景','运动','客观图']))
CSS='body{font-family:Arial,sans-serif;background:#f3f5f8;color:#172536;margin:24px}a{color:#12569c}table{border-collapse:collapse;width:100%}td,th{padding:7px;border:1px solid #ced6e0;text-align:left;overflow-wrap:anywhere}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}.card{background:white;padding:12px;border-radius:10px}.card img{width:100%}.warn{color:#ad3f1b}details{margin:8px 0}code{overflow-wrap:anywhere}'
def page(title,body):return '<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><style>'+CSS+'</style><body><h1>'+html.escape(title)+'</h1>'+body+'</body></html>'
def result(y,p):
 if y==-1:return 'UNKNOWN'
 return ('TP' if p else 'FN') if y==1 else ('FP' if p else 'TN')
def tier(row,label):
 if row['labels'][label]==-1:return 'UNKNOWN'
 ev=row.get('evidence',{}).get(label,'')
 if ev.startswith(('human_review:','user_review:')):return 'HUMAN'
 if ev.startswith('manual:'):return 'MANUAL'
 return 'WEAK/RULE'
def select_cases(gt,scores,thresholds,n,seed):
 predictions=scores>=thresholds
 chosen={};stats={}
 for j,label in enumerate(LABELS):
  status=[result(int(gt[i,j]),bool(predictions[i,j])) for i in range(len(gt))]
  cnt=Counter(status);chosen[label]={};stats[label]={'pool_counts':dict(cnt)}
  for group,wanted in [('correct',{'TP','TN'}),('error',{'FP','FN'})]:
   pool=[i for i,v in enumerate(status) if v in wanted]
   # Canonical index order is sorted image path, not NPZ storage order.
   rng=random.Random(str(seed)+':'+label+':'+group)
   chosen[label][group]=rng.sample(pool,min(n,len(pool)))
   stats[label][group]={'available':len(pool),'selected':len(chosen[label][group]),'shortfall':max(0,n-len(pool)),'selected_outcomes':dict(Counter(status[i] for i in chosen[label][group]))}
 return chosen,stats

def read_inputs(run,root):
 paths={'config':run/'config.json','complete':run/'COMPLETE.json','thresholds':run/'thresholds.json','predictions':run/'test_predictions.npz','manifest':root/'test.jsonl'}
 hashes={k:sha(v) for k,v in paths.items()}
 config=json.loads(paths['config'].read_text());complete=json.loads(paths['complete'].read_text())
 if complete.get('status')!='TRAINING_COMPLETE':raise ValueError('Training completion marker required')
 if config.get('labels')!=LABELS:raise ValueError('Prediction label order does not match NAS8')
 if config.get('data_sha256',{}).get('test.jsonl')!=hashes['manifest']:raise ValueError('Test manifest differs from training input')
 th=json.loads(paths['thresholds'].read_text())
 if set(th)!=set(LABELS):raise ValueError('Invalid threshold labels')
 thresholds=np.array([th[k] for k in LABELS],dtype=float)
 if not np.all(np.isfinite(thresholds)) or np.any((thresholds<0)|(thresholds>1)):raise ValueError('Invalid thresholds')
 rows=load(paths['manifest']);mapping={r['image']:r for r in rows}
 if len(mapping)!=len(rows) or any(r.get('split')!='test' for r in rows):raise ValueError('Duplicate image or non-test row')
 with np.load(paths['predictions'],allow_pickle=False) as z:
  images=z['image'];gt=z['gt'].copy();scores=z['scores'].copy()
  if images.ndim!=1 or images.dtype.kind not in ('U','S'):raise ValueError('Image paths must be a string vector')
  image_paths=[str(v) if isinstance(v,(str,np.str_)) else v.decode('utf-8') for v in images]
 if not image_paths or len(set(image_paths))!=len(image_paths):raise ValueError('Empty/duplicate prediction paths')
 if set(image_paths)!=set(mapping):raise ValueError('Predictions must cover exactly the test manifest')
 if gt.shape!=(len(rows),8) or scores.shape!=gt.shape:raise ValueError('Prediction shape mismatch')
 if not np.isfinite(scores).all() or np.any((scores<0)|(scores>1)):raise ValueError('Expected finite sigmoid probabilities')
 if not np.isin(gt,[-1,0,1]).all():raise ValueError('Invalid GT values')
 ordered=[mapping[p] for p in image_paths]
 expected=np.array([[r['labels'][k] for k in LABELS] for r in ordered])
 if not np.array_equal(gt,expected):raise ValueError('Saved GT and manifest disagree; do not visualize misaligned results')
 idx=sorted(range(len(ordered)),key=lambda i:ordered[i]['image'])
 return [ordered[i] for i in idx],gt[idx],scores[idx],thresholds,config,paths,hashes

def font(size):
 for p in ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf']:
  if Path(p).is_file():return ImageFont.truetype(p,size)
 try:return ImageFont.load_default(size=size)
 except TypeError:return ImageFont.load_default()
def render_card(path,row,score,threshold,focus,status,wh):
 canvas=Image.new('RGB',(1360,1060),'#f7f9fc');d=ImageDraw.Draw(canvas)
 f=font(21);small=font(16);heading=font(30)
 color='#bd3434' if status in ('FP','FN') else '#177552'
 d.text((24,18),focus+' / '+status+'   GT='+str(row['labels'][focus]),font=heading,fill=color)
 d.text((24,65),'TEST | source: '+row['source']+' | Focus result only; other labels may differ',font=f,fill='#25354a')
 with Image.open(row['image']) as im:
  im=ImageOps.exif_transpose(im).convert('RGB');preview=ImageOps.contain(im,(880,480));canvas.paste(preview,(24+(880-preview.width)//2,110+(480-preview.height)//2))
  # Context inset only: PIL bilinear preview, not a rerun of OpenCV inference.
  inp=im.resize(tuple(wh),Image.Resampling.BILINEAR);inp=inp.resize((384,216),Image.Resampling.BILINEAR);canvas.paste(inp,(948,110))
 d.text((948,339),'Input aspect preview '+str(wh[0])+'x'+str(wh[1]),font=small,fill='#25354a')
 positives=[k for k in LABELS if row['labels'][k]==1];predicted=[k for k,p,t in zip(LABELS,score,threshold) if p>=t]
 text='GT+: '+(', '.join(positives) or '(none)')+'\n\nPRED+: '+(', '.join(predicted) or '(none)')
 yy=380
 for part in text.split('\n'):
  for line in textwrap.wrap(part,33) or ['']:
   d.text((948,yy),line,font=small,fill='#25354a');yy+=22
 d.text((24,603),'GT 1=yes  0=no  ?=unknown | WEAK/RULE is not individually verified GT',font=f,fill='#8b4c16')
 columns=[24,250,325,410,550,695,830]
 headers=['LABEL','GT','PRED','PROBABILITY','THRESHOLD','RESULT','EVIDENCE']
 for x,h in zip(columns,headers):d.text((x,640),h,font=small,fill='#445469')
 for j,k in enumerate(LABELS):
  y=673+j*34;pred=bool(score[j]>=threshold[j]);outcome=result(row['labels'][k],pred)
  bg='#fde7e7' if outcome in ('FP','FN') else ('#e8f4ed' if outcome!='UNKNOWN' else '#e9edf2')
  d.rectangle((20,y-2,1340,y+30),fill=bg)
  values=[k,'?' if row['labels'][k]==-1 else str(row['labels'][k]),str(int(pred)),format(float(score[j]),'.6f'),format(float(threshold[j]),'.6f'),outcome,tier(row,k)]
  for x,v in zip(columns,values):d.text((x,y),v,font=f,fill='#182737')
 d.text((24,958),'FP: predicted yes, GT no | FN: predicted no, GT yes | UNKNOWN: not scored',font=small,fill='#25354a')
 lines=textwrap.wrap(row['image'],145)
 for n,line in enumerate(lines[:3]):d.text((24,984+n*21),line,font=small,fill='#445469')
 canvas.save(path,quality=92)

def detail_table(row,scores,threshold):
 head='<table><tr><th>类别</th><th>GT</th><th>预测</th><th>概率</th><th>阈值</th><th>结果</th><th>依据</th></tr>'
 for j,k in enumerate(LABELS):
  y=row['labels'][k];p=bool(scores[j]>=threshold[j]);ev=row.get('evidence',{}).get(k,'')
  head+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [NAMES[k], '?' if y==-1 else y,int(p),'%.6f'%scores[j],'%.6f'%threshold[j],result(y,p),tier(row,k)+' | '+ev])+'</tr>'
 return head+'</table>'
def run(run_dir,data_root,out,per_group=10,seed=20260917):
 run_dir=Path(run_dir).resolve();data_root=Path(data_root).resolve();out=Path(out).resolve();archive=out.with_suffix('.zip')
 if per_group<1:raise ValueError('per-group must be positive')
 if out.exists() or archive.exists():raise ValueError('New output directory and ZIP required')
 if out==data_root or data_root in out.parents or out==run_dir:raise ValueError('Output must not overwrite label/model directories')
 rows,gt,scores,threshold,config,paths,hashes=read_inputs(run_dir,data_root)
 cases,stats=select_cases(gt,scores,threshold,per_group,seed)
 out.mkdir(parents=True,mode=0o700)
 try:
  records=[];index=['<p>每类随机抽取正确与错误各%d张。正确=TP+TN；错误=FP+FN；未知GT不判对错。每个分组内不重复，不同类别可出现同一图片。</p>'%per_group,
   '<p class="warn">这是分类别挑选的展示集，不代表测试集错误率。弱/规则标签下的对错仅表示与该标签一致或不一致。雨雪弱负例不是逐图审核GT。</p>']
  for j,label in enumerate(LABELS):
   index.append('<h2>'+NAMES[label]+' / '+label+'</h2><p>完整test计数：'+html.escape(json.dumps(stats[label]['pool_counts']))+'</p><ul>')
   for group in ('correct','error'):
    folder=out/label/group;folder.mkdir(parents=True);cards=[];panels=[]
    for number,i in enumerate(cases[label][group],1):
     row=rows[i];status=result(int(gt[i,j]),bool(scores[i,j]>=threshold[j]));name='%02d_%s.jpg'%(number,status)
     render_card(folder/name,row,scores[i],threshold,label,status,config['input_wh']);panels.append(folder/name)
     record={'focus_class':label,'group':group,'outcome':status,'image':row['image'],'source':row['source'],'focus_gt':int(gt[i,j]),'focus_prediction':int(scores[i,j]>=threshold[j]),'focus_probability':float(scores[i,j]),'focus_threshold':float(threshold[j]),'focus_evidence_tier':tier(row,label),'focus_evidence':row.get('evidence',{}).get(label,''),'card':str((folder/name).relative_to(out)),'all_gt':row['labels'],'all_predictions':dict(zip(LABELS,map(int,scores[i]>=threshold))),'all_probabilities':dict(zip(LABELS,map(float,scores[i])))}
     records.append(record)
     cards.append('<div class="card"><a href="'+name+'"><img loading="lazy" src="'+name+'"></a><p>'+html.escape(status+' | '+row['source']+' | '+row['image'])+'</p><details><summary>八类GT、预测与完整标注依据</summary>'+detail_table(row,scores[i],threshold)+'</details></div>')
    if panels:
     contact=Image.new('RGB',(680*2,530*((len(panels)+1)//2)),'white')
     for n,p in enumerate(panels):
      with Image.open(p) as im:contact.paste(im.resize((680,530)),((n%2)*680,(n//2)*530))
     contact.save(folder/'contact.jpg',quality=90)
    st=stats[label][group];title=NAMES[label]+' '+group+' '+str(st['selected'])+'/'+str(per_group)
    warning='<p class="warn">候选不足，缺%d张；没有跨集合取图或重复凑数。</p>'%st['shortfall'] if st['shortfall'] else ''
    (folder/'index.html').write_text(page(title,'<a href="../../index.html">返回总览</a>'+warning+'<div class="grid">'+''.join(cards)+'</div>'),encoding='utf-8')
    index.append('<li><a href="%s/%s/index.html">%s</a>（总候选%d，抽取%d，缺%d）</li>'%(label,group,html.escape(title),st['available'],st['selected'],st['shortfall']))
   index.append('</ul>')
  (out/'index.html').write_text(page('NAS8 Test：GT 与预测分类', ''.join(index)),encoding='utf-8')
  (out/'selected_cases.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
  fields=['focus_class','group','outcome','image','source','focus_gt','focus_prediction','focus_probability','focus_threshold','focus_evidence_tier','focus_evidence','card']
  with (out/'selected_cases.csv').open('w',encoding='utf-8-sig',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(records)
  for key,p in paths.items():
   if sha(p)!=hashes[key]:raise ValueError('Input changed during visualization: '+key)
  audit={'status':'VISUALIZATION_COMPLETE','seed':seed,'per_class_per_group_requested':per_group,'display_records':len(records),'unique_images':len({r['image'] for r in records}),'by_class':stats,'source_counts_display_records':dict(Counter(r['source'] for r in records)),'input_files':{k:str(p) for k,p in paths.items()},'input_sha256':hashes,'thresholds':dict(zip(LABELS,map(float,threshold))),'inference':'reuses saved test sigmoid scores; no inference or training rerun','unknown_gt_scored':False,'sampling':'uniform random without replacement within each class/correct-or-error pool; repeats across classes allowed','weak_labels':'retained and marked; not individually verified ground truth','input_unchanged':True}
  dump(out/'summary.json',audit)
  with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
   for p in sorted(out.rglob('*')):
    if p.is_file():z.write(p,str(Path(out.name)/p.relative_to(out)))
  print(json.dumps({'status':audit['status'],'output':str(out),'zip':str(archive),'display_records':len(records),'unique_images':audit['unique_images'],'shortfalls':{k:{g:stats[k][g]['shortfall'] for g in ('correct','error')} for k in LABELS}},ensure_ascii=False,indent=2))
  return audit
 except Exception as e:
  dump(out/'BLOCKED.json',{'error':str(e)});raise
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--per-group',type=int,default=10);p.add_argument('--seed',type=int,default=20260917)
 a=p.parse_args();run(a.run_dir,a.data_root,a.output_dir,a.per_group,a.seed)
