"""Merge all three votes; render only the fixed 200-image review selection."""
import argparse
from collections import Counter
import csv
import html
import os
from pathlib import Path
import shutil
from common import LABELS, read, dump, digest, sha, load_run, RecordStore, fuse, approved_legacy_resume


def compatible_completed_signature(completed, signature, plan, metadata, preview_only):
    if completed['signature'] == signature:
        return True
    # Existing preview200 remains immutable. Only an exactly pinned legacy
    # policy with identical plan and teacher metadata is eligible for reuse.
    if not all(approved_legacy_resume(m.get('code_hashes')) for m in metadata):
        return False
    legacy = metadata[0]['code_hashes']
    expected = digest({'plan': plan, 'teacher_metadata': metadata,
                       'merge_code': legacy['merge.py'], 'common_code': legacy['common.py'],
                       'preview_only': preview_only})
    return completed['signature'] == expected


def render(rows, folder):
    from PIL import Image, ImageOps, ImageDraw, ImageFont
    gallery=folder/'review200'; gallery.mkdir(exist_ok=True)
    articles=[]
    fields=['id','image','split','source','reviewed','reviewer','notes']+LABELS
    with (folder/'review200_template.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for row in rows:
            item=row['original']; m=row['mobile']; q=row['qwen']; final=row['labels']
            writer.writerow(dict(id=item['id'],image=item['image'],split=item['split'],source=item['source'],reviewed=0))
            with Image.open(item['image']) as im:
                photo=ImageOps.contain(im.convert('RGB'),(600,520))
            card=Image.new('RGB',(1200,600),'white'); card.paste(photo,(0,50))
            draw=ImageDraw.Draw(card)
            font=ImageFont.load_default(size=20); small=ImageFont.load_default(size=16)
            draw.text((10,10),item['id']+' | '+item['source']+' | '+item['split']+' | PSEUDO-LABEL REVIEW',fill='black',font=small)
            draw.text((615,42),'Class',fill='black',font=font)
            for x,title in zip((830,895,970,1045,1120),('Old','M-old','M-new','Qwen','Vote')):
                draw.text((x,42),title,fill='black',font=small)
            for j,k in enumerate(LABELS):
                values=[item['labels'][k],m['legacy_labels'][k],m['labels'][k],q['labels'][k],final[k]]
                show=['?' if v==-1 else str(v) for v in values]
                color='darkred' if row['vote_details'][k]['flags'] else 'black'
                draw.text((615,80+j*49),k,fill=color,font=font)
                for x,value in zip((830,895,970,1045,1120),show):
                    draw.text((x,80+j*49),value,fill=color,font=font)
                draw.text((615,100+j*49),'M score %.3f | margin %+.3f'%(m['display_scores'][k],sum(m['cosine_margins'][k])/3),fill='gray',font=small)
            draw.text((10,575),'1=present; 0=absent; ?=unknown. Votes/scores are NOT human GT or calibrated confidence.',fill='black',font=small)
            card.save(gallery/(item['id']+'.jpg'),quality=88)
            table='<tr><th>类别</th><th>旧标签</th><th>M旧提示词</th><th>M新投票</th><th>Qwen</th><th>合并建议</th><th>Qwen视觉依据 / 冲突</th></tr>'
            for k in LABELS:
                values=[item['labels'][k],m['legacy_labels'][k],m['labels'][k],q['labels'][k],final[k]]
                table+='<tr><td>'+k+'</td>'+''.join('<td>'+('?' if v==-1 else str(v))+'</td>' for v in values)+'<td>'+html.escape(str(q.get('evidence',{}).get(k,''))+' | '+', '.join(row['vote_details'][k]['flags']))+'</td></tr>'
            articles.append('<article data-source="'+html.escape(item['source'],quote=True)+'" data-conflict="'+str(int(row['needs_review']))+'"><h2>'+html.escape(item['id']+' '+item['source']+' '+item['split'])+'</h2><img loading="lazy" src="review200/'+item['id']+'.jpg"><p>'+html.escape(item['image'])+'</p><table>'+table+'</table></article>')
    sources=sorted({r['original']['source'] for r in rows})
    page='''<!doctype html><meta charset="utf-8"><title>NAS8 三方标注审核 200</title>
<style>body{font:15px sans-serif;margin:24px;background:#eee}article{background:white;padding:18px;margin:20px 0}img{max-width:100%}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:6px}p{overflow-wrap:anywhere}</style>
<h1>200 张三方标签对照</h1><p>旧标签有噪声；M 旧提示词仅对照、不参与第四票。合并为伪标签建议。? 为未知；模型得分和投票比例不等于正确率。本页固定 200 张，其余图片继续标注。雪山应考虑 snow；outdoor 与 landscape 可共存；运动场地按本项目定义计 sports。</p>
<label>来源 <select id="src"><option value="">全部</option>'''+''.join('<option>'+html.escape(s)+'</option>' for s in sources)+'''</select></label> <label><input id="conf" type="checkbox">仅分歧/缺少共识</label>'''+''.join(articles)+'''
<script>function filter(){document.querySelectorAll('article').forEach(a=>{a.hidden=(src.value&&a.dataset.source!==src.value)||(conf.checked&&a.dataset.conflict!=='1')})}src.onchange=filter;conf.onchange=filter;</script>'''
    (folder/'index.html').write_text(page,encoding='utf-8')
    for start in range(0,len(rows),10):
        sheet=Image.new('RGB',(1200,1500),'#eeeeee')
        for j,r in enumerate(rows[start:start+10]):
            with Image.open(gallery/(r['original']['id']+'.jpg')) as im:
                sheet.paste(im.resize((600,300)),((j%2)*600,(j//2)*300))
        sheet.save(folder/('contact_%02d.jpg'%(start//10+1)),quality=88)


def run(a):
    root,plan,items=load_run(a.output_root)
    if a.preview_only: items=items[:200]
    directories=[root/'mobile']+[root/('qwen_%02d'%s) for s in range(a.shards)]
    metadata=[read(d/'metadata.json') for d in directories]
    if any(m['selection']!=plan['items_digest'] for m in metadata): raise ValueError('Teacher/selection mismatch')
    for s,m in enumerate(metadata[1:]):
        if m['shard']!=s or m['shards']!=a.shards: raise ValueError('Qwen shard mismatch')
    for m in metadata[2:]:
        for k in metadata[1]:
            if k!='shard' and m[k]!=metadata[1][k]: raise ValueError('Qwen workers used different model/prompt/config')
    signature=digest({'plan':plan,'teacher_metadata':metadata,'merge_code':sha(Path(__file__)),'common_code':sha(Path(__file__).with_name('common.py')),'preview_only':a.preview_only})
    dest=root/('preview200' if a.preview_only else 'merged')
    if (dest/'COMPLETE.json').exists():
        completed=read(dest/'COMPLETE.json')
        if not compatible_completed_signature(completed,signature,plan,metadata,a.preview_only):
            raise ValueError('Existing merge has a different policy')
        for n,h in completed['files'].items():
            if sha(dest/n)!=h: raise ValueError('Completed merge output modified')
        print('MERGE_REUSED',str(dest),flush=True); return
    if dest.exists(): raise ValueError('Incomplete final output; preserve and diagnose')
    stores=[RecordStore(d/'results.sqlite',digest(m),readonly=True) for d,m in zip(directories,metadata)]
    # These DBs must have stopped writing; main run script waits for the workers.
    dbhash={str(d/'results.sqlite'):sha(d/'results.sqlite') for d in directories}
    build=root/(dest.name+'.building.'+str(os.getpid())); build.mkdir()
    import json
    outputs={s:(build/(s+'.jsonl')).open('w',encoding='utf-8') for s in ('train','val','test')}
    audit=(build/'votes.jsonl').open('w',encoding='utf-8')
    conflictfile=(build/'conflicts.jsonl').open('w',encoding='utf-8')
    counts={s:{k:Counter() for k in LABELS} for s in outputs}
    changes={k:Counter() for k in LABELS}; parse_errors=0; previews=[]; supervision_missing=0
    prompt_changes={k:Counter() for k in LABELS}
    voter_counts={name:{k:Counter() for k in LABELS} for name in ('old','mobile','qwen')}
    agreement={k:Counter() for k in LABELS}
    for i,item in enumerate(items):
        m=stores[0].get(item); q=stores[1+i%a.shards].get(item)
        if m is None or q is None: raise ValueError('Missing annotation: '+item['id']+'; resume teacher workers')
        if sha(item['image'])!=item['image_sha256']: raise ValueError('Image changed before merge')
        labels,detail=fuse(item['labels'],m['labels'],q['labels'],item.get('evidence'))
        needs=any(v['flags'] for v in detail.values()) or -1 in labels.values()
        row={'id':item['id'],'image':item['image'],'original':item,'mobile':m,'qwen':q,'labels':labels,'vote_details':detail,'needs_review':needs}
        audit.write(json.dumps(row,ensure_ascii=False)+'\n')
        if needs: conflictfile.write(json.dumps({'id':item['id'],'image':item['image'],'labels':labels,'vote_details':detail},ensure_ascii=False)+'\n')
        new=dict(item,original_labels=item['labels'],original_evidence=item.get('evidence',{}),labels=labels,
                 evidence={k:'pseudo_three_votes_v1:'+str(detail[k]['votes']) for k in LABELS},
                 annotation_kind='pseudo_label_proposal',has_vote_supervision=any(v!=-1 for v in labels.values()))
        outputs[item['split']].write(json.dumps(new,ensure_ascii=False)+'\n')
        for k in LABELS:
            counts[item['split']][k][str(labels[k])]+=1
            changes[k][str(item['labels'][k])+'->'+str(labels[k])]+=1
            prompt_changes[k][str(m['legacy_labels'][k])+'->'+str(m['labels'][k])]+=1
            for name, yy in [('old',item['labels']),('mobile',m['labels']),('qwen',q['labels'])]:
                voter_counts[name][k][str(yy[k])]+=1
            agreement[k]['disagreement' if detail[k]['positive_votes'] and detail[k]['negative_votes'] else 'no_disagreement']+=1
            agreement[k]['majority' if detail[k]['majority_proposal']!=-1 else 'insufficient_votes']+=1
        if q['status']!='OK': parse_errors+=1
        if not new['has_vote_supervision']: supervision_missing+=1
        if item['preview']: previews.append(row)
    for f in list(outputs.values())+[audit,conflictfile]: f.close()
    for store in stores: store.close()
    for path,h in dbhash.items():
        if sha(path)!=h: raise ValueError('Teacher DB changed during merge')
    load_run(root)
    render(previews,build)
    if not a.preview_only:
        reference=build/'original_evaluation'; reference.mkdir()
        for n in plan['input_hashes']:
            if n in ('val.jsonl','test.jsonl','val_strict.jsonl','test_strict.jsonl'):
                shutil.copyfile(Path(plan['input_root'])/n,reference/n)
    dump(build/'summary.json',{'schema':'nas8_three_votes_v1','status':'ANNOTATED_REVIEW_REQUIRED',
         'records':len(items),'preview':len(previews),'labels':LABELS,'split_class_counts':counts,
         'old_to_fused_transitions':changes,'mobile_old_to_new_prompt_transitions':prompt_changes,
         'voter_label_counts':voter_counts,'agreement_counts':agreement,
         'qwen_parse_error_abstentions':parse_errors,'all_unknown_records':supervision_missing,
         'positive_records_deleted':0,'source_records_deleted':0,'splits_changed':False,
         'human_verified_gt':False,'soft_probability_targets_produced':False,
         'note':'Agreement fraction is NOT probability. No test accuracy evaluated against teacher-generated labels. Review before training.'})
    files={str(p.relative_to(build)):sha(p) for p in build.rglob('*') if p.is_file()}
    dump(build/'COMPLETE.json',{'status':'PREVIEW_COMPLETE' if a.preview_only else 'ANNOTATION_COMPLETE_REVIEW_REQUIRED',
         'signature':signature,'records':len(items),'files':files,'NEW_TRAINING_STARTED':False})
    build.rename(dest)
    # Package only review cards, contact sheets, template and summary, not images/weights/whole database.
    import zipfile
    with zipfile.ZipFile(root/('preview200.zip' if a.preview_only else 'review200_final.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(dest.rglob('*')):
            if p.is_file() and (p.parent.name=='review200' or p.name.startswith('contact_') or p.name in ('index.html','review200_template.csv','summary.json')):
                z.write(p,str(p.relative_to(dest)))
    print('PREVIEW_COMPLETE' if a.preview_only else 'ANNOTATION_COMPLETE_REVIEW_REQUIRED',str(dest),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--shards',type=int,default=2)
    p.add_argument('--preview-only',action='store_true')
    a=p.parse_args()
    if a.shards<1: p.error('Invalid shards')
    run(a)
