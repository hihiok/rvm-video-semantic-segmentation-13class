"""GT-only review cards. Fixed table columns, measured wrapping, dynamic height."""
from __future__ import annotations
import csv,html,random
from collections import Counter
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont,ImageOps
from policy import LABELS,REPORTED
from storage import Blocked,dump

QUOTAS={'coco':15,'places365':20,'seg13':15,'mirflickr':20,'nuswide':20,'10_scenes':10}

def font(size):
    for p in ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf']:
        if Path(p).is_file():return ImageFont.truetype(p,size)
    return ImageFont.load_default()

def wrap(text,f,width):
    # Binary-search a fitting prefix. Handles long tokens without per-character scans.
    draw=ImageDraw.Draw(Image.new('RGB',(1,1)));text=str(text);lines=[]
    while text:
        if draw.textbbox((0,0),text,font=f)[2] <= width:
            lines.append(text);break
        lo,hi=1,len(text)
        while lo<hi:
            mid=(lo+hi+1)//2
            if draw.textbbox((0,0),text[:mid],font=f)[2] <= width:lo=mid
            else:hi=mid-1
        lines.append(text[:lo]);text=text[lo:]
    return lines or ['']


def select100(rows,seed):
    rng=random.Random(seed);chosen=[];seen=set()
    for source,count in QUOTAS.items():
        rs=[r for r in rows if r['source']==source];rng.shuffle(rs)
        # Keep review representative when a source (e.g. NUS21) has many rows
        # without any target supervision. The shuffle remains seeded within tiers.
        rs.sort(key=lambda r:not r['use_for_training_manifest'])
        prioritized=[r for r in rs if Path(r['image']).name in REPORTED]
        for label in LABELS:
            prioritized += [r for r in rs if r['labels'][label]==1][:2]
        prioritized+=rs;have=0
        for r in prioritized:
            if r['group_id'] in seen:continue
            chosen.append(r);seen.add(r['group_id']);have+=1
            if have==count:break
        if have<count:raise Blocked('GT100 source coverage insufficient: %s need=%d found_unique=%d'%(source,count,have))
    if len(chosen)!=100:raise Blocked('Internal GT100 count error')
    return chosen

def draw_card(r,index,out):
    f=font(21);large=font(26);line=29;left=32;W=1200
    with Image.open(r['image']) as im:thumb=ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'),(1100,470))
    header=['#%03d | source=%s | split=%s | %dx%d'%(index,r['source'],r['split'],r['width'],r['height']),
        'GT only (no predictions): 1=positive, 0=negative, ?=not supervised',
        'Selected for manifests: '+str(r['use_for_training_manifest'])+' | '+r.get('exclusion_reason',''),
        'Image: '+r['image'],'Source detail: '+str(r['detail'])]
    head=[x for text in header for x in wrap(text,f,W-2*left)]
    table=[]
    for l in LABELS:
        reason=r['evidence'].get(l,'No direct evidence for this label')
        lines=wrap(reason,f,700);table.append((l,lines,max(44,len(lines)*line+16)))
    H=530+len(head)*line+55+sum(t[2] for t in table)+40
    c=Image.new('RGB',(W,H),(248,249,251));c.paste(thumb,((W-thumb.width)//2,20));d=ImageDraw.Draw(c)
    y=510
    for text in head:d.text((left,y),text,font=f,fill=(25,30,40));y+=line
    y+=12;d.rectangle((left,y,W-left,y+40),fill=(218,226,236))
    d.text((40,y+6),'LABEL',font=f,fill='black');d.text((305,y+6),'GT',font=f,fill='black');d.text((415,y+6),'EVIDENCE / LIMITATION',font=f,fill='black');y+=44
    for l,lines,height in table:
        d.line((left,y,W-left,y),fill=(195,201,211))
        v=r['labels'][l];state='?' if v==-1 else str(v)
        d.text((40,y+8),l,font=f,fill=(25,30,40));d.text((310,y+8),state,font=large,fill=(25,90,45) if v==1 else (60,65,75))
        for k,text in enumerate(lines):d.text((415,y+8+k*line),text,font=f,fill=(45,50,60))
        y+=height
    if y+20>H:raise Blocked('Visualization layout overflow')
    c.save(out,quality=94)
    return {'width':W,'height':H,'last_row_bottom':y}

def export_review(rows,out,seed):
    out=Path(out);folder=out/'gt100';folder.mkdir();chosen=select100(rows,seed);cards=[];measure=[]
    with (out/'gt100.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['index','source','image','split','use_for_training_manifest']+LABELS);writer.writeheader()
        for n,r in enumerate(chosen,1):
            name='%03d_%s_%s.jpg'%(n,r['source'],Path(r['image']).stem)
            measure.append(draw_card(r,n,folder/name));cards.append((name,r))
            writer.writerow(dict(index=n,source=r['source'],image=r['image'],split=r['split'],use_for_training_manifest=r['use_for_training_manifest'],**r['labels']))
    # Contact pages: 10 readable image-only thumbnails each, full table always in individual card.
    for start in range(0,100,10):
        sheet=Image.new('RGB',(1500,800),(245,245,245));d=ImageDraw.Draw(sheet)
        for k,(_,r) in enumerate(cards[start:start+10]):
            with Image.open(r['image']) as im:thumb=ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'),(280,290))
            x=(k%5)*300;y=(k//5)*400;sheet.paste(thumb,(x+(300-thumb.width)//2,y+10))
            d.text((x+10,y+305),'#%03d %s'%(start+k+1,r['source']),font=font(16),fill='black')
            text='GT+: '+(', '.join(l for l in LABELS if r['labels'][l]==1) or '(none; see GT0/?)')
            for j,line in enumerate(wrap(text,font(15),280)):d.text((x+10,y+330+18*j),line,font=font(15),fill='black')
        sheet.save(out/('contact_%02d.jpg'%(start//10+1)),quality=92)
    items=['<a href="%s"><img loading="lazy" src="%s"></a>'%(html.escape(n),html.escape(n)) for n,r in cards]
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>NAS8 GT100</title><style>body{font-family:sans-serif}img{width:580px;vertical-align:top;margin:12px;max-width:95vw}</style><h1>NAS8 GT100: click to view full resolution</h1><p>These are partial, source-derived GT, not predictions. Unknown is not negative. Weak category labels still need review.</p>'+''.join(items),encoding='utf-8')
    dump(out/'gt100_audit.json',{'count':100,'source_counts':dict(Counter(r['source'] for r in chosen)),
        'positive_coverage':{l:sum(r['labels'][l]==1 for r in chosen) for l in LABELS},'layouts':measure})
    return chosen

def review_template(rows,out):
    # Only explicitly human-reviewed rows are accepted by the next run.
    fields=['image','reviewed','reviewer','note']+LABELS
    with Path(out).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow(dict(image=r['image'],reviewed='',reviewer='',note='',**dict.fromkeys(LABELS,'')))
