"""Accept only the exact audited retrieval metadata; supply snow-positive NAS8 GT."""
import json
from itertools import zip_longest
from pathlib import Path
from policy import unknown,assign,new_record
from storage import Blocked,sha,image_lines

def load_profile():
    path=Path(__file__).with_name('nus21_verified_profile.json')
    profile=json.loads(path.read_text())
    if (profile['schema']!='nus21_pinned_profile_v1' or profile['width']!=21
        or len(set(profile['class_names']))!=21 or profile['class_names'][17]!='snow'
        or {'nighttime','sports'} & set(profile['class_names'])):
        raise Blocked('Invalid pinned NUS21 profile')
    return profile

def validated_pairs(lookup,profile):
    files={}
    # Check ALL six files before yielding any row, including query metadata.
    for name,expected in profile['files'].items():
        path=lookup(name)
        if path.stat().st_size!=expected['bytes'] or sha(path)!=expected['sha256']:
            raise Blocked('NUS21_METADATA_FINGERPRINT_MISMATCH: '+name+'; preserve source and return new diagnostics; never guess a different order')
        files[name]=path
    seen=set();width=profile['width']
    for split in ('database','test'):
        imgfile=files[split+'_img.txt'];names=image_lines(imgfile)
        if len(names)!=profile['expected_rows'][split]:raise Blocked('NUS21 image-list row count mismatch')
        with files[split+'_label.txt'].open(encoding='utf-8-sig') as f,files[split+'_label_onehot.txt'].open(encoding='utf-8-sig') as g:
            count=0
            for i,(name,indices,hot) in enumerate(zip_longest(names,f,g)):
                if name is None or indices is None or hot is None:raise Blocked('NUS21 paired row count mismatch')
                key=name.replace('\\','/').lower()
                if key in seen:raise Blocked('NUS21 query/database overlap or duplicate path')
                seen.add(key)
                values=hot.split()
                if len(values)!=width or any(v not in ('0','1') for v in values):
                    raise Blocked('NUS21 expected binary multi-hot row')
                try:ids=[int(x) for x in indices.split()]
                except ValueError:raise Blocked('NUS21 expected integer index list')
                vals=bytes(map(int,values))
                if ids!=[j for j,v in enumerate(vals) if v]:raise Blocked('NUS21 index/multi-hot disagreement')
                count+=1
                yield split,i,name,vals
            if count!=profile['expected_rows'][split]:raise Blocked('NUS21 label row count mismatch')

def top21_rows(index,lookup,seed,audit,excluded,split_hash):
    profile=load_profile();rows=[];available={'database':0,'test':0};snow={'database':0,'test':0}
    lists={k:str(lookup(k+'_img.txt')) for k in available}
    for split,i,name,values in validated_pairs(lookup,profile):
        path=index.resolve(name)
        if path is None:
            excluded.append({'source':'nuswide','image':name,'native_row':i,'native_partition':split,
                             'reason':'image_missing_keep_annotation_row_index'})
            continue
        available[split]+=1;y=unknown();ev={}
        if values[17]:
            assign(y,ev,'rain_snow',1,'manual:NUS_snow_positive_OR;top21_pinned_column_profile')
            snow[split]+=1
        preferred='test' if split=='test' else split_hash('nus:'+name,seed)
        if split=='database' and preferred=='test':preferred='train'
        rows.append(new_record(path,'nuswide',preferred,'retrieval_top21_verified_snow_only',y,ev,
            original_labels=dict(zip(profile['class_names'],values)),native_image_list=lists[split],
            native_row=i,native_name=name,native_partition=split,sample_id='nus:'+name.replace('\\','/')))
    if not all(available.values()):raise Blocked('NUS21 no photos resolved in a retrieval partition')
    audit['NUS_WIDE']={'format':'retrieval_top21_verified','image_index_count':index.count,
        'metadata_profile':profile,'available':available,'snow_positive_before_quality':snow,
        'target_supervision':'snow positives only; nighttime and sports absent; snow=0 does not imply no rain',
        'split_policy':'retrieval query -> test; database -> deterministic train/val; NOT official NUS Train/Test',
        'unlisted_images_policy':'Photos outside the two paired lists have no verified labels and are not added',
        'missing_rows_excluded':sum(x.get('source')=='nuswide' for x in excluded)}
    print('NUS21 VERIFIED: snow positives',snow,'night/sports remain unknown',flush=True)
    return rows
