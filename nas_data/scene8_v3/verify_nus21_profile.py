#!/usr/bin/env python3
"""Reproduce column recovery from official GT statistics; no image-order guessing."""
import argparse,hashlib,json,zipfile
from collections import Counter
from pathlib import Path
from storage import Blocked,dump

def binary_matrix(data,width):
    rows=[]
    for n,line in enumerate(data.decode('utf-8-sig').splitlines(),1):
        tokens=line.split()
        if len(tokens)!=width or any(x not in ('0','1') for x in tokens):
            raise Blocked('Invalid binary matrix at row '+str(n))
        rows.append(bytes(map(int,tokens)))
    return rows

def signatures(rows):
    return Counter(sum(v<<i for i,v in enumerate(row)) for row in rows)

def histogram_sha(counter):
    return hashlib.sha256(json.dumps(sorted(counter.items()),separators=(',',':')).encode()).hexdigest()

def verify(diagnostics,official):
    metadata={};matrices=[];all_names=[]
    with zipfile.ZipFile(diagnostics) as z:
        def read(name):
            paths=[p for p in z.namelist() if p.startswith('metadata/') and p.split('/')[-1]==name]
            if len(paths)!=1:raise Blocked('Expected one diagnostic member: '+name)
            data=z.read(paths[0])
            metadata[name]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
            return data
        for split,expected in (('database',193734),('test',2100)):
            names=read(split+'_img.txt').decode().splitlines()
            values=binary_matrix(read(split+'_label_onehot.txt'),21)
            indices=[list(map(int,line.split())) for line in read(split+'_label.txt').decode().splitlines()]
            if not len(names)==len(values)==len(indices)==expected:raise Blocked('Unexpected retrieval row counts')
            if not all(x==[i for i,v in enumerate(row) if v] for x,row in zip(indices,values)):
                raise Blocked('Integer indices and multi-hot rows disagree')
            matrices.extend(values);all_names.extend(names)
    if len(set(all_names))!=len(all_names):raise Blocked('Repeated/overlapping retrieval paths')
    gt={};official_files={}
    with zipfile.ZipFile(official) as z:
        for name in z.namelist():
            if '/AllLabels/Labels_' not in name:continue
            key=name.split('/')[-1][7:-4];raw=z.read(name)
            tokens=raw.split()
            if len(tokens)!=269648 or any(v not in (b'0',b'1') for v in tokens):raise Blocked('Unexpected official GT shape')
            gt[key]=bytes(map(int,tokens))
            official_files[key]=hashlib.sha256(raw).hexdigest()
    if len(gt)!=81:raise Blocked('Expected 81 official concepts')
    official_counts={k:sum(v) for k,v in gt.items()}
    counts=[sum(row[i] for row in matrices) for i in range(21)]
    order=[]
    for count in counts:
        candidates=[k for k,v in official_counts.items() if v==count]
        if len(candidates)!=1:raise Blocked('Column marginal does not identify a unique concept')
        order.append(candidates[0])
    if order!=sorted(gt,key=lambda k:-official_counts[k])[:21]:
        raise Blocked('Recovered order is not official top-21 by count')
    native=signatures(zip(*(gt[k] for k in order)));native.pop(0,None)
    retrieval=signatures(matrices)
    if native!=retrieval:raise Blocked('Full 21-label joint distribution mismatch')
    return {
        'schema':'nus21_pinned_profile_v1','width':21,'class_names':order,
        'files':metadata,'expected_rows':{'database':193734,'test':2100},
        'positive_counts':dict(zip(order,counts)),
        'verification':{
            'official_zip_sha256':hashlib.sha256(Path(official).read_bytes()).hexdigest(),
            'official_source':'https://huggingface.co/datasets/Lxyhaha/NUS-WIDE/blob/main/NUS-WIDE.zip',
            'official_source_entrypoint':'https://github.com/NExTplusplus/NUS-WIDE',
            'official_label_sha256':{k:official_files[k] for k in order},
            'method':'unique official positive counts identify columns, then compare complete nonempty joint-label histograms; integer and multi-hot rows checked exactly',
            'joint_label_patterns':len(native),'joint_label_histogram_sha256':histogram_sha(native),
            'joint_label_histogram_equal':True,'nonempty_official_rows':sum(native.values()),
            'image_paths_unique_and_query_database_disjoint':True,
            'limitations':'Statistical recovery of column semantics for these exact pinned files; no independent per-image comparison to official ImageList, which was not supplied. Image/label pairing follows explicit retrieval lists. This is not the official Train/Test split.'
        },
        'nas8_policy':{'snow_positive':'rain_snow=1','snow_negative':'rain_snow=-1',
                       'all_other_nas8_labels':-1,'missing_target_concepts':['nighttime','sports']}}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnostics',type=Path,required=True)
    p.add_argument('--official-gt-zip',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise Blocked('Do not overwrite an existing evidence profile')
    result=verify(a.diagnostics,a.official_gt_zip);dump(a.output,result)
    print('NUS21_PROFILE_VERIFIED',result['verification']['joint_label_patterns'],result['positive_counts']['snow'])

if __name__=='__main__':main()
