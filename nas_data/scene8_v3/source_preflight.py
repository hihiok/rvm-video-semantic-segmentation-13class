#!/usr/bin/env python3
"""Read-only source validation and bounded, image-free NUS evidence export."""
from __future__ import annotations
import argparse, json, re, zipfile
from collections import Counter
from pathlib import Path
from storage import Blocked, dump, sha

# Explicit metadata names only: no images, user files, credentials or features.
def wanted(name):
    name=name.lower()
    if Path(name).suffix not in ('','.txt','.md','.json','.csv','.html','.htm'):return False
    return (name.startswith(('readme','concepts','class_names','classes','label_names'))
            or name in ('database_img.txt','database_label.txt','database_label_onehot.txt',
                        'test_img.txt','test_label.txt','test_label_onehot.txt',
                        'img_tc10.txt','targets_tc10.txt','targets_onehot_tc10.txt',
                        'imagelist.txt','trainimagelist.txt','testimagelist.txt')
            or re.fullmatch(r'labels_(nighttime|sports|snow|soccer|running|swimmers|surf)(_(train|test))?\.txt',name) is not None)

def describe(path,root):
    widths=Counter();tokens=Counter();examples=[];rows=0;blanks=0;numeric=True
    name=path.name.lower()
    free_text=name.startswith(('readme','concepts','class_names','classes','label_names'))
    with path.open(encoding='utf-8-sig',errors='replace') as f:
        for n,line in enumerate(f,1):
            line=line.rstrip('\r\n');rows=n
            if len(examples)<8:examples.append({'row_1based':n,'text':line[:500]})
            if not line.strip():blanks+=1
            if free_text:continue
            parts=line.split();widths[len(parts)]+=1
            if not all(re.fullmatch(r'-?\d+',v) for v in parts):numeric=False
            if numeric:
                for v in parts:
                    if v in tokens or len(tokens)<256:tokens[v]+=1
    return {'path':path.relative_to(root).as_posix(),'bytes':path.stat().st_size,
            'sha256':sha(path),'rows':rows,'blank_rows':blanks,
            'columns_histogram':dict(widths),'all_tokens_integer':numeric if not free_text else None,
            'integer_token_counts_capped_256':dict(tokens) if numeric and not free_text else {},
            'first_rows':examples}

def export_nus_diagnostics(root,output_root):
    root=Path(root).resolve();out=Path(output_root).resolve()
    if not root.is_dir():raise Blocked('NUS image/cache root missing: '+str(root))
    if out==root or root in out.parents or out in root.parents:
        raise Blocked('Diagnostic output must be separate from the source root')
    if out.exists():raise Blocked('Diagnostic output already exists; choose a fresh path')
    out.mkdir(parents=True)
    files=sorted((p for p in root.rglob('*') if p.is_file() and wanted(p.name)),key=str)
    report={'status':'EVIDENCE_ONLY_NOT_TRAINING_GT','source_root':str(root),
            'metadata':[],'missing_core_files':[],'limits':[],
            'interpretation':'Numeric indices/column counts alone do not establish concept names or complete negatives.'}
    budget=32*1024**2;copied=0
    with zipfile.ZipFile(out/'nus_diagnostics.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for path in files:
            if path.is_symlink() or root not in path.resolve().parents:
                raise Blocked('Metadata symlink/outside source root: '+str(path))
            size=path.stat().st_size
            if size>16*1024**2:
                report['limits'].append({'path':path.relative_to(root).as_posix(),'reason':'over_16MiB_file_limit','bytes':size})
                continue
            item=describe(path,root);report['metadata'].append(item)
            if copied+size<=budget:
                z.write(path,'metadata/'+item['path']);copied+=size;item['full_file_in_bundle']=True
            else:
                item['full_file_in_bundle']=False;report['limits'].append({'path':item['path'],'reason':'32MiB_total_copy_budget'})
        names={x['path'].split('/')[-1].lower() for x in report['metadata']}
        report['missing_core_files']=[x for x in ('Concepts81.txt','TrainImagelist.txt','TestImagelist.txt') if x.lower() not in names]
        report['format_detected']='retrieval_numeric_unverified' if 'database_label.txt' in names else 'official_or_unknown'
        report['metadata_bytes_in_bundle']=copied
        z.writestr('nus_format_report.json',json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    dump(out/'nus_format_report.json',report)
    print('NUS_DIAGNOSTICS:',out/'nus_diagnostics.zip',flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nus-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--mir-image-root',type=Path,required=True)
    p.add_argument('--mir-annotation-root',type=Path,required=True)
    p.add_argument('--nus-metadata-root',type=Path)
    a=p.parse_args()
    out=a.output_root.resolve()
    for src in (a.mir_image_root,a.mir_annotation_root,a.nus_metadata_root):
        if src is not None:
            src=src.resolve()
            if out==src or src in out.parents or out in src.parents:
                raise Blocked('Diagnostic output overlaps a source directory')
    from sources import mir_rows,nus_rows
    export_nus_diagnostics(a.nus_root,a.output_root)
    audit={};excluded=[];status={'training_started':False,'manifests_written':False}
    try:
        rows=mir_rows(a.mir_image_root,a.mir_annotation_root,20260910,audit,excluded)
        status['MIR_STATUS']='PASS';status['MIR_IMAGES']=len(rows)
        del rows
        rows=nus_rows(a.nus_root,20260910,audit,excluded,metadata_root=a.nus_metadata_root)
        status.update(status='SOURCE_PREFLIGHT_PASS',NUS_ROWS=len(rows),HUMAN_ACTION_REQUIRED=False)
    except Exception as e:
        status.update(status='BLOCKED',error=str(e),HUMAN_ACTION_REQUIRED=True,
                      action='Return nus_diagnostics.zip plus source_preflight.json and source_audit.json to ChatGPT. No local patch.')
    dump(a.output_root/'source_audit.json',audit)
    dump(a.output_root/'source_preflight.json',status)
    with zipfile.ZipFile(a.output_root/'nus_diagnostics.zip','a',compression=zipfile.ZIP_DEFLATED) as z:
        z.write(a.output_root/'source_audit.json','source_audit.json')
        z.write(a.output_root/'source_preflight.json','source_preflight.json')
    print(json.dumps(status,ensure_ascii=False,indent=2))
    return 0 if status['status']=='SOURCE_PREFLIGHT_PASS' else 2

if __name__=='__main__':raise SystemExit(main())
