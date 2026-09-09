#!/usr/bin/env python3
"""Download/verify MIRFLICKR-25K v080 and prepare conservative NAS manifests.
No torch, GPU, original FSD, other datasets, or training jobs are used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import stat
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from PIL import Image, ImageDraw, ImageOps

LABELS = ['night', 'indoor', 'rain_snow', 'office', 'outdoor', 'landscape', 'sports', 'objective_image']
URL_BASE = 'https://press.liacs.nl/mirflickr/mirflickr25k.v3b/'
ARCHIVES = [('mirflickr25k.zip', 'a23d0a8564ee84cda5622a6c2f947785', 'images'),
            ('mirflickr25k_annotations_v080.zip', None, 'annotations_v080')]
DEFAULT_ROOT = '/data/pub1/z00919662/segmentation/datasets/MIRFLICKR25K'
PREP_VERSION = 'mir25k_nas8_v1'


class PreparationError(RuntimeError):
    pass


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def write_jsonl(path, rows):
    with Path(path).open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def rain_snow_or(rain, snow):
    if rain not in (-1, 0, 1) or snow not in (-1, 0, 1):
        raise ValueError('Expected {-1,0,1}')
    if rain == 1 or snow == 1:
        return 1
    return 0 if rain == 0 and snow == 0 else -1


def safe_zip_infos(z):
    result, seen, total = [], set(), 0
    for info in z.infolist():
        p = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        if (not info.filename or '\\' in info.filename or p.is_absolute() or
                '..' in p.parts or any(':' in part for part in p.parts) or
                stat.S_ISLNK(mode) or info.flag_bits & 1):
            raise PreparationError('Unsafe/encrypted ZIP member: ' + info.filename)
        key = str(p)
        if key in seen:
            raise PreparationError('Duplicate ZIP member: ' + key)
        seen.add(key)
        total += info.file_size
        if total > 20 * 1024 ** 3 or info.file_size > 1024 ** 3:
            raise PreparationError('Unexpected ZIP expansion size')
        result.append(info)
    if not result:
        raise PreparationError('Empty ZIP')
    return result


def verify_archive(path, md5=None):
    if md5 and digest(path, 'md5') != md5:
        raise PreparationError('Official image MD5 mismatch: ' + str(path))
    try:
        with zipfile.ZipFile(path) as z:
            infos = safe_zip_infos(z)
            bad = z.testzip()
            if bad:
                raise PreparationError('ZIP CRC failure: ' + bad)
            return {'sha256': digest(path), 'md5': digest(path, 'md5'),
                    'bytes': Path(path).stat().st_size, 'members': len(infos),
                    'uncompressed_bytes': sum(x.file_size for x in infos),
                    'official_md5_checked': md5 is not None}
    except zipfile.BadZipFile as e:
        raise PreparationError('Not a valid ZIP: ' + str(path)) from e


def acquire(path, offline, insecure):
    if path.is_file():
        return
    if offline:
        raise PreparationError('MANUAL_DOWNLOAD_REQUIRED: missing ' + str(path))
    if shutil.which('curl') is None:
        raise PreparationError('MANUAL_DOWNLOAD_REQUIRED: curl is not installed')
    partial = Path(str(path) + '.part')
    cmd = ['curl', '--fail', '--location', '--retry', '2', '--retry-delay', '5',
           '--connect-timeout', '30', '--max-time', '7200', '--speed-time', '120',
           '--speed-limit', '1024', '--proto', '=https', '--proto-redir', '=https',
           '--continue-at', '-', '--output', str(partial), URL_BASE + path.name]
    if insecure:
        cmd.insert(1, '--insecure')
    # Proxy is inherited, never printed. Keep curl progress/error log private.
    log = path.parent / (path.name + '.curl.log')
    with log.open('ab') as f:
        os.chmod(log, 0o600)
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None:
                print('DOWNLOAD', path.name, 'partial_bytes=', partial.stat().st_size if partial.exists() else 0, flush=True)
                time.sleep(10)
        except BaseException:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
            raise
    if proc.returncode != 0:
        raise PreparationError('MANUAL_DOWNLOAD_REQUIRED: curl exit=%d for %s; partial kept; private log=%s' %
                               (proc.returncode, path.name, log))
    if not partial.is_file():
        raise PreparationError('Download did not create the expected file')
    # Validate before promoting; corrupted files are preserved for inspection.
    expected = next(x[1] for x in ARCHIVES if x[0] == path.name)
    verify_archive(partial, expected)
    os.replace(partial, path)


def extract_once(archive, dest, metadata):
    marker = dest / '_MIR_PREP_COMPLETE.json'
    if dest.exists():
        if not marker.is_file():
            raise PreparationError('Existing extraction lacks completion marker; do not overwrite: ' + str(dest))
        saved = json.loads(marker.read_text(encoding='utf-8'))
        if saved.get('archive_sha256') != metadata['sha256']:
            raise PreparationError('Archive differs from existing extraction: ' + str(dest))
        return
    staging = dest.with_name(dest.name + '.extracting')
    if staging.exists():
        raise PreparationError('Incomplete extraction preserved; move it aside before rerun: ' + str(staging))
    staging.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        infos = safe_zip_infos(z)
        for info in infos:
            target = staging.joinpath(*PurePosixPath(info.filename).parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, target.open('xb') as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
    write_json(staging / '_MIR_PREP_COMPLETE.json', {'archive_sha256': metadata['sha256'], 'archive': archive.name})
    staging.rename(dest)


def parse_id_list(path, valid_ids):
    out = set()
    for line_no, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        if not re.fullmatch(r'[0-9]+', line):
            raise PreparationError('Unexpected annotation format: %s:%d; expected one 1-based image ID per line' % (path, line_no))
        idx = int(line)
        if idx not in valid_ids:
            raise PreparationError('Annotation image ID out of range: %s:%d:%d' % (path, line_no, idx))
        if idx in out:
            raise PreparationError('Duplicate annotation image ID: %s:%d' % (path, idx))
        out.add(idx)
    return out


def read_annotations(root, valid_ids):
    # Read only the manual annotation archive, never meta/tags from the image ZIP.
    night_paths = [p for p in root.rglob('*') if p.is_file() and p.name.lower() == 'night.txt']
    if len(night_paths) != 1:
        raise PreparationError('Expected one manual night.txt; inspect annotations_v080 directory')
    directory = night_paths[0].parent
    readmes = [p for p in root.rglob('*') if p.is_file() and p.name.lower().startswith('readme')]
    if not readmes:
        raise PreparationError('Manual annotation README missing; verify archive version')
    sets, provenance = {}, {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() != '.txt' or path.name.lower().startswith('readme'):
            continue
        name = path.stem.lower()
        if not re.fullmatch(r'[a-z][a-z0-9_]*', name) or name in sets:
            raise PreparationError('Unexpected/duplicate manual concept file: ' + str(path))
        sets[name] = parse_id_list(path, valid_ids)
        provenance[name] = {'file': str(path.resolve()), 'sha256': digest(path), 'positive_count': len(sets[name])}
    if not {'night', 'indoor'} <= set(sets):
        raise PreparationError('Manual night and indoor concepts are required')
    for name, ids in sets.items():
        base = re.sub(r'_r\d+$', '', name)
        if base != name and (base not in sets or not ids <= sets[base]):
            raise PreparationError('Relevance IDs not a subset of potential IDs: ' + name)
    return sets, {'files': provenance, 'readmes': [str(p.resolve()) for p in readmes]}


def project_labels(idx, annotations):
    labels = dict.fromkeys(LABELS, -1)
    evidence = {}
    for name in ('night', 'indoor'):
        potential = annotations[name]
        relevant = annotations.get(name + '_r1')
        if idx not in potential:
            labels[name] = 0
            evidence[name] = 'absent_from_complete_manual_potential:' + name
        elif relevant is None or idx in relevant:
            labels[name] = 1
            evidence[name] = 'manual_positive:' + (name + '_r1' if relevant is not None else name)
        else:
            evidence[name] = 'potential_but_not_relevant:unknown'
    # No outdoor=complement(indoor), no landscape=water/plant, no objective=0.
    # No rain/snow annotation exists here: OR(-1,-1) must remain unknown.
    labels['rain_snow'] = rain_snow_or(-1, -1)
    return labels, evidence


def scan_images(root, annotations, expected_count=25000):
    paths = {}
    for p in root.rglob('*'):
        if p.is_file():
            m = re.fullmatch(r'im([1-9][0-9]*)\.jpg', p.name, re.I)
            if m:
                idx = int(m.group(1))
                if idx in paths:
                    raise PreparationError('Duplicate imID path: ' + str(idx))
                paths[idx] = p
    expected = set(range(1, expected_count + 1))
    if set(paths) != expected:
        raise PreparationError('Expected im1.jpg..im%d.jpg exactly; found=%d missing=%s extra=%s' %
                               (expected_count, len(paths), sorted(expected-set(paths))[:20], sorted(set(paths)-expected)[:20]))
    rows = []
    for n, idx in enumerate(sorted(paths), 1):
        path = paths[idx]
        # Fully decode, check corruption, record raw and oriented sizes, and hash RGB pixels.
        with Image.open(path) as im:
            raw_w, raw_h = im.size
            rgb = ImageOps.exif_transpose(im).convert('RGB')
            rgb.load()
            w, h = rgb.size
            ph = hashlib.sha256(('%dx%d:' % (w, h)).encode() + rgb.tobytes()).hexdigest()
        y, evidence = project_labels(idx, annotations)
        rows.append({'sample_id': 'mirflickr25k:%d' % idx, 'image_id': idx,
                     'image': str(path.resolve()), 'source': 'mirflickr25k_v080',
                     'labels': y, 'label_evidence': evidence,
                     'original_labels': {key: int(idx in ids) for key, ids in sorted(annotations.items())},
                     'width': w, 'height': h, 'raw_width': raw_w, 'raw_height': raw_h,
                     'sha256_file': digest(path), 'sha256_rgb': ph, 'group_id': 'rgb:' + ph})
        if n % 1000 == 0:
            print('IMAGE_AUDIT', n, '/', expected_count, flush=True)
    return rows


def assign_splits(rows, seed):
    groups = defaultdict(list)
    for r in rows:
        groups[r['group_id']].append(r)
    conflicts = []
    for gid, members in groups.items():
        for label in ('night', 'indoor'):
            values = {r['labels'][label] for r in members} - {-1}
            if len(values) > 1:
                conflicts.append({'group_id': gid, 'label': label, 'ids': [r['image_id'] for r in members]})
                for r in members:
                    r['labels'][label] = -1
                    r['label_evidence'][label] = 'duplicate_manual_label_conflict:unknown'
    strata = defaultdict(list)
    for gid, members in groups.items():
        key = tuple(tuple(sorted({m['labels'][l] for m in members})) for l in ('night', 'indoor'))
        strata[key].append(gid)
    rng = random.Random(seed)
    splits = {s: [] for s in ('train', 'val', 'test')}
    for key in sorted(strata):
        ids = sorted(strata[key]); rng.shuffle(ids)
        # Custom 80/10/10 split, stratified jointly by night/indoor, identical pixels grouped.
        nv = max(1, int(round(len(ids) * .1))) if len(ids) >= 3 else 0
        nt = nv
        for k, gid in enumerate(ids):
            s = 'val' if k < nv else 'test' if k < nv + nt else 'train'
            for row in groups[gid]:
                row['split'] = s
                splits[s].append(row)
    for s in splits:
        splits[s].sort(key=lambda r: r['image_id'])
    idsets = {s: {r['group_id'] for r in v} for s, v in splits.items()}
    overlaps = {'%s_%s' % (a,b): len(idsets[a] & idsets[b])
                for a,b in [('train','val'),('train','test'),('val','test')]}
    if any(overlaps.values()):
        raise PreparationError('Internal duplicate grouping failed')
    return splits, {'groups': len(groups), 'same_pixel_duplicate_records': len(rows)-len(groups),
                    'cross_split_identical_rgb_overlap': overlaps, 'conflicts': conflicts,
                    'limits': 'Exact file/pixel duplicates only; re-encoded/near duplicates and overlaps with old datasets are NOT certified absent.'}


def counts(rows):
    return {l: {state: sum(r['labels'][l] == value for r in rows)
                for state,value in [('positive',1),('negative',0),('unknown',-1)]} for l in LABELS}


def quantile(values, q):
    a = sorted(values)
    if not a:
        return None
    pos = (len(a)-1) * q; lo = int(pos); hi = min(lo+1,len(a)-1)
    return round(a[lo] + (a[hi]-a[lo])*(pos-lo), 2)


def resolution_report(rows):
    stats = {name: {'p10': quantile([r[key] for r in rows], .1),
                    'p50': quantile([r[key] for r in rows], .5),
                    'p90': quantile([r[key] for r in rows], .9)}
             for name,key in [('width','width'),('height','height')]}
    subgroups = {'all_train': rows}
    for label in ('night','indoor'):
        for val in (0,1):
            subgroups['%s_%d' % (label,val)] = [r for r in rows if r['labels'][label] == val]
    candidates = []
    for w,h in [(256,144),(320,180),(384,216),(512,288),(640,360)]:
        rates = {k: sum(min(w/r['width'],h/r['height'])>1.000001 for r in rr)/len(rr)
                 for k,rr in subgroups.items() if len(rr)>=50 or k=='all_train'}
        candidates.append({'width': w, 'height': h, 'letterbox_upscale_fraction': rates,
                           'native_size_gate_pass': all(v <= .1 for v in rates.values())})
    good = [c for c in candidates if c['native_size_gate_pass']]
    upper = [good[-1]['width'],good[-1]['height']] if good else None
    return {'scope':'MIRFLICKR custom TRAIN only; not final combined training dataset', 'count':len(rows),
            'dimensions': stats, 'portrait_fraction':sum(r['height']>r['width'] for r in rows)/len(rows),
            'candidates_16x9':candidates, 'native_resolution_upper_bound_wh':upper,
            'policy':'Largest candidate with <=10% upscaled images in all_train and each night/indoor state with >=50 samples, assuming letterbox. This is an upper bound, NOT an accuracy-optimal training size. Recompute after merging sources; compare smaller inputs on validation. No original image was resized.'}


def make_gallery(rows, output, seed, limit=48):
    rng = random.Random(seed); ordered = list(rows); rng.shuffle(ordered)
    chosen = []; used = set()
    for label in ('night','indoor'):
        for value in (1,0,-1):
            for r in [x for x in ordered if x['labels'][label]==value][:8]:
                if r['image_id'] not in used:
                    chosen.append(r); used.add(r['image_id'])
    for r in ordered:
        if len(chosen) >= limit: break
        if r['image_id'] not in used:
            chosen.append(r); used.add(r['image_id'])
    target = output/'review'; target.mkdir()
    cards=[]
    for r in chosen[:limit]:
        # Thumbnails are NEW audit outputs; source files and annotations remain untouched.
        with Image.open(r['image']) as image:
            thumb = ImageOps.contain(ImageOps.exif_transpose(image).convert('RGB'), (640,400))
        canvas = Image.new('RGB',(720,440+24*(len(LABELS)+3)),(248,248,248))
        canvas.paste(thumb,((720-thumb.width)//2,20))
        d=ImageDraw.Draw(canvas)
        d.text((24,430), 'im%d | original %dx%d | %s' % (r['image_id'],r['width'],r['height'],r['split']), fill='black')
        d.text((24,454), 'LABEL                         GT (1 positive / 0 negative / ? unknown)',fill='black')
        for j,l in enumerate(LABELS):
            d.text((24,478+j*24),l,fill='black')
            d.text((280,478+j*24),'?' if r['labels'][l]==-1 else str(r['labels'][l]),fill='black')
        name='im%d.jpg'%r['image_id']; canvas.save(target/name, quality=92)
        cards.append('<figure><img loading="lazy" src="%s"><figcaption>%s</figcaption></figure>'%(name,html.escape(r['image'])))
    (target/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>MIRFLICKR label audit</title><style>body{font-family:sans-serif}figure{display:inline-block;vertical-align:top;width:720px;margin:12px}img{max-width:100%}figcaption{overflow-wrap:anywhere}</style><h1>Manual annotation audit (not model predictions)</h1>'+''.join(cards),encoding='utf-8')
    with (output/'review_samples.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['image_id','image','split']+LABELS)
        for r in chosen[:limit]:w.writerow([r['image_id'],r['image'],r['split']]+[r['labels'][l] for l in LABELS])


def prepare(root, output, seed, archive_metadata, expected_count=25000):
    ann, provenance = read_annotations(root/'raw'/'annotations_v080', set(range(1, expected_count + 1)))
    rows = scan_images(root/'raw'/'images', ann, expected_count)
    splits, audit = assign_splits(rows, seed)
    # Train/val/test need only the TWO mapped concepts, not fabricated eight-class coverage.
    for split, rr in splits.items():
        c=counts(rr)
        for label in ('night','indoor'):
            if c[label]['positive']==0 or c[label]['negative']==0:
                raise PreparationError('Insufficient %s supervision in %s' % (label,split))
    for name, rr in splits.items():
        write_jsonl(output/(name+'.jsonl'),rr)
        with (output/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['image_id','image','width','height','group_id']+LABELS)
            for r in rr:w.writerow([r[k] for k in ['image_id','image','width','height','group_id']]+[r['labels'][l] for l in LABELS])
    write_jsonl(output/'original_manual_labels.jsonl',[{'image_id':r['image_id'],'original_labels':r['original_labels']} for r in rows])
    write_json(output/'annotation_audit.json',provenance)
    write_json(output/'duplicate_audit.json',audit)
    write_json(output/'resolution_audit.json',resolution_report(splits['train']))
    summary = {'status':'PREPARED_MIRFLICKR25K','prep_version':PREP_VERSION,'num_images':len(rows),
               'annotation_concepts':sorted(ann),'labels':LABELS,'supervised_labels':['night','indoor'],
               'expected_unknown_labels':[l for l in LABELS if l not in ('night','indoor')],
               'split_policy':'Custom 80/10/10 by identical decoded RGB groups, night/indoor joint strata; NOT official benchmark split',
               'seed':seed,'splits':{s:{'images':len(rr),'labels':counts(rr)} for s,rr in splits.items()},
               'archives':archive_metadata,'annotation_policy':'positive=_r1 if present, else potential; negative=outside complete manual potential list; potential-only outside _r1=unknown',
               'night_negative_meaning':'not-night per manual potential concept; NOT evidence for Day or Outdoor',
               'READY_FOR_8LABEL_TRAINING':False,'HUMAN_ACTION_REQUIRED':False,
               'warnings':['Only a source dataset is prepared. No training started or existing training modified.',
                           'No rain/snow/office/outdoor/landscape/sports/objective labels are invented.',
                           'Native annotation quality and product-semantic fit still require review.',
                           'Final combined-dataset leakage audit and train-resolution audit are still required.',
                           'Preserve original license/creator metadata; this task does not certify commercial permissions.']}
    write_json(output/'dataset_summary.json',summary)
    make_gallery(rows,output,seed)
    write_json(output/'PREPARED.json',{'version':PREP_VERSION,'seed':seed,'archives':archive_metadata})
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,default=Path(DEFAULT_ROOT))
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--offline',action='store_true',help='Use existing ZIPs only; never access network')
    p.add_argument('--insecure',action='store_true',help='Explicit corporate TLS-interception bypass for curl only')
    p.add_argument('--download-only',action='store_true')
    p.add_argument('--seed',type=int,default=20260909)
    args=p.parse_args();root=args.dataset_root.resolve()
    out=(args.output_dir or root/'derived'/PREP_VERSION).resolve()
    if out == root or out == root/'raw' or root/'raw' in out.parents or root/'downloads' in out.parents or root in out.parents and out.parent == root:
        raise PreparationError('Output must be isolated from raw/download directories (default derived/ is safe)')
    root.mkdir(parents=True,exist_ok=True);(root/'downloads').mkdir(exist_ok=True)
    metadata={}
    try:
        for name,md5,kind in ARCHIVES:
            archive=root/'downloads'/name
            acquire(archive,args.offline,args.insecure)
            metadata[name]=verify_archive(archive,md5)
        write_json(root/'archive_audit.json',metadata)
        if args.download_only:
            print('STATUS: ARCHIVES_VERIFIED\nHUMAN_ACTION_REQUIRED: NO');return
        # Leave existing derived output untouched. Use --output-dir for a new build.
        if out.exists():
            marker=out/'PREPARED.json'
            if marker.is_file():
                old=json.loads(marker.read_text(encoding='utf-8'))
                if old == {'version':PREP_VERSION,'seed':args.seed,'archives':metadata}:
                    required=['train.jsonl','val.jsonl','test.jsonl','dataset_summary.json','resolution_audit.json']
                    if all((out/n).is_file() for n in required):
                        print('STATUS: ALREADY_PREPARED\nOUTPUT: %s\nHUMAN_ACTION_REQUIRED: NO'%out);return
            raise PreparationError('Existing derived output preserved. Choose a new --output-dir: '+str(out))
        for name,_,kind in ARCHIVES:
            needed=metadata[name]['uncompressed_bytes'] + 1024**3
            if not (root/'raw'/kind).exists() and shutil.disk_usage(root).free < needed:
                raise PreparationError('Insufficient disk space for safe extraction: '+name)
            extract_once(root/'downloads'/name,root/'raw'/kind,metadata[name])
        out.mkdir(parents=True)
        prepare(root,out,args.seed,metadata)
        print('OUTPUT:',out,'\nSTATUS: PREPARED_MIRFLICKR25K\nHUMAN_ACTION_REQUIRED: NO',flush=True)
    except (PreparationError,OSError,ValueError,zipfile.BadZipFile) as e:
        report={'STATUS':'BLOCKED','HUMAN_ACTION_REQUIRED':True,'reason':str(e),
                'download_files':[{'url':URL_BASE+n,'save_to':str(root/'downloads'/n)} for n,_,_ in ARCHIVES],
                'instruction':'Download missing official ZIP(s) manually if needed; keep exact names. Do not delete originals or patch code. Rerun with --offline after uploading ZIPs. For code/layout failures return traceback and archive inventory.'}
        write_json(root/'PREPARE_BLOCKED.json',report)
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
        raise


if __name__ == '__main__':
    main()
