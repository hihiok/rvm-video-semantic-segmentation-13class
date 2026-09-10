"""NAS8 V3 manifest validation and masked metrics. No model dependency."""
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

LABELS = ['night', 'indoor', 'rain_snow', 'office', 'outdoor', 'landscape', 'sports', 'objective_image']
NAMES = ['夜景', '室内', '雨/雪', '办公场景', '户外', '自然风景', '运动', '客观图']
# Fixed before test inference; no tuning on val/test, no 8-way softmax.
PROMPTS = {
    'night': (['a real photograph of a scene at nighttime', 'a night scene photographed after sunset'],
              ['a real photograph of a scene in daytime', 'an indoor scene in ordinary lighting, not a night scene']),
    'indoor': (['a photograph taken inside a room or building', 'an indoor environment'],
               ['a photograph taken in an outdoor environment', 'an open-air exterior scene']),
    'rain_snow': (['a photograph with visible rain or snow', 'a scene showing rainfall, snowfall, or snow on the ground'],
                  ['a photograph with neither rain nor snow present', 'a scene without rain and without snow']),
    'office': (['a photograph of an office workspace or office meeting room', 'a professional office, cubicle, or home office'],
               ['a scene that is not an office workspace', 'a non-office environment']),
    'outdoor': (['a photograph taken in an outdoor environment', 'an open-air exterior scene'],
                ['a photograph taken inside a room or building', 'an indoor environment']),
    'landscape': (['a photograph whose main subject is a natural landscape', 'natural scenery, mountains, forest, or sea as the main subject'],
                  ['a photograph whose main subject is a person, animal, object, or city', 'a cityscape or close-up subject rather than natural scenery']),
    'sports': (['a photograph of sports activity or a sporting event', 'a sports venue, court, playing field, or people doing sports'],
               ['a scene unrelated to sports', 'a non-sports scene or activity']),
    'objective_image': (['an image quality test pattern or resolution chart', 'a computer-generated calibration pattern, color bars, or test chart'],
                        ['a real photograph of a scene rather than an image quality test pattern', 'a real office or room containing monitors, not a test chart image']),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read_manifest(path, split):
    rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    seen, groups = set(), set()
    for r in rows:
        if set(r.get('labels', {})) != set(LABELS):
            raise ValueError('Expected exact V3 eight label dictionary: ' + str(path))
        if any(type(v) is not int or v not in (-1, 0, 1) for v in r['labels'].values()):
            raise ValueError('Labels must be integer -1/0/1')
        if r.get('split') != split or not r.get('group_id'):
            raise ValueError('Missing or inconsistent split/group_id')
        image = str(Path(r['image']).resolve())
        if image in seen or r['group_id'] in groups:
            raise ValueError('Duplicate image or group in ' + str(path))
        if not Path(r['image']).is_absolute() or not Path(r['image']).is_file():
            raise ValueError('Missing or non-absolute image: ' + r['image'])
        if not isinstance(r.get('evidence'), dict):
            raise ValueError('V3 evidence missing')
        seen.add(image)
        groups.add(r['group_id'])
    return rows


def audit(root, split):
    root = Path(root)
    meta = json.loads((root/'summary.json').read_text())
    if meta.get('schema') != 'nas8_source_curation_v3' or meta.get('labels') != LABELS:
        raise ValueError('Not a NAS8 clean V3 dataset; do not use old eight/nine-label manifests')
    if (root/'BLOCKED.json').exists() or not (root/'PREPARED.json').is_file():
        raise ValueError('Dataset preparation incomplete or blocked')
    sets, target = {}, None
    hashes = {'summary.json': sha(root/'summary.json'), 'PREPARED.json': sha(root/'PREPARED.json')}
    for s in ('train', 'val', 'test'):
        rows = read_manifest(root/(s+'.jsonl'), s)
        sets[s] = ({r['group_id'] for r in rows}, {str(Path(r['image']).resolve()) for r in rows})
        hashes[s+'.jsonl'] = sha(root/(s+'.jsonl'))
        if s == split:
            target = rows
    for a, b in [('train','val'), ('train','test'), ('val','test')]:
        if any(sets[a][k] & sets[b][k] for k in (0, 1)):
            raise ValueError('Cross-split group/path overlap: ' + a + '/' + b)
    strict_path = root/(split+'_strict.jsonl')
    strict = read_manifest(strict_path, split)
    hashes[strict_path.name] = sha(strict_path)
    full = {r['image']: r for r in target}
    for r in strict:
        if r['image'] not in full or r['group_id'] != full[r['image']]['group_id']:
            raise ValueError('Strict manifest must be a matching subset')
        for l in LABELS:
            if r['labels'][l] != -1:
                if r['labels'][l] != full[r['image']]['labels'][l]:
                    raise ValueError('Strict/full GT conflict')
                if not any(t in r['evidence'].get(l, '') for t in ('manual:', 'human_review:', 'user_review:')):
                    raise ValueError('Weak evidence leaked into strict GT')
    if not target:
        raise ValueError('Empty selected evaluation split')
    return target, strict, {'schema': meta['schema'], 'split': split, 'hashes': hashes,
                            'training_quality_blockers': meta.get('training_quality_blockers', []),
                            'dataset_human_action_required': meta.get('HUMAN_ACTION_REQUIRED', False),
                            'dataset_status': meta.get('status'), 'images': len(target)}


def metrics(gt, scores):
    gt, scores = np.asarray(gt).reshape(-1, 8), np.asarray(scores).reshape(-1, 8)
    if gt.shape != scores.shape or not np.isin(gt, [-1,0,1]).all():
        raise ValueError('Invalid metric shapes or labels')
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Invalid scores')
    rows = []
    for j, label in enumerate(LABELS):
        known = gt[:,j] != -1
        y, s = gt[known,j], scores[known,j]
        p = s >= 0.5
        pos, neg = int((y==1).sum()), int((y==0).sum())
        tp, fp = int(((y==1)&p).sum()), int(((y==0)&p).sum())
        fn, tn = pos-tp, neg-fp
        supported = pos > 0 and neg > 0
        rows.append(dict(label=label, positive=pos, negative=neg, unknown=int((~known).sum()),
            status='OK' if supported else ('NO_GT' if not len(y) else 'SINGLE_CLASS_GT'),
            tp=tp, fp=fp, fn=fn, tn=tn,
            precision=(tp/(tp+fp) if tp+fp else 0.0) if supported else None,
            recall=tp/pos if pos else None,
            specificity=tn/neg if neg else None,
            f1=(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0) if supported else None,
            accuracy=(tp+tn)/len(y) if supported else None,
            ap=float(average_precision_score(y,s)) if supported else None,
            auc=float(roc_auc_score(y,s)) if supported else None))
    covered = [r for r in rows if r['status']=='OK']
    known = gt != -1
    yp, pp = gt[known], scores[known]>=0.5
    tp, fp, fn = int(((yp==1)&pp).sum()), int(((yp==0)&pp).sum()), int(((yp==1)&~pp).sum())
    summary = dict(covered_labels=[r['label'] for r in covered], covered_count=len(covered),
        missing_or_single_class=[r['label'] for r in rows if r['status']!='OK'],
        macro_f1_covered=float(np.mean([r['f1'] for r in covered])) if covered else None,
        macro_ap_covered=float(np.mean([r['ap'] for r in covered])) if covered else None,
        macro_f1_all8=float(np.mean([r['f1'] for r in covered])) if len(covered)==8 else None,
        macro_ap_all8=float(np.mean([r['ap'] for r in covered])) if len(covered)==8 else None,
        micro_f1_known_pairs=2*tp/(2*tp+fp+fn) if (yp==1).any() and (yp==0).any() else None,
        threshold=0.5, unknown_policy='-1 excluded, never negative')
    return {'per_class':rows,'summary':summary}


def protocol(rows, strict_rows, scores):
    strict_map = {r['image']:r for r in strict_rows}
    strict_gt = [[strict_map[r['image']]['labels'][l] if r['image'] in strict_map else -1 for l in LABELS] for r in rows]
    full_gt = [[r['labels'][l] for l in LABELS] for r in rows]
    result = {'strict':metrics(strict_gt, scores), 'mapped':metrics(full_gt, scores)}
    for source in sorted({r['source'] for r in rows}):
        ix = [i for i,r in enumerate(rows) if r['source']==source]
        result['source_mapped_'+source] = metrics(np.asarray(full_gt)[ix], np.asarray(scores)[ix])
    # Do not let user-defined synthetic pattern negatives hide real-photo night/weather errors.
    ix = [i for i,r in enumerate(rows) if r['labels']['objective_image']==0 and
          ('photo' in r['evidence'].get('objective_image','').lower() or r['source'] in ('mirflickr','nuswide','places365','coco','seg13'))]
    result['real_photo_mapped_subset'] = metrics(np.asarray(full_gt)[ix], np.asarray(scores)[ix])
    return result
