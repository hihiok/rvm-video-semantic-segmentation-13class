"""Freeze ALL current Rev10 rows. Pick 200 representative review rows first."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import random
from common import LABELS, sha, read, dump, jsonl, valid_labels, load_run

QUOTAS = {'coco': 15, 'places365': 60, 'seg13': 20, 'mirflickr': 35, 'nuswide': 35, '10_scenes': 35}


def preview_select(rows, seed):
    rng = random.Random(seed)
    chosen, seen = [], set()
    # Round robin within each source boosts landscape/sports/rare positives and hard negatives.
    for source, quota in QUOTAS.items():
        pool = [r for r in rows if r['source'] == source]
        rng.shuffle(pool)
        pools = [[r for r in pool if r['labels'][label] == 1] for label in
                 ('landscape', 'sports', 'landscape', 'sports', 'rain_snow', 'office', 'objective_image', 'night')]
        pools += [[r for r in pool if r['labels']['outdoor'] == 1 and r['labels']['landscape'] != 1], pool]
        iters = [iter(p) for p in pools]
        count = 0
        while count < quota:
            added = False
            for it in iters:
                r = next((r for r in it if r['image'] not in seen), None)
                if r is not None:
                    chosen.append(r); seen.add(r['image']); count += 1; added = True
                if count == quota:
                    break
            if not added:
                break
    pool = [r for r in rows if r['image'] not in seen]
    rng.shuffle(pool)
    chosen.extend(pool[:200-len(chosen)])
    if len(chosen) != 200:
        raise ValueError('Need at least 200 distinct images')
    return chosen


def run(a):
    source, out = a.input_root.resolve(), a.output_root.resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError('Output must be a separate sibling of input')
    if out.exists():
        if (out/'plan.json').exists():
            _, plan, _ = load_run(out)
            if plan['input_root'] != str(source) or plan['seed'] != a.seed:
                raise ValueError('Different resume input or seed')
            print('PREPARE_REUSED', plan['count'], flush=True)
            return
        raise ValueError('Incomplete prepare directory; retain it and use a new output')
    if (source/'BLOCKED.json').exists():
        raise ValueError('Blocked source dataset')
    names = ['PREPARED.json', 'summary.json', 'source_audit.json'] + [s+'.jsonl' for s in ('train','val','test')]
    names += [s+'_strict.jsonl' for s in ('val','test') if (source/(s+'_strict.jsonl')).exists()]
    hashes = {n: sha(source/n) for n in names}
    summary = read(source/'summary.json')
    if summary.get('schema') != 'nas8_weather_rev10':
        raise ValueError('Expected the latest Rev10 input')
    import json
    rows, paths, groups, pixels = [], set(), set(), set()
    for split in ('train','val','test'):
        with (source/(split+'.jsonl')).open(encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                valid_labels(r['labels'])
                image = str(Path(r['image']).resolve())
                if r.get('split') != split or not r.get('group_id') or not Path(image).is_file():
                    raise ValueError('Missing image/split/group')
                if image in paths or r['group_id'] in groups or (r.get('rgb_sha256') and r['rgb_sha256'] in pixels):
                    raise ValueError('Duplicate path/group/pixels including cross-split leakage')
                paths.add(image); groups.add(r['group_id'])
                if r.get('rgb_sha256'): pixels.add(r['rgb_sha256'])
                r = dict(r, image=image)
                rows.append(r)
    preview = preview_select(rows, a.seed)
    preview_paths = {r['image'] for r in preview}
    ordered = preview + [r for r in rows if r['image'] not in preview_paths]
    out.mkdir(parents=True, mode=0o700)
    # Full byte fingerprints allow both teachers to verify they saw the identical immutable image.
    items = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, (r, image_hash) in enumerate(zip(ordered, pool.map(sha, [r['image'] for r in ordered]))):
            item = dict(r, id='%07d' % i, image_sha256=image_hash, preview=i < 200)
            items.append(item)
            if i % 10000 == 0: print('FINGERPRINT', i, len(ordered), flush=True)
    jsonl(out/'items.jsonl', items)
    for n, h in hashes.items():
        if sha(source/n) != h: raise ValueError('Input changed during prepare')
    dump(out/'plan.json', {'schema':'nas8_three_votes_v1', 'input_root':str(source), 'seed':a.seed,
         'input_hashes':hashes, 'items_digest':sha(out/'items.jsonl'), 'count':len(items),
         'preview_count':200, 'source_counts':dict(Counter(r['source'] for r in items)),
         'split_counts':dict(Counter(r['split'] for r in items)),
         'preview_source_counts':dict(Counter(r['source'] for r in preview)), 'requested_preview_quotas':QUOTAS,
         'preview_positive_counts':{k:sum(r['labels'][k]==1 for r in preview) for k in LABELS},
         'evaluation_policy':'All splits get pseudo-label sidecars. Original held-out GT stays separately frozen. No pseudo-label accuracy claims.'})
    print('ALL_IMAGES_PREPARED', len(items), 'PREVIEW', 200, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20260917)
    run(p.parse_args())
