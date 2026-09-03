#!/usr/bin/env python3
"""Finalize generated manifests without touching any source dataset.

Places365 and SEG13 are valuable for scene labels but are intentionally not used
as objective-image negatives: otherwise their very large record counts swamp the
Computer_synthesized positive class. Objective negatives remain from sampled
COCO photos and non-objective 10_scenes classes.
"""
import argparse, json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',type=Path,required=True); a=ap.parse_args()
    stats={}
    for split in ('train','val','test'):
        p=a.data_root/f'{split}.jsonl'; rows=[]; changed=0
        for line in p.read_text(encoding='utf-8').splitlines():
            if not line.strip(): continue
            r=json.loads(line)
            if r.get('source') in {'places365','seg13'} and int(r['labels'].get('objective_image',-1))==0:
                r['labels']['objective_image']=-1; changed+=1
            rows.append(r)
        with p.open('w',encoding='utf-8') as f:
            for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
        stats[split]={'records':len(rows),'objective_negatives_removed_from_places_seg':changed}
    out=a.data_root/'finalize_manifest_summary.json'; out.write_text(json.dumps(stats,indent=2),encoding='utf-8')
    print(json.dumps(stats,indent=2))
if __name__=='__main__': main()
