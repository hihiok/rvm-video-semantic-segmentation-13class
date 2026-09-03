#!/usr/bin/env python3
import argparse, json, random
from pathlib import Path

LABELS=["night","indoor","rain_snow","office","outdoor","landscape","sports","objective_image"]

def load(p): return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',type=Path,required=True); ap.add_argument('--output-root',type=Path,required=True); ap.add_argument('--per-state',type=int,default=24); ap.add_argument('--seed',type=int,default=20260903); a=ap.parse_args()
    rng=random.Random(a.seed); a.output_root.mkdir(parents=True,exist_ok=True)
    for split in ('train','val','test'):
        rows=load(a.data_root/f'{split}.jsonl'); chosen={}
        for label in LABELS:
            for state in (1,0):
                c=[r for r in rows if int(r['labels'][label])==state]
                if not c: raise SystemExit(f'missing smoke coverage {split}:{label}:{state}')
                for r in rng.sample(c,min(len(c),a.per_state)): chosen[r['image']]=r
        out=list(chosen.values()); rng.shuffle(out)
        with (a.output_root/f'{split}.jsonl').open('w',encoding='utf-8') as f:
            for r in out: f.write(json.dumps(r,ensure_ascii=False)+'\n')
        print(split,len(out))
if __name__=='__main__': main()
