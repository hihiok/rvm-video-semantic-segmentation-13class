#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from pathlib import Path

LABELS = ["night", "indoor", "rain_snow", "office", "outdoor", "landscape", "sports", "objective_image"]


def load(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def underlying_id(r):
    p = Path(r["image"])
    stem = p.stem.lower()
    m = re.search(r"(?<!\d)(\d{12})(?!\d)", stem)
    if m:
        return "coco:" + m.group(1)
    return "path:" + str(p.resolve())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data-root", type=Path, required=True); args = ap.parse_args()
    rows = {s: load(args.data_root / f"{s}.jsonl") for s in ("train","val","test")}
    fail = []
    for split, items in rows.items():
        print(f"{split}: records={len(items)}")
        by_source = defaultdict(int)
        for r in items: by_source[r.get("source", "unknown")] += 1
        print("  by_source:", dict(by_source))
        for l in LABELS:
            p = sum(int(r["labels"][l]) == 1 for r in items); n = sum(int(r["labels"][l]) == 0 for r in items); u = len(items)-p-n
            print(f"  {l}: pos={p} neg={n} unknown={u}")
            if p == 0 or n == 0: fail.append(f"coverage {split}:{l}:pos={p},neg={n}")
    ids = {s: defaultdict(list) for s in rows}
    for s, items in rows.items():
        for r in items: ids[s][underlying_id(r)].append(r)
    for a,b in (("train","val"),("train","test"),("val","test")):
        overlap = set(ids[a]) & set(ids[b])
        print(f"underlying_overlap {a}/{b}: {len(overlap)}")
        if a == "train" and overlap:
            examples = sorted(overlap)[:20]
            fail.append(f"HARD leakage {a}/{b}: count={len(overlap)} examples={examples}")
        elif overlap:
            print(f"WARNING_ONLY {a}/{b} overlap accepted; examples={sorted(overlap)[:10]}")
    if fail:
        print("AUDIT=FAIL")
        for x in fail: print("FAIL_REASON:", x)
        raise SystemExit(2)
    print("AUDIT=PASS")

if __name__ == "__main__": main()
