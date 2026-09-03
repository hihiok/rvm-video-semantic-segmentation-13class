# Continue NAS 9-label UltraFace-slim training after leakage audit policy clarification

This file SUPERSEDES the previous leakage stop rule in the older runbooks.

## User-approved leakage policy

The user explicitly accepts overlap between validation and test.

New hard rule:

- TRAIN vs VAL underlying duplicate image: HARD FAIL / STOP.
- TRAIN vs TEST underlying duplicate image: HARD FAIL / STOP.
- VAL vs TEST duplicate image: ALLOWED. Report as WARNING ONLY. Do NOT stop.
- Duplicate images inside the same split: ALLOWED. Report only.
- COCO and SEG13 derivative copies of the same image inside the same split: ALLOWED.

Important reporting caveat: because per-class thresholds are calibrated on validation, val/test overlap means final test metrics are not a strictly independent generalization estimate and may be optimistic. Report this warning, but continue training because the user explicitly accepts it.

The previously observed 2037 val/test COCO-vs-SEG derivative overlaps are therefore NOT a blocker.

## 1. Repository update rule — mandatory before doing anything else

Repository:

```text
https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git
```

Branch:

```text
agent/nas-ultraface-slim-9label-v1-fullrun
```

Expected working copy:

```bash
export REPO_URL="https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git"
export BRANCH="agent/nas-ultraface-slim-9label-v1-fullrun"
export PROJECT_ROOT="/data/pub1/z00919662/segmentation/nas-ultraface-slim-9label-v1-fullrun"
```

Use exactly this update procedure:

```bash
set -euo pipefail

cd "${PROJECT_ROOT}"

# HARD FAIL if local code was modified. Do not stash, reset, commit, or patch it.
if [[ -n "$(git status --porcelain)" ]]; then
  echo "HARD_FAIL: local repository has modifications"
  git status --short
  echo "HUMAN_ACTION_REQUIRED: YES"
  exit 2
fi

# Fetch only the approved branch.
git fetch origin "${BRANCH}"

# Checkout approved branch.
git checkout "${BRANCH}"

# Fast-forward only. Never merge divergent local work and never force reset.
git merge --ff-only "origin/${BRANCH}"

LOCAL_HEAD="$(git rev-parse HEAD)"
REMOTE_HEAD="$(git rev-parse "origin/${BRANCH}")"
CURRENT_BRANCH="$(git branch --show-current)"

echo "CURRENT_BRANCH=${CURRENT_BRANCH}"
echo "LOCAL_HEAD=${LOCAL_HEAD}"
echo "REMOTE_HEAD=${REMOTE_HEAD}"

if [[ "${CURRENT_BRANCH}" != "${BRANCH}" ]]; then
  echo "HARD_FAIL: wrong branch"
  echo "HUMAN_ACTION_REQUIRED: YES"
  exit 2
fi

if [[ "${LOCAL_HEAD}" != "${REMOTE_HEAD}" ]]; then
  echo "HARD_FAIL: local HEAD does not equal origin branch HEAD"
  echo "HUMAN_ACTION_REQUIRED: YES"
  exit 2
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "HARD_FAIL: working tree became dirty after update"
  git status --short
  echo "HUMAN_ACTION_REQUIRED: YES"
  exit 2
fi
```

Repository policy:

- Do NOT modify source code locally.
- Do NOT use `git reset --hard` to hide local changes.
- Do NOT use `git stash` to hide local changes.
- Do NOT create local commits.
- Do NOT switch to another branch.
- Do NOT cherry-pick patches.
- Do NOT edit Python/config/MD to bypass an error.
- Any required code change must first be made in GitHub by ChatGPT/user, then CodeAgent fetches and fast-forwards to it.

## 2. Verify prepared files

```bash
cd "${PROJECT_ROOT}"
python -m compileall -q nas_scene_multilabel
python nas_scene_multilabel/tools/test_places_folder_mapping.py
```

Required:

```text
PLACES_FOLDER_MAPPING_TEST=PASS
```

If mapping test or compile check fails: HARD FAIL.

## 3. Dataset and manifest paths

Use existing data only:

```bash
export COCO_ROOT="/data/pub1/z00919662/segmentation/datasets/coco"
export PLACES_ROOT="/data/pub1/z00919662/segmentation/datasets/places365"
export SEG_ROOT="/data/pub1/z00919662/segmentation/datasets/COCO_ADE_13cls_16x9_640x360"
export LABEL_ROOT="/data/pub1/z00919662/segmentation/datasets/NAS_9label_multilabel_manifests"
export OUTPUT_ROOT="/data/pub1/z00919662/segmentation/nas_scene_tagging/ultraface_slim_9label_v1"
```

Do NOT download any dataset.
Do NOT alter source images/masks/annotations.
Do NOT copy images into LABEL_ROOT.

Existing manifests from the previous successful build may be reused if all four files exist and are non-empty:

```bash
test -s "${LABEL_ROOT}/train.jsonl"
test -s "${LABEL_ROOT}/val.jsonl"
test -s "${LABEL_ROOT}/test.jsonl"
test -s "${LABEL_ROOT}/dataset_summary.json"
```

If missing/corrupt, regenerate using the prepared `prepare_dataset.py`; do not change its code.

## 4. Updated leakage audit — train isolation only

### 4.1 Exact absolute-path overlap

TRAIN must not share exact paths with VAL or TEST.
VAL/TEST exact overlap is allowed but should be reported.

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path('/data/pub1/z00919662/segmentation/datasets/NAS_9label_multilabel_manifests')
sets = {}
for s in ['train','val','test']:
    sets[s] = {
        json.loads(x)['image']
        for x in (root/f'{s}.jsonl').read_text(encoding='utf-8').splitlines()
        if x.strip()
    }

for a,b in [('train','val'),('train','test'),('val','test')]:
    overlap = sets[a] & sets[b]
    print(f'{a}/{b} exact_path_overlap={len(overlap)}')
    if a == 'train' and overlap:
        raise SystemExit(f'HARD_FAIL exact path leakage: {a}/{b} count={len(overlap)}')
    if b == 'train' and overlap:
        raise SystemExit(f'HARD_FAIL exact path leakage: {a}/{b} count={len(overlap)}')
PY
```

### 4.2 Underlying COCO image ID overlap across COCO and SEG13 derivative paths

Use filename numeric stem to identify COCO-derived images when possible.

TRAIN must be disjoint from both VAL and TEST at underlying COCO-image level.
VAL and TEST overlap is allowed.

```bash
python - <<'PY'
import json, re
from collections import defaultdict
from pathlib import Path

root = Path('/data/pub1/z00919662/segmentation/datasets/NAS_9label_multilabel_manifests')

def coco_id(rec):
    # COCO/SEG derivatives generally preserve the 12-digit COCO stem.
    stem = Path(rec['image']).stem
    m = re.search(r'(\d{12})$', stem)
    if not m:
        return None
    # Restrict this audit to records that can plausibly be COCO-derived.
    src = rec.get('source', '')
    path = rec.get('image', '')
    if src not in {'coco2017','seg13'} and '/coco/' not in path and 'COCO_ADE_13cls' not in path:
        return None
    return m.group(1)

ids = defaultdict(lambda: defaultdict(list))
for split in ['train','val','test']:
    for line in (root/f'{split}.jsonl').read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        cid = coco_id(r)
        if cid:
            ids[split][cid].append((r.get('source',''), r['image']))

train_ids = set(ids['train'])
val_ids = set(ids['val'])
test_ids = set(ids['test'])

train_val = train_ids & val_ids
train_test = train_ids & test_ids
val_test = val_ids & test_ids

print('UNDERLYING_COCO_ID_OVERLAP')
print('train/val =', len(train_val))
print('train/test =', len(train_test))
print('val/test =', len(val_test), '(ALLOWED WARNING ONLY)')

if train_val:
    ex = next(iter(train_val))
    print('example train/val id=', ex)
    print('train=', ids['train'][ex][:3])
    print('val=', ids['val'][ex][:3])
    raise SystemExit(f'HARD_FAIL underlying COCO leakage train/val: {len(train_val)}')

if train_test:
    ex = next(iter(train_test))
    print('example train/test id=', ex)
    print('train=', ids['train'][ex][:3])
    print('test=', ids['test'][ex][:3])
    raise SystemExit(f'HARD_FAIL underlying COCO leakage train/test: {len(train_test)}')

if val_test:
    ex = next(iter(val_test))
    print('WARNING: val/test overlap accepted by user')
    print('example id=', ex)
    print('val=', ids['val'][ex][:3])
    print('test=', ids['test'][ex][:3])
PY
```

Acceptance criteria:

```text
train/val underlying overlap = 0
train/test underlying overlap = 0
val/test underlying overlap = any value allowed
```

The previously reported ~2037 val/test overlaps are accepted and MUST NOT trigger HUMAN_ACTION_REQUIRED.

## 5. Coverage audit

All nine labels still need positive and negative supervision in train, val and test.

If any split/class has zero positive or zero negative supervision: HARD FAIL.

The already-reported condition that all nine classes have positive and negative supervision in all splits is acceptable; re-check it from `dataset_summary.json` before training.

## 6. Failure / stop policy

### HARD FAIL — immediately stop and output `HUMAN_ACTION_REQUIRED: YES`

Stop only for one of these:

1. Git working tree is dirty before repository update.
2. Git fetch/checkout/ff-only update fails.
3. Local branch != approved branch.
4. Local HEAD != `origin/agent/nas-ultraface-slim-9label-v1-fullrun` HEAD after update.
5. Prepared Python compile/mapping test fails.
6. Any required source dataset root is missing.
7. Manifest generation fails because of real unsupported dataset structure/code bug.
8. Any class in train/val/test has zero positive or zero negative supervision.
9. TRAIN overlaps VAL at exact-image or identifiable underlying COCO-image level.
10. TRAIN overlaps TEST at exact-image or identifiable underlying COCO-image level.
11. No usable CUDA environment exists.
12. No genuinely free GPU exists.
13. Smoke training fails for a reason other than batch-size OOM.
14. Batch-size OOM persists even after trying 256 -> 128 -> 64 -> 32.
15. Formal training crashes and cannot resume from an existing valid `last.pth` without source-code changes.
16. ONNX export/check fails and requires a code change.
17. Any real code/config change is required to proceed.

When HARD FAIL occurs:

- Do not patch locally.
- Do not alter manifests to hide the failure.
- Do not change class mappings, thresholds, input resolution, model width, or loss.
- Report exact command, traceback, file/line where applicable, observed paths/state, and the minimal action needed from the user/ChatGPT.

### WARNING ONLY — continue automatically; DO NOT stop

The following are NOT blockers:

1. VAL overlaps TEST, including COCO-vs-SEG derivative duplicates.
2. The previously observed 2037 val/test duplicate COCO images.
3. Same underlying image appears twice within the same split through COCO and SEG13.
4. Same source image contributes supervision for different known labels within the same split.
5. Test metrics are not strictly independent because of val/test overlap; report this caveat.
6. Batch size 256 OOMs but a smaller allowed batch size works.

## 7. Continue execution if train isolation passes

Do NOT wait for user confirmation if:

```text
train/val overlap = 0
train/test overlap = 0
all 9 labels have positive+negative supervision
mapping test = PASS
repository is clean and synchronized
```

Continue automatically with the existing full-run settings:

1. choose one free GPU;
2. 1-epoch smoke at 224x224, base_channel=16;
3. if batch 256 OOMs, try 128/64/32 only;
4. if smoke passes, formal 60-epoch training;
5. validation per-class threshold calibration;
6. test evaluation with those thresholds;
7. ONNX export + `onnx.checker.check_model`;
8. final report.

Use the prepared `train.py` CLI with `--data-root`; do not invent `--amp` or separate manifest CLI args.

## 8. Final report additions

In addition to the normal model/training metrics, include:

```text
REPOSITORY_SYNC:
branch:
local_head:
remote_head:
working_tree_clean: YES/NO

LEAKAGE_POLICY:
train_vs_val_must_be_zero: YES
train_vs_test_must_be_zero: YES
val_vs_test_allowed: YES

LEAKAGE_AUDIT:
exact train/val:
exact train/test:
exact val/test:
underlying COCO train/val:
underlying COCO train/test:
underlying COCO val/test:

VAL_TEST_OVERLAP_WARNING:
The user explicitly accepts val/test overlap. Because thresholds are calibrated on validation, test metrics are not strictly independent and may be optimistic.
```

If train isolation passes and the rest of the pipeline succeeds:

```text
HUMAN_ACTION_REQUIRED: NO
```
