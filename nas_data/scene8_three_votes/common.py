"""Read-only inputs, content-bound resumability, and conservative per-label voting."""
import hashlib
import json
import math
import os
import sqlite3
import fcntl
from pathlib import Path
from prompts import LABELS


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def dump(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def jsonl(path, rows):
    with Path(path).open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def valid_labels(y, keys=LABELS):
    if not isinstance(y, dict) or set(y) != set(keys) or any(type(v) is not int or v not in (-1, 0, 1) for v in y.values()):
        raise ValueError('Invalid exact label dictionary')


def load_run(root):
    root = Path(root).resolve()
    plan = read(root / 'plan.json')
    if sha(root / 'items.jsonl') != plan['items_digest']:
        raise ValueError('Changed full annotation selection')
    for name, h in plan['input_hashes'].items():
        if sha(Path(plan['input_root']) / name) != h:
            raise ValueError('Source manifest changed: ' + name)
    with (root / 'items.jsonl').open(encoding='utf-8') as f:
        items = [json.loads(line) for line in f]
    if len(items) != plan['count']:
        raise ValueError('Incomplete selection')
    return root, plan, items


class RecordStore:
    """One process per shard; atomic SQLite commits survive interruption."""
    def __init__(self, path, run_digest, readonly=False):
        self.run_digest = run_digest
        self.lock = None
        if readonly:
            self.db = sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True)
        else:
            self.lock = open(str(path) + '.lock', 'a')
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.db = sqlite3.connect(str(path))
            self.db.execute('CREATE TABLE IF NOT EXISTS results (id TEXT PRIMARY KEY, result TEXT NOT NULL)')
            self.db.commit()

    def get(self, item):
        row = self.db.execute('SELECT result FROM results WHERE id=?', (item['id'],)).fetchone()
        if row is None:
            return None
        result = json.loads(row[0])
        if result.get('image_sha256') != item['image_sha256'] or result.get('run_digest') != self.run_digest:
            raise ValueError('Mismatched cached image/model/prompt')
        valid_labels(result['labels'])
        return result

    def put(self, item, result):
        valid_labels(result['labels'])
        result.update(id=item['id'], image_sha256=item['image_sha256'], run_digest=self.run_digest)
        self.db.execute('INSERT INTO results VALUES (?, ?)', (item['id'], json.dumps(result, ensure_ascii=False, allow_nan=False)))
        self.db.commit()

    def close(self):
        self.db.close()
        if self.lock:
            self.lock.close()


def stage_lock(root, name, meta):
    code = {p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    meta = dict(meta, code_hashes=code)
    d = root / name
    d.mkdir(exist_ok=True)
    p = d / 'metadata.json'
    if p.exists() and read(p) != meta:
        raise ValueError('Resume refused: model, code, prompts or selection changed. Use a new output directory.')
    if not p.exists():
        dump(p, meta)
    return d, digest(meta)


def weather_or(rain, snow):
    if rain == 1 or snow == 1:
        return 1
    return 0 if rain == snow == 0 else -1


def parse_qwen(text):
    s = text.strip()
    if s.startswith('```json') and s.endswith('```'):
        s = s[7:-3].strip()
    obj = json.loads(s)
    if set(obj) != {'labels', 'evidence'}:
        raise ValueError('Qwen must return exactly labels/evidence')
    keys = LABELS + ['rain', 'snow']
    valid_labels(obj['labels'], keys)
    if not isinstance(obj['evidence'], dict) or set(obj['evidence']) != set(keys) or any(not isinstance(v, str) or not v.strip() or len(v) > 600 for v in obj['evidence'].values()):
        raise ValueError('Missing or malformed Qwen visual evidence')
    supplied = obj['labels']['rain_snow']
    obj['labels']['rain_snow'] = weather_or(obj['labels']['rain'], obj['labels']['snow'])
    obj['weather_or_corrected'] = supplied != obj['labels']['rain_snow']
    obj['spatial_conflict'] = obj['labels']['indoor'] == obj['labels']['outdoor'] == 1
    if obj['spatial_conflict']:
        obj['labels']['indoor'] = obj['labels']['outdoor'] = -1
    return obj


def mobile_vote(margins, abstain_margin=0.02):
    if len(margins) != 3 or not all(math.isfinite(v) for v in margins):
        raise ValueError('Expected three finite cosine margins')
    # Prompt robustness within ONE model, not three independent votes.
    positive = sum(v >= abstain_margin for v in margins)
    negative = sum(v <= -abstain_margin for v in margins)
    mean = sum(margins) / 3
    if positive >= 2 and negative == 0 and mean >= abstain_margin:
        return 1
    if negative >= 2 and positive == 0 and mean <= -abstain_margin:
        return 0
    return -1


def fuse(old, mobile, qwen, evidence=None):
    for y in (old, mobile, qwen):
        valid_labels(y)
    evidence = evidence or {}
    out, details = {}, {}
    for label in LABELS:
        votes = [old[label], mobile[label], qwen[label]]
        pos, neg = votes.count(1), votes.count(0)
        majority = 1 if pos >= 2 else (0 if neg >= 2 else -1)
        final = majority
        flags = []
        if pos and neg:
            flags.append('disagreement')
        # Protect scarce positives and explicit reviewer decisions from silent automatic reversal.
        if old[label] == 1 and majority == 0:
            final = -1
            flags.append('old_positive_reversal_requires_review')
        trusted = any(t in str(evidence.get(label, '')) for t in ('human_review:', 'user_review:'))
        if trusted and majority not in (-1, old[label]):
            final = -1
            flags.append('human_review_conflict')
        out[label] = final
        details[label] = {'votes': votes, 'positive_votes': pos, 'negative_votes': neg,
                          'abstentions': votes.count(-1), 'majority_proposal': majority,
                          'agreement_fraction': max(pos, neg) / (pos + neg) if pos + neg else None,
                          'flags': flags}
    if out['indoor'] == out['outdoor'] == 1:
        out['indoor'] = out['outdoor'] = -1
        for label in ('indoor', 'outdoor'):
            details[label]['flags'].append('fused_indoor_outdoor_conflict_requires_review')
    return out, details
