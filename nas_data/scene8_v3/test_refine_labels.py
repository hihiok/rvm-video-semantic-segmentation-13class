import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from policy import LABELS, unknown
from storage import Blocked, dump, jsonl, sha
from curate import count_labels
from visualize import QUOTAS
from refine_labels import cap_negatives, apply_confirmed_fixes, refine, load_completed
from test_pipeline import make_image


def row(name,labels,split='train',source='coco',detail=''):
    y=unknown();y.update(labels)
    return dict(image='/fixture/'+name+'.jpg',sample_id=name,source=source,detail=detail,
                split=split,preferred_split=split,group_id=name,labels=y,
                evidence={l:'weak:fixture' for l in labels},use_for_training_manifest=True,width=96,height=72)

class RefineLabels(unittest.TestCase):
    def test_all_classes_all_splits_cap_and_keep_positives(self):
        rows=[]
        for split in ('train','val','test'):
            for label in LABELS:
                rows.append(row(split+label+'positive',{label:1},split))
                rows.extend(row(split+label+str(i),{label:0},split) for i in range(5))
        report=cap_negatives(rows,42)
        for split in report:
            for label in LABELS:
                self.assertEqual(report[split]['after'][label]['positive'],1)
                self.assertLessEqual(report[split]['after'][label]['negative'],1)
        self.assertTrue(all(r['use_for_training_manifest'] for r in rows if 1 in r['labels'].values()))

    def test_remove_objective_only_first_keep_mixed_label_positives(self):
        rows=[row('objpos',{'objective_image':1}),row('pure',{'objective_image':0}),
              row('office1',{'office':1,'objective_image':0}),row('office2',{'office':1,'objective_image':0})]
        report=cap_negatives(rows,4)
        self.assertFalse(rows[1]['use_for_training_manifest'])
        self.assertEqual(rows[1]['labels']['objective_image'],0)
        self.assertTrue(all(r['use_for_training_manifest'] and r['labels']['office']==1 for r in rows[2:]))
        self.assertEqual(sum(r['labels']['objective_image']==-1 for r in rows[2:]),1)
        self.assertEqual(report['train']['negative_labels_masked_on_retained_rows']['objective_image'],1)

    def test_rare_positives_survive_other_class_negative_caps(self):
        rare=('rain_snow','office','landscape','sports','objective_image')
        rows=[row(l,{l:1,'indoor':0,'night':0}) for l in rare]
        before=copy.deepcopy(rows);cap_negatives(rows,42)
        for original,r in zip(before,rows):
            self.assertTrue(r['use_for_training_manifest'])
            for l in rare:
                if original['labels'][l]==1:self.assertEqual(r['labels'][l],1)
            self.assertEqual(r['labels']['indoor'],-1)
            self.assertEqual(r['labels']['night'],-1)

    def test_zero_positive_and_unknown_not_negative(self):
        rows=[row('pure',{'indoor':0}),row('mixed',{'night':1,'indoor':0}),row('unknown',{'night':1})]
        report=cap_negatives(rows,4)
        self.assertEqual(report['train']['after']['indoor']['negative'],0)
        self.assertEqual(rows[1]['labels']['indoor'],-1)
        self.assertTrue(rows[1]['use_for_training_manifest'])
        self.assertEqual(rows[2]['labels']['indoor'],-1)

    def test_deterministic_and_idempotent(self):
        rows=[row('p',{'office':1})]+[row('n'+str(i),{'office':0}) for i in range(20)]
        reversed_rows=copy.deepcopy(rows[::-1])
        cap_negatives(rows,5);cap_negatives(reversed_rows,5)
        self.assertEqual({r['image'] for r in rows if r['use_for_training_manifest']},
                         {r['image'] for r in reversed_rows if r['use_for_training_manifest']})
        before=copy.deepcopy(rows);cap_negatives(rows,5);self.assertEqual(rows,before)

    def test_confirmed_fixes_are_source_scoped(self):
        rows=[row('Night1',{'night':1},source='10_scenes',detail='Night'),
              row('Night2',{'night':1},source='mirflickr',detail='Night'),
              row('im1414',{'indoor':1,'night':0},source='mirflickr'),
              row('im14411',{'indoor':1,'night':1},source='mirflickr'),
              row('im1414',{'indoor':1},source='coco')]
        changes=apply_confirmed_fixes(rows)
        self.assertEqual(len(changes),5)
        self.assertEqual(rows[0]['labels']['outdoor'],1)
        self.assertEqual(rows[1]['labels']['outdoor'],-1)
        self.assertEqual(rows[2]['labels']['indoor'],0)
        self.assertEqual(rows[3]['labels']['outdoor'],1)
        self.assertEqual(rows[4]['labels']['indoor'],1)
        self.assertEqual(rows[3]['labels']['rain_snow'],-1)

    def test_input_rejects_blocked_or_missing_prepared(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);dump(root/'BLOCKED.json',{})
            with self.assertRaises(Blocked):load_completed(root)
            (root/'BLOCKED.json').unlink()
            with self.assertRaises(FileNotFoundError):load_completed(root)

    def test_complete_refinement_original_unchanged_gt100_and_ratios(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);old=root/'old';old.mkdir();rows=[];i=0
            for source,n in QUOTAS.items():
                for j in range(n):
                    split=('train','val','test')[j%3]
                    labels={'objective_image':0};detail=''
                    if source=='10_scenes':
                        detail='Night' if j%2 else 'Computer_synthesized'
                        labels={'night':1} if j%2 else {'objective_image':1}
                    elif source=='places365':labels={'indoor':1,'office':0,'objective_image':0}
                    elif source=='nuswide':labels={'rain_snow':1}
                    elif source=='mirflickr':labels={'night':0,'indoor':0}
                    r=row(source+str(j),labels,split,source,detail)
                    path=root/'images'/source/(str(j)+'.jpg');make_image(path,i);i+=1
                    r['image']=str(path);rows.append(r)
            splits={}
            for split in ('train','val','test'):
                rr=[r for r in rows if r['split']==split];jsonl(old/(split+'.jsonl'),rr)
                splits[split]={'records':len(rr),'labels':count_labels(rr)}
            dump(old/'summary.json',{'status':'PREPARED_REVIEW_REQUIRED','splits':splits})
            dump(old/'PREPARED.json',{});dump(old/'source_audit.json',{'objective_sampling':{'max_negative_to_positive':5}})
            hashes={p.name:sha(p) for p in old.iterdir()}
            with contextlib.redirect_stdout(io.StringIO()):summary=refine(old,root/'new',42)
            self.assertTrue(summary['negative_cap_passed'])
            self.assertFalse(summary['training_started'])
            for split in summary['splits']:
                for label in LABELS:
                    counts=summary['splits'][split]['labels'][label]
                    self.assertLessEqual(counts['negative'],counts['positive'])
            for name,h in hashes.items():self.assertEqual(sha(old/name),h)
            gt=json.loads((root/'new/gt100_audit.json').read_text())
            self.assertEqual(gt['source_counts'],QUOTAS)
            self.assertEqual(len(list((root/'new/gt100').glob('*.jpg'))),100)
            self.assertTrue((root/'new/PREPARED.json').is_file())
            for split in ('train','val','test'):
                for r in map(json.loads,(root/'new'/(split+'.jsonl')).read_text().splitlines()):
                    self.assertEqual(r['split'],r['preferred_split'])
            with self.assertRaises(Blocked):refine(old,root/'new',42)
            with self.assertRaises(Blocked):refine(old,old/'bad',42)

if __name__=='__main__':unittest.main()
