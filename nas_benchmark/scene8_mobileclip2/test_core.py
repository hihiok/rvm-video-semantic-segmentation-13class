import copy
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

import numpy as np
from PIL import Image
from core import LABELS, PROMPTS, metrics, audit, protocol


class MetricsTests(unittest.TestCase):
    def test_taxonomy_and_prompts(self):
        self.assertEqual(LABELS,['night','indoor','rain_snow','office','outdoor','landscape','sports','objective_image'])
        self.assertEqual(set(PROMPTS),set(LABELS))
        self.assertIn('natural landscape',PROMPTS['landscape'][0][0])
        self.assertIn('neither rain nor snow',PROMPTS['rain_snow'][1][0])

    def test_unknown_not_negative(self):
        y=np.full((3,8),-1);y[:2,0]=[1,0]
        s=np.full((3,8),0.99);s[:2,0]=[0.9,0.1]
        out=metrics(y,s)
        self.assertEqual(out['per_class'][0]['f1'],1)
        self.assertEqual(out['per_class'][0]['unknown'],1)
        self.assertIsNone(out['per_class'][1]['accuracy'])
        self.assertIsNone(out['summary']['macro_f1_all8'])
        self.assertEqual(out['summary']['covered_count'],1)

    def test_empty_strict(self):
        out=metrics(np.empty((0,8)),np.empty((0,8)))
        self.assertEqual(out['summary']['covered_count'],0)
        self.assertIsNone(out['summary']['micro_f1_known_pairs'])

    def test_weather_positive_only(self):
        y=np.full((2,8),-1);y[:,2]=1
        s=np.full((2,8),0.9);s[0,2]=0.1
        r=metrics(y,s)['per_class'][2]
        self.assertEqual(r['recall'],0.5)
        self.assertIsNone(r['f1']);self.assertIsNone(r['ap'])
        self.assertEqual(r['status'],'SINGLE_CLASS_GT')

    def test_negative_only(self):
        r=metrics(np.zeros((2,8)),np.zeros((2,8)))['per_class'][0]
        self.assertEqual(r['specificity'],1)
        self.assertIsNone(r['accuracy']);self.assertIsNone(r['recall'])

    def test_perfect_all8(self):
        y=np.array([[0]*8,[1]*8]);out=metrics(y,y)
        self.assertEqual(out['summary']['macro_f1_all8'],1)
        self.assertEqual(out['summary']['macro_ap_all8'],1)

    def test_invalid_scores(self):
        for v in (float('nan'),float('inf'),-0.1,1.1):
            with self.assertRaises(ValueError):metrics(np.zeros((1,8)),np.full((1,8),v))


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.rows={}
        for i,s in enumerate(('train','val','test')):
            image=self.root/(s+'.png');Image.new('RGB',(20,20)).save(image)
            r=dict(image=str(image),source='nuswide',split=s,group_id=s,
                   labels=dict.fromkeys(LABELS,-1),evidence={'night':'manual:NUS_nighttime'})
            r['labels']['night']=i%2
            self.rows[s]=r
            self.write(s,[r])
            if s!='train':self.write(s+'_strict',[r])
        (self.root/'summary.json').write_text(json.dumps(dict(schema='nas8_source_curation_v3',labels=LABELS)))
        (self.root/'PREPARED.json').write_text('{}')

    def tearDown(self):self.temp.cleanup()

    def write(self,s,rows):
        (self.root/(s+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))

    def test_valid(self):
        rows,strict,meta=audit(self.root,'test')
        self.assertEqual(len(rows),1);self.assertEqual(len(strict),1)
        self.assertEqual(meta['split'],'test')

    def test_old_schema_rejected(self):
        (self.root/'summary.json').write_text('{}')
        with self.assertRaises(ValueError):audit(self.root,'test')

    def test_rev3_preflight_cli(self):
        meta=dict(schema='nas8_source_curation_v3_rev3',labels=LABELS,
                  status='PREPARED_REVIEW_REQUIRED',HUMAN_ACTION_REQUIRED=True,
                  training_quality_blockers=['rain_snow_real_negative_shortage'])
        (self.root/'summary.json').write_text(json.dumps(meta))
        r=copy.deepcopy(self.rows['test'])
        r['labels']=dict.fromkeys(LABELS,-1);r['labels']['rain_snow']=1
        r['evidence']={'rain_snow':'manual:NUS21_verified_snow_positive'}
        self.write('test',[r]);self.write('test_strict',[r])
        rows,strict,info=audit(self.root,'test')
        self.assertEqual(info['schema'],meta['schema'])
        self.assertTrue(info['dataset_human_action_required'])
        self.assertEqual(info['training_quality_blockers'],meta['training_quality_blockers'])
        result=protocol(rows,strict,np.ones((1,8)))['strict']
        self.assertEqual(result['per_class'][2]['status'],'SINGLE_CLASS_GT')
        self.assertIsNone(result['summary']['macro_f1_all8'])
        self.test_preflight_cli_without_torch()

    def test_rev3_keeps_safety_checks(self):
        meta=dict(schema='nas8_source_curation_v3_rev3',labels=LABELS)
        (self.root/'summary.json').write_text(json.dumps(meta))
        (self.root/'BLOCKED.json').write_text('{}')
        with self.assertRaises(ValueError):audit(self.root,'test')
        (self.root/'BLOCKED.json').unlink()
        meta['labels']=list(reversed(LABELS))
        (self.root/'summary.json').write_text(json.dumps(meta))
        with self.assertRaises(ValueError):audit(self.root,'test')

    def test_unverified_schema_rejected(self):
        for schema in ('nas8_source_curation_v3_rev4','nas8_source_curation_v3_custom'):
            with self.subTest(schema=schema):
                (self.root/'summary.json').write_text(json.dumps(dict(schema=schema,labels=LABELS)))
                with self.assertRaises(ValueError):audit(self.root,'test')

    def test_overlap_rejected(self):
        r=copy.deepcopy(self.rows['test']);r['group_id']='train';self.write('test',[r])
        with self.assertRaises(ValueError):audit(self.root,'test')

    def test_strict_conflict(self):
        r=copy.deepcopy(self.rows['test']);r['labels']['night']=1;self.write('test_strict',[r])
        with self.assertRaises(ValueError):audit(self.root,'test')

    def test_strict_weak_rejected(self):
        r=copy.deepcopy(self.rows['test']);r['evidence']['night']='weak:brightness';self.write('test_strict',[r])
        with self.assertRaises(ValueError):audit(self.root,'test')

    def test_blocked_rejected(self):
        (self.root/'BLOCKED.json').write_text('{}')
        with self.assertRaises(ValueError):audit(self.root,'test')

    def test_empty_strict_allowed(self):
        self.write('test_strict',[])
        rows,strict,_=audit(self.root,'test')
        p=protocol(rows,strict,np.zeros((1,8)))
        self.assertEqual(p['strict']['summary']['covered_count'],0)
        self.assertEqual(p['mapped']['per_class'][0]['negative'],1)

    def test_eight_vector_not_accepted(self):
        r=copy.deepcopy(self.rows['test']);r['labels']=[0]*8;self.write('test',[r])
        with self.assertRaises(ValueError):audit(self.root,'test')

    def test_preflight_cli_without_torch(self):
        with tempfile.TemporaryDirectory() as other:
            out=Path(other)/'audit'
            p=subprocess.run([sys.executable,str(Path(__file__).with_name('run.py')),
                              '--label-root',str(self.root),'--output-dir',str(out),
                              '--preflight-only'],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertEqual(json.loads((out/'STATUS.json').read_text())['status'],'DATA_PREFLIGHT_PASS')
            self.assertFalse((out/'metrics.json').exists())


if __name__=='__main__':unittest.main()
