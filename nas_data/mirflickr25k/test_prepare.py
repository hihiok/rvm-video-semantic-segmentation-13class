#!/usr/bin/env python3
"""Offline unit/integration tests. No network, GPUs, or user dataset needed."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from PIL import Image
import prepare as m


class Tests(unittest.TestCase):
    def test_rain_snow_or_all_states(self):
        for a in (-1,0,1):
            for b in (-1,0,1):
                expected=1 if 1 in (a,b) else (0 if a==b==0 else -1)
                self.assertEqual(m.rain_snow_or(a,b),expected)
        with self.assertRaises(ValueError): m.rain_snow_or(2,0)

    def test_relevance_mapping(self):
        ann={'night':{1,2},'night_r1':{1},'indoor':{2,3}}
        self.assertEqual(m.project_labels(1,ann)[0]['night'],1)
        self.assertEqual(m.project_labels(2,ann)[0]['night'],-1)
        self.assertEqual(m.project_labels(3,ann)[0]['night'],0)
        self.assertEqual(m.project_labels(2,ann)[0]['indoor'],1)
        y,_=m.project_labels(4,ann)
        self.assertEqual(y['indoor'],0)
        for l in ('outdoor','objective_image','rain_snow','landscape','sports','office'):
            self.assertEqual(y[l],-1)

    def test_numeric_lists(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'night.txt';p.write_text('1\n2\n\n')
            self.assertEqual(m.parse_id_list(p,{1,2,3}),{1,2})
            for text in ('1 1\n','0\n','1\n1\n','100\n'):
                p.write_text(text)
                with self.assertRaises(m.PreparationError):m.parse_id_list(p,{1,2,3})

    def test_annotation_scope_and_subset(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            (root/'README.txt').write_text('Test annotation README')
            (root/'night.txt').write_text('1\n2\n')
            (root/'night_r1.txt').write_text('1\n')
            (root/'indoor.txt').write_text('3\n')
            sets,_=m.read_annotations(root,{1,2,3,4})
            self.assertEqual(sets['night_r1'],{1})
            (root/'night_r1.txt').write_text('4\n')
            with self.assertRaises(m.PreparationError):m.read_annotations(root,{1,2,3,4})

    def test_unsafe_zip(self):
        for name in ('../escape.txt','/absolute','safe/../../escape','C:/escape','a\\b'):
            b=io.BytesIO()
            with zipfile.ZipFile(b,'w') as z:z.writestr(name,b'data')
            b.seek(0)
            with zipfile.ZipFile(b) as z:
                with self.assertRaises(m.PreparationError):m.safe_zip_infos(z)

    def test_archive_checksum(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'x.zip'
            with zipfile.ZipFile(p,'w') as z:z.writestr('good/data.txt',b'x')
            self.assertTrue(m.verify_archive(p,m.digest(p,'md5'))['official_md5_checked'])
            with self.assertRaises(m.PreparationError):m.verify_archive(p,'0'*32)

    def test_extraction_idempotent_no_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'x.zip';dest=Path(t)/'raw'
            with zipfile.ZipFile(p,'w') as z:z.writestr('folder/a.txt',b'data')
            info=m.verify_archive(p);m.extract_once(p,dest,info)
            before=(dest/'folder/a.txt').stat().st_mtime_ns
            m.extract_once(p,dest,info)
            self.assertEqual((dest/'folder/a.txt').stat().st_mtime_ns,before)
            info['sha256']='bad'
            with self.assertRaises(m.PreparationError):m.extract_once(p,dest,info)

    def test_group_split_and_conflict(self):
        rows=[]
        for i in range(60):
            y=dict.fromkeys(m.LABELS,-1);y['night']=i%2;y['indoor']=(i//2)%2
            rows.append({'image_id':i,'group_id':str(i),'labels':y,'label_evidence':{}})
        dupe=copy.deepcopy(rows[0]);dupe['image_id']=100;dupe['labels']['night']=1;rows.append(dupe)
        split,audit=m.assign_splits(copy.deepcopy(rows),42)
        split2,_=m.assign_splits(copy.deepcopy(rows),42)
        self.assertEqual(split,split2)
        self.assertEqual(audit['same_pixel_duplicate_records'],1)
        self.assertTrue(audit['conflicts'])
        members=[r for rr in split.values() for r in rr if r['group_id']=='0']
        self.assertEqual(len({r['split'] for r in members}),1)
        self.assertTrue(all(r['labels']['night']==-1 for r in members))
        self.assertTrue(all(v==0 for v in audit['cross_split_identical_rgb_overlap'].values()))

    def test_resolution_does_not_force_640(self):
        rows=[]
        for i in range(120):
            rows.append({'width':320,'height':240,'labels':{'night':i%2,'indoor':i%2}})
        report=m.resolution_report(rows)
        self.assertEqual(report['native_resolution_upper_bound_wh'],[384,216])
        self.assertFalse(report['candidates_16x9'][-1]['native_size_gate_pass'])

    def test_missing_images_fail(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);Image.new('RGB',(20,20)).save(p/'im1.jpg')
            with self.assertRaises(m.PreparationError):m.scan_images(p,{'night':set(),'indoor':set()},2)

    def test_full_small_fixture_and_immutable_sources(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);images=root/'raw/images/mirflickr';ann=root/'raw/annotations_v080/ann';out=root/'derived/test'
            images.mkdir(parents=True);ann.mkdir(parents=True);out.mkdir(parents=True)
            for i in range(1,81):
                Image.new('RGB',(80+i,70+i),(i,2*i,3*i)).save(images/('im%d.jpg'%i),quality=100)
            (ann/'README.txt').write_text('Fixture, not the official README')
            (ann/'night.txt').write_text('\n'.join(str(i) for i in range(1,81) if i%2==0))
            (ann/'night_r1.txt').write_text('\n'.join(str(i) for i in range(1,81) if i%4==0))
            (ann/'indoor.txt').write_text('\n'.join(str(i) for i in range(1,81) if i%3==0))
            before={str(p):m.digest(p) for p in (root/'raw').rglob('*') if p.is_file()}
            with contextlib.redirect_stdout(io.StringIO()):m.prepare(root,out,42,{},expected_count=80)
            after={str(p):m.digest(p) for p in (root/'raw').rglob('*') if p.is_file()}
            self.assertEqual(before,after)
            summary=json.loads((out/'dataset_summary.json').read_text())
            self.assertEqual(summary['num_images'],80)
            self.assertFalse(summary['READY_FOR_8LABEL_TRAINING'])
            for split in ('train','val','test'):
                rows=[json.loads(x) for x in (out/(split+'.jsonl')).read_text().splitlines()]
                self.assertTrue(rows)
                self.assertTrue(all(r['labels']['rain_snow']==-1 for r in rows))
            self.assertEqual(len(list((out/'review').glob('*.jpg'))),48)
            self.assertTrue((out/'review/index.html').is_file())
            with Image.open(next((out/'review').glob('*.jpg'))) as preview:
                self.assertEqual(preview.size,(720,704))


if __name__=='__main__':unittest.main(verbosity=2)
