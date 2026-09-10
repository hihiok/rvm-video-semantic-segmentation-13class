"""Regression checks for evidence-pinned retrieval parsing and partial supervision."""
import copy,tempfile,unittest,json,subprocess,sys,zipfile
from pathlib import Path
from unittest.mock import patch
import nus21
from storage import Blocked,sha,ImageIndex
from sources import split_hash
from test_pipeline import make_image

def fixture(root):
    profile=copy.deepcopy(nus21.load_profile())
    for split,names,indices in (
        ('database',['images/z.jpg','images/a.jpg'],[[17],[0,1]]),
        ('test',['images/q.jpg'],[[17]])):
        (root/(split+'_img.txt')).write_text('\n'.join(names)+'\n')
        (root/(split+'_label.txt')).write_text('\n'.join(' '.join(map(str,x)) for x in indices)+'\n')
        (root/(split+'_label_onehot.txt')).write_text('\n'.join(' '.join(str(int(i in x)) for i in range(21)) for x in indices)+'\n')
        profile['expected_rows'][split]=len(names)
    repin(root,profile)
    return profile

def repin(root,profile):
    for name in profile['files']:
        f=root/name;profile['files'][name]={'bytes':f.stat().st_size,'sha256':sha(f)}

class Top21(unittest.TestCase):
    def test_order_and_integer_multihot_alignment(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p=fixture(root)
            rows=list(nus21.validated_pairs(lambda x:root/x,p))
            self.assertEqual([r[2] for r in rows],['images/z.jpg','images/a.jpg','images/q.jpg'])
            self.assertEqual([r[3][17] for r in rows],[1,0,1])

    def test_changed_query_file_rejected_before_first_yield(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p=fixture(root)
            (root/'test_label.txt').write_text('1\n')
            with self.assertRaisesRegex(Blocked,'FINGERPRINT'):
                next(nus21.validated_pairs(lambda x:root/x,p))

    def test_mismatched_indices_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p=fixture(root)
            (root/'database_label.txt').write_text('16\n0 1\n');repin(root,p)
            with self.assertRaisesRegex(Blocked,'disagreement'):
                list(nus21.validated_pairs(lambda x:root/x,p))

    def test_blank_and_wrong_width_rejected(self):
        for content in ('\n','0 1\n',' '.join(['2']*21)+'\n'):
            with tempfile.TemporaryDirectory() as t:
                root=Path(t);p=fixture(root)
                (root/'database_label_onehot.txt').write_text(content);repin(root,p)
                with self.assertRaises(Blocked):list(nus21.validated_pairs(lambda x:root/x,p))

    def test_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p=fixture(root)
            (root/'test_img.txt').write_text('images/z.jpg\n');repin(root,p)
            with self.assertRaisesRegex(Blocked,'overlap'):
                list(nus21.validated_pairs(lambda x:root/x,p))

    def test_missing_image_does_not_shift_labels(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p=fixture(root);audit={};excluded=[]
            make_image(root/'images/a.jpg',1);make_image(root/'images/q.jpg',2)
            with patch('nus21.load_profile',return_value=p):
                rows=nus21.top21_rows(ImageIndex(root),lambda x:root/x,42,audit,excluded,split_hash)
            self.assertEqual(len(excluded),1);self.assertEqual(rows[0]['native_row'],1)
            self.assertTrue(all(v==-1 for v in rows[0]['labels'].values()))
            self.assertEqual(rows[1]['labels']['rain_snow'],1)
            self.assertEqual(rows[1]['preferred_split'],'test')
            self.assertEqual(rows[1]['labels']['night'],-1)
            self.assertEqual(rows[1]['labels']['sports'],-1)
            self.assertEqual(rows[1]['labels']['landscape'],-1)

    def test_only_snow_positive_supervision(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p=fixture(root);audit={}
            for n in ('z','a','q'):make_image(root/('images/'+n+'.jpg'),ord(n))
            with patch('nus21.load_profile',return_value=p):
                rows=nus21.top21_rows(ImageIndex(root),lambda x:root/x,42,audit,[],split_hash)
            self.assertEqual([r['labels']['rain_snow'] for r in rows],[1,-1,1])
            self.assertEqual(sum(v!=-1 for r in rows for v in r['labels'].values()),2)
            self.assertEqual(audit['NUS_WIDE']['snow_positive_before_quality'],{'database':1,'test':1})

    def test_production_profile_has_evidence_and_snow_at_17(self):
        p=nus21.load_profile()
        self.assertEqual(p['class_names'][17],'snow')
        self.assertNotIn('nighttime',p['class_names']);self.assertNotIn('sports',p['class_names'])
        self.assertEqual(p['verification']['joint_label_patterns'],4580)
        self.assertTrue(p['verification']['joint_label_histogram_equal'])
        self.assertEqual(p['positive_counts']['snow'],5404)
        self.assertEqual(sum(p['expected_rows'].values()),195834)

    def test_cli_allows_reusing_nus_inside_cache(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);cache=root/'cache';(cache/'nus').mkdir(parents=True)
            for name in ('mir.zip','ann.zip'):
                with zipfile.ZipFile(root/name,'w') as z:z.writestr('README.txt','fixture')
            cmd=[sys.executable,str(Path(__file__).with_name('prepare_all.py')),
                 '--inspect-only','--mir-images',str(root/'mir.zip'),'--mir-annotations',str(root/'ann.zip'),
                 '--nus-root',str(cache/'nus'),'--cache-root',str(cache),'--output-root',str(root/'out'),
                 '--old-root',str(root/'old'),'--source-roots']+[str(root/('source'+str(i))) for i in range(4)]
            result=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads((root/'out/summary.json').read_text())['status'],'INSPECTED_ONLY')

    def test_gt100_prefers_usable_nus_rows(self):
        from visualize import select100,QUOTAS
        from policy import unknown
        rows=[]
        for src,count in QUOTAS.items():
            for i in range(count*(3 if src=='nuswide' else 1)):
                rows.append(dict(source=src,image='/fixture/'+src+str(i)+'.jpg',
                                 labels=unknown(),group_id=src+str(i),
                                 use_for_training_manifest=i<count))
        chosen=select100(rows,42)
        self.assertEqual(len(chosen),100)
        self.assertTrue(all(r['use_for_training_manifest'] for r in chosen if r['source']=='nuswide'))

if __name__=='__main__':unittest.main(verbosity=2)
