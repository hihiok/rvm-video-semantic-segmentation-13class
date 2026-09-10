#!/usr/bin/env python3
"""Offline regression tests. No server access, downloads, torch or GPU."""
import contextlib,csv,hashlib,io,json,tempfile,unittest,zipfile,copy
from pathlib import Path
from PIL import Image
import policy as p
import storage as s
import sources as a
import curate as c
import visualize as v
from prepare_all import apply_reviews

HERE=Path(__file__).parent

def make_image(path,num):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    # Reproducible nonuniform content keeps identity tests meaningful.
    data=bytearray()
    for y in range(72):
        for x in range(96):data.extend(((x*7+num*13)%256,(y*11+num*5)%256,(x+y+num*23)%256))
    Image.frombytes('RGB',(96,72),bytes(data)).save(path)


def record(path,src='coco',split='train',i=0):
    y=p.unknown();y['objective_image']=0
    return p.new_record(path,src,split,'office',y,{'objective_image':'weak:photo'},sample_id=src+':'+str(i))


class Policies(unittest.TestCase):
    def test_or_all9(self):
        for x in (-1,0,1):
            for y in (-1,0,1):self.assertEqual(p.rain_or_snow(x,y),1 if 1 in (x,y) else 0 if x==y==0 else -1)
        with self.assertRaises(ValueError):p.rain_or_snow(3,0)
    def test_nus_snow_negative_unknown(self):
        y,_=p.map_nus({'nighttime':0,'sports':1,'snow':0});self.assertEqual(y['rain_snow'],-1);self.assertEqual(y['night'],0)
        y,_=p.map_nus({'nighttime':1,'sports':0,'snow':1});self.assertEqual(y['rain_snow'],1)
    def test_mir_potential_vs_relevant(self):
        sets={'night':{1,2},'night_r1':{1},'indoor':{2}}
        self.assertEqual([p.map_mir(i,sets)[0]['night'] for i in [1,2,3]],[1,-1,0])
        self.assertEqual(p.map_mir(3,sets)[0]['outdoor'],-1)
    def test_places_natural_not_city(self):
        self.assertEqual(p.map_places('downtown',2)[0]['landscape'],0)
        self.assertEqual(p.map_places('mountain',2)[0]['landscape'],1)
        self.assertEqual(p.map_places('field/wild',2)[0]['landscape'],-1)
        self.assertEqual(p.map_places('river',2)[0]['sports'],-1)
    def test_office_and_object_negative(self):
        y,_=p.map_places('home_office',1);self.assertEqual((y['office'],y['objective_image']),(1,0))
        y,_=p.map_places('computer_room',1);self.assertEqual(y['office'],-1);self.assertEqual(y['objective_image'],0)
    def test_no_places_weather_or_night_assumption(self):
        for cat in ('snowfield','glacier','beach','office','airplane_cabin'):
            y,_=p.map_places(cat,1);self.assertEqual(y['rain_snow'],-1);self.assertEqual(y['night'],-1)
    def test_no_ten_non_night_negative(self):
        y,_,_=p.map_ten('Sports');self.assertEqual(y['night'],-1);self.assertEqual(y['sports'],1)
        y,_,why=p.map_ten('Landscape');self.assertEqual(y['landscape'],-1);self.assertTrue(why)
        y,_,_=p.map_ten('Computer_synthesized');self.assertEqual(y['objective_image'],1)
    def test_official_places_checksum(self):
        data=(HERE/'places365_io.txt').read_bytes();gitsha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        self.assertEqual(gitsha,'6e111935c0524f3d0b26f62c71445bfacc5a0fe2')
        io,flat=a.places_map(HERE/'places365_io.txt');self.assertEqual(len(io),365);self.assertEqual(flat[p.norm('field-cultivated')],'field/cultivated')

class Formats(unittest.TestCase):
    def test_safe_zip_rejects_traversal(self):
        for name in ['../bad.txt','/bad.txt','C:/bad.txt','a\\bad.txt']:
            b=io.BytesIO()
            with zipfile.ZipFile(b,'w') as z:z.writestr(name,'x')
            b.seek(0)
            with zipfile.ZipFile(b) as z:
                with self.assertRaises(s.Blocked):s.safe_members(z)
    def test_extract_preserves_archive_and_existing_files(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);z=root/'x.zip'
            with zipfile.ZipFile(z,'w') as f:f.writestr('ann/night.txt','1\n');f.writestr('features.mat','skip')
            before=s.sha(z);dest=root/'extract';s.extract(z,dest,root/'inventory')
            self.assertEqual(s.sha(z),before);self.assertFalse((dest/'features.mat').exists())
            mtime=(dest/'ann/night.txt').stat().st_mtime_ns;s.extract(z,dest,root/'inventory')
            self.assertEqual((dest/'ann/night.txt').stat().st_mtime_ns,mtime)
            (dest/'ann/night.txt').unlink()
            with self.assertRaises(s.Blocked):s.extract(z,dest,root/'inventory')
    def test_nus_order_not_sorted(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);make_image(root/'Flickr/scene/z.jpg',2);make_image(root/'Flickr/scene/a.jpg',3)
            concepts=['nighttime','sports','snow']+['test%d'%i for i in range(78)]
            (root/'Concepts81.txt').write_text('\n'.join(concepts));(root/'Imagelist.txt').write_text('scene\\z.jpg\nscene\\a.jpg\n')
            for k in ('nighttime','sports','snow'):(root/('Labels_'+k+'.txt')).write_text('1\n0\n')
            rows=a.nus_rows(root,42,{},[])
            self.assertEqual(Path(rows[0]['image']).name,'z.jpg');self.assertEqual(rows[0]['labels']['night'],1)
            self.assertEqual(Path(rows[1]['image']).name,'a.jpg');self.assertEqual(rows[1]['labels']['rain_snow'],-1)
            (root/'Labels_snow.txt').write_text('1\n')
            with self.assertRaises(s.Blocked):a.nus_rows(root,42,{},[])
    def test_nus_missing_images_does_not_shift_labels(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);make_image(root/'Flickr/a.jpg',2)
            (root/'Concepts81.txt').write_text('\n'.join(['nighttime','sports','snow']+['k%d'%i for i in range(78)]))
            (root/'Imagelist.txt').write_text('missing.jpg\na.jpg\n')
            for k in ('nighttime','sports','snow'):(root/('Labels_'+k+'.txt')).write_text('1\n0\n')
            ex=[];rows=a.nus_rows(root,4,{},ex)
            self.assertEqual(rows[0]['native_row'],1);self.assertEqual(rows[0]['labels']['night'],0);self.assertEqual(len(ex),1)
    def test_nus_rejects_blank_rows(self):
        with tempfile.TemporaryDirectory() as t:
            x=Path(t)/'labels.txt';x.write_text('0\n\n1\n')
            with self.assertRaises(s.Blocked):s.binary_lines(x)
    def test_nus_ambiguous_basename(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);make_image(root/'a/x.jpg',1);make_image(root/'b/x.jpg',2)
            idx=s.ImageIndex(root)
            self.assertTrue(str(idx.resolve('a/x.jpg')).endswith('a/x.jpg'))
            with self.assertRaises(s.Blocked):idx.resolve('x.jpg')
    def test_mir_fixture(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);im=root/'images';ann=root/'annotations';ann.mkdir()
            for i in range(1,9):make_image(im/('im%d.jpg'%i),i)
            (ann/'README.txt').write_text('fixture');(ann/'night.txt').write_text('1\n2\n');(ann/'indoor.txt').write_text('3\n4\n')
            audit={};rows=a.mir_rows(im,ann,42,audit,[],expected=8)
            self.assertEqual(len(rows),8);self.assertEqual(rows[-1]['labels']['night'],0);self.assertEqual(rows[-1]['labels']['sports'],-1)
            (ann/'night_r1.txt').write_text('8\n')
            audit={};rows=a.mir_rows(im,ann,42,audit,[],expected=8)
            self.assertEqual(rows[-1]['labels']['night'],1)
            self.assertEqual(audit['MIRFLICKR']['relevant_outside_potential']['night_r1']['ids'],[8])

    def test_mir_unused_people_difference_is_audited(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);ann=root/'ann';ann.mkdir()
            for i in range(1,9):make_image(root/'images'/('im%d.jpg'%i),i)
            for name,text in {'README.txt':'fixture','night.txt':'1\n','indoor.txt':'2\n',
                              'people.txt':'1\n2\n','people_r1.txt':'3\n4\n5\n6\n'}.items():
                (ann/name).write_text(text)
            audit={};rows=a.mir_rows(root/'images',ann,42,audit,[],expected=8)
            self.assertEqual(len(rows),8)
            extra=audit['MIRFLICKR']['relevant_outside_potential']['people_r1']
            self.assertEqual(extra['count'],4);self.assertFalse(extra['used_for_nas8'])
            self.assertEqual(rows[2]['labels']['night'],0)
            (ann/'people_r1.txt').write_text('9\n')
            with self.assertRaises(s.Blocked):a.mir_rows(root/'images',ann,42,{},[],expected=8)
            (ann/'people_r1.txt').write_text('3\n3\n')
            with self.assertRaises(s.Blocked):a.mir_rows(root/'images',ann,42,{},[],expected=8)

    def test_nus_separate_metadata_exact_order(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);meta=root/'meta';meta.mkdir();photos=root/'photos'
            make_image(photos/'scene/z.jpg',1);make_image(photos/'scene/a.jpg',2)
            (meta/'Concepts81.txt').write_text('\n'.join(['nighttime','sports','snow']+['k%d'%i for i in range(78)]))
            (meta/'TrainImagelist.txt').write_text('scene/z.jpg\n')
            (meta/'TestImagelist.txt').write_text('scene/a.jpg\n')
            for concept in ('nighttime','sports','snow'):
                (meta/('Labels_'+concept+'_Train.txt')).write_text('1\n')
                (meta/('Labels_'+concept+'_Test.txt')).write_text('0\n')
            rows=a.nus_rows(photos,42,{},[],metadata_root=meta)
            self.assertEqual(rows[0]['labels']['night'],1)
            self.assertEqual(rows[1]['labels']['night'],0)
            self.assertEqual(rows[1]['preferred_split'],'test')
            self.assertEqual(rows[1]['labels']['rain_snow'],-1)

    def test_nus_retrieval_indices_never_guessed(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);make_image(root/'x.jpg',1)
            (root/'database_img.txt').write_text('x.jpg\n')
            (root/'database_label.txt').write_text('0 2 20\n')
            with self.assertRaisesRegex(s.Blocked,'NUS_RETRIEVAL_MAPPING_UNVERIFIED'):
                a.nus_rows(root,42,{},[])

    def test_diagnostic_bundle_preserves_rows_and_has_no_images(self):
        from source_preflight import export_nus_diagnostics
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'source';root.mkdir();out=Path(t)/'diagnostics'
            (root/'README.md').write_text('Top classes 21')
            (root/'database_img.txt').write_text('z.jpg\na.jpg\n')
            (root/'database_label.txt').write_text('0 2\n\n20\n')
            (root/'proxy.md').write_text('PRIVATE NOT INCLUDED')
            make_image(root/'z.jpg',1)
            before=s.sha(root/'database_label.txt')
            report=export_nus_diagnostics(root,out)
            self.assertEqual(s.sha(root/'database_label.txt'),before)
            item=next(x for x in report['metadata'] if x['path']=='database_label.txt')
            self.assertEqual((item['rows'],item['blank_rows']),(3,1))
            with zipfile.ZipFile(out/'nus_diagnostics.zip') as z:
                self.assertFalse(any(x.endswith('.jpg') or 'proxy' in x for x in z.namelist()))
                self.assertEqual(z.read('metadata/database_label.txt'),b'0 2\n\n20\n')
            with self.assertRaises(s.Blocked):export_nus_diagnostics(root,out)
            with self.assertRaises(s.Blocked):export_nus_diagnostics(root,root/'bad')

class Curation(unittest.TestCase):
    def test_legacy_seg_rules_removed_and_user_fix(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);old=root/'old';old.mkdir();dataset=root/'dataset';image=dataset/'ADE_train_00001854.jpg';make_image(image,1)
            row={'image':str(image),'source':'seg13','labels':dict.fromkeys(p.LABELS,1),'detail':'natural_ratio=.8'}
            for split in ('train','val','test'):s.jsonl(old/(split+'.jsonl'),[row])
            rows=a.legacy_rows(old,HERE/'places365_io.txt',[dataset],{},[])
            self.assertEqual(rows[0]['labels']['sports'],1);self.assertEqual(rows[0]['labels']['landscape'],0)
            self.assertEqual(rows[0]['labels']['rain_snow'],-1);self.assertEqual(rows[0]['labels']['night'],-1)
    def test_qualified_coco_id_only(self):
        for source in ('mirflickr','nuswide','10_scenes'):self.assertIsNone(c.coco_id({'source':source,'image':'000000465180.jpg'}))
        self.assertEqual(c.coco_id({'source':'seg13','image':'000000465180.jpg'}),'coco:000000465180')
    def test_duplicates_split_eval_priority(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);make_image(root/'a.jpg',1);make_image(root/'b.jpg',1)
            rows=c.quality_scan([record(root/'a.jpg',split='train'),record(root/'b.jpg',split='test',i=1)],[])
            rows=c.group_and_split(rows,{})
            self.assertEqual({r['split'] for r in rows},{'test'});self.assertEqual(sum(r['use_for_training_manifest'] for r in rows),1)
    def test_coco_derivative_no_blind_label_merge(self):
        rows=[]
        for i,source in enumerate(('coco','seg13')):
            r=record('/'+source+'/000000123456.jpg',source,i=i);r['rgb_sha256']=str(i);r['labels']['night']=i;rows.append(r)
        rows=c.group_and_split(rows,{})
        self.assertNotEqual(rows[0]['labels']['night'],rows[1]['labels']['night']);self.assertEqual(rows[0]['group_id'],rows[1]['group_id'])
    def test_exact_pixel_conflict_unknown(self):
        rows=[]
        for i in (0,1):
            r=record('/x%d.jpg'%i,i=i);r['rgb_sha256']='same';r['labels']['night']=i;rows.append(r)
        audit={};c.group_and_split(rows,audit);self.assertEqual(rows[0]['labels']['night'],-1);self.assertTrue(audit['grouping']['label_conflicts'])
    def test_manual_override_requires_approval(self):
        with tempfile.TemporaryDirectory() as t:
            r=record(Path(t)/'a.jpg');path=Path(t)/'r.csv'
            path.write_text('image,reviewed,reviewer,night\n%s,,human,0\n'%r['image'])
            apply_reviews([r],path,{});self.assertEqual(r['labels']['night'],-1)
            path.write_text('image,reviewed,reviewer,night\n%s,1,human,0\n'%r['image'])
            apply_reviews([r],path,{});self.assertEqual(r['labels']['night'],0)
    def test_weather_negatives_cannot_be_only_synthetic(self):
        rows=[]
        for split in ('train','val','test'):
            for i in range(32):
                r=record('/x',src='10_scenes');r.update(split=split,group_id=split+str(i),use_for_training_manifest=True)
                r['labels']['rain_snow']=0;r['evidence']['rain_snow']='weak:user_defined_test_pattern_folder';rows.append(r)
        _,blocks,_=c.quality_gates(rows);self.assertTrue(any('real-photo negatives' in b for b in blocks))
    def test_resolution_uses_small_images(self):
        rows=[]
        for i in range(60):
            r=record('/x',src='mirflickr');r.update(width=200,height=140,split='train',use_for_training_manifest=True);rows.append(r)
        self.assertFalse(c.resolution(rows)['candidates'][-1]['native_source_gate'])

class Visual(unittest.TestCase):
    def test_100_unique_sources_and_layout(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);rows=[];n=0
            for source,q in v.QUOTAS.items():
                for i in range(q+2):
                    path=root/source/('%03d.jpg'%i);make_image(path,n+1)
                    r=record(path,source,i=n);r.update(width=96,height=72,split='train',group_id=str(n),use_for_training_manifest=True)
                    r['detail']='long_path_detail_'*(30 if n==0 else 1);r['evidence']['objective_image']='long_evidence_'*(35 if n==0 else 1)
                    rows.append(r);n+=1
            out=root/'review';out.mkdir();chosen=v.export_review(rows,out,42)
            self.assertEqual(len(chosen),100);self.assertEqual(len(list((out/'gt100').glob('*.jpg'))),100)
            report=json.loads((out/'gt100_audit.json').read_text());self.assertEqual(report['source_counts'],v.QUOTAS)
            self.assertTrue(all(x['last_row_bottom']<x['height'] for x in report['layouts']))
            # Retain one card locally for visual inspection during code preparation.
            import shutil,os
            if os.environ.get('VIZ_TEST_COPY'):
                shutil.copyfile(sorted((out/'gt100').glob('*.jpg'))[0],Path(os.environ['VIZ_TEST_COPY']))

if __name__=='__main__':unittest.main(verbosity=2)
