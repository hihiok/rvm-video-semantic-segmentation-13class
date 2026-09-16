import unittest,tempfile,json,csv
from pathlib import Path
from test_refine_labels import row,make_image
from weather_negatives import run,apply,select
from storage import dump,jsonl,sha,Blocked
class WeatherTests(unittest.TestCase):
 def test_no_scene_implies_weather(self):
  rows=[row(str(i),{g:1}) for i,g in enumerate(('night','indoor','office','sports'))]
  self.assertEqual(len(select(rows,10)),4)
  self.assertTrue(all(r['labels']['rain_snow']==-1 for r in rows))
 def test_reject_unverified_and_positive(self):
  r=row('x',{'night':1,'rain_snow':1})
  with self.assertRaises(Blocked):apply([r],[{'split':'train','image':r['image'],'reviewed':'1'}])
  self.assertEqual(r['labels']['rain_snow'],1)
 def test_two_pass_and_preservation(self):
  with tempfile.TemporaryDirectory() as t:
   base=Path(t);old=base/'old';old.mkdir();rows=[]
   for i,labels in enumerate([{'night':1},{'objective_image':1,'rain_snow':0},{'rain_snow':1}]):
    r=row(str(i),labels);p=base/(str(i)+'.jpg');make_image(p,i);r['image']=str(p);rows.append(r)
   for split in ('train','val','test'):jsonl(old/(split+'.jsonl'),rows if split=='train' else [])
   dump(old/'summary.json',{'status':'PREPARED_REVIEW_REQUIRED','splits':{s:{'records':3 if s=='train' else 0} for s in ('train','val','test')}})
   dump(old/'PREPARED.json',{});dump(old/'source_audit.json',{})
   before=sha(old/'train.jsonl');out=base/'new';run(old,out,limit=10)
   rr=[json.loads(x) for x in (out/'train.jsonl').read_text().splitlines()]
   self.assertEqual([r['labels']['rain_snow'] for r in rr],[-1,-1,1])
   review=out/'weather_review_template.csv'
   with review.open(encoding='utf-8-sig') as f:reader=csv.DictReader(f);fields=reader.fieldnames;vv=list(reader)
   vv[0].update(reviewed='1',rain_absent='1',snow_absent='1',reviewer='human')
   with review.open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(vv)
   run(out,base/'applied',review=review)
   final=[json.loads(x) for x in (base/'applied/train.jsonl').read_text().splitlines()]
   self.assertEqual([r['labels']['rain_snow'] for r in final],[0,-1,1]);self.assertEqual(sha(old/'train.jsonl'),before)
   self.assertEqual(len(final),3)
   vv[0]['image_sha256']='bad'
   with self.assertRaises(Blocked):apply(rr,vv)
if __name__=='__main__':unittest.main()
