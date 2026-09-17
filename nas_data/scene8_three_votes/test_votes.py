import argparse
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from common import LABELS, fuse, mobile_vote, parse_qwen, sha, read, dump, load_run, stage_lock, RecordStore
from prepare import run as prepare_run
from merge import run as merge_run


def labels(**kw):
    return dict({k:-1 for k in LABELS},**kw)


class VotingTests(unittest.TestCase):
    def test_unknown_abstains_not_negative(self):
        result,d=fuse(labels(),labels(),labels())
        self.assertEqual(result,labels())
        self.assertIsNone(d['night']['agreement_fraction'])
        result,_=fuse(labels(),labels(night=1),labels(night=1))
        self.assertEqual(result['night'],1)

    def test_one_vote_is_insufficient(self):
        result,_=fuse(labels(night=1),labels(),labels())
        self.assertEqual(result['night'],-1)

    def test_two_to_one_and_positive_reversal_guard(self):
        result,d=fuse(labels(office=0),labels(office=1),labels(office=1))
        self.assertEqual(result['office'],1)
        result,d=fuse(labels(office=1),labels(office=0),labels(office=0))
        self.assertEqual(result['office'],-1)
        self.assertEqual(d['office']['majority_proposal'],0)
        self.assertIn('old_positive_reversal_requires_review',d['office']['flags'])

    def test_human_negative_conflict(self):
        result,_=fuse(labels(indoor=0),labels(indoor=1),labels(indoor=1),{'indoor':'user_review:confirmed'})
        self.assertEqual(result['indoor'],-1)

    def test_spatial_conflict_abstains_no_fake_positive(self):
        y=labels(indoor=1,outdoor=1)
        result,_=fuse(y,y,y)
        self.assertEqual(result['indoor'],-1)
        self.assertEqual(result['outdoor'],-1)

    def test_outdoor_landscape_and_sports_independent(self):
        y=labels(outdoor=1,landscape=1,sports=1,rain_snow=1)
        result,_=fuse(y,y,labels())
        self.assertEqual(result,y)

    def test_mobile_prompt_disagreement_and_margin(self):
        self.assertEqual(mobile_vote([.05,.03,.01]),1)
        self.assertEqual(mobile_vote([-.05,-.03,-.01]),0)
        self.assertEqual(mobile_vote([.05,.04,-.03]),-1)
        self.assertEqual(mobile_vote([.001,.001,.001]),-1)
        with self.assertRaises(ValueError): mobile_vote([float('nan'),0,0])

    def test_weather_visible_snow_or_rain(self):
        obj={'labels':dict(labels(rain_snow=0),rain=-1,snow=1),
             'evidence':{k:'Visible observation' for k in LABELS+['rain','snow']}}
        parsed=parse_qwen(json.dumps(obj))
        self.assertEqual(parsed['labels']['rain_snow'],1)
        obj['labels'].update(rain=0,snow=-1)
        self.assertEqual(parse_qwen(json.dumps(obj))['labels']['rain_snow'],-1)
        obj['labels'].update(rain=0,snow=0)
        self.assertEqual(parse_qwen(json.dumps(obj))['labels']['rain_snow'],0)

    def test_malformed_qwen_not_accepted(self):
        with self.assertRaises(ValueError): parse_qwen('{"labels":{}}')
        obj={'labels':dict(labels(),rain=0,snow=0),'evidence':{k:'x' for k in LABELS+['rain','snow']}}
        obj['labels']['night']=True
        with self.assertRaises(ValueError): parse_qwen(json.dumps(obj))

    def test_store_resume_model_and_image_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            item={'id':'a','image_sha256':'hash'}
            store=RecordStore(Path(tmp)/'r.db','sig')
            self.assertIsNone(store.get(item))
            store.put(item,{'labels':labels()})
            self.assertEqual(store.get(item)['labels'],labels())
            with self.assertRaises(ValueError): store.get(dict(item,image_sha256='different'))
            store.close()
            wrong=RecordStore(Path(tmp)/'r.db','wrong',readonly=True)
            with self.assertRaises(ValueError): wrong.get(item)
            wrong.close()

    def test_full_pipeline_more_than_preview_no_source_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); source=base/'source'; source.mkdir(); out=base/'out'
            dump(source/'summary.json',{'schema':'nas8_weather_rev10'})
            dump(source/'PREPARED.json',{});dump(source/'source_audit.json',{})
            splitrows={s:[] for s in ('train','val','test')}
            sources=['coco','places365','seg13','mirflickr','nuswide','10_scenes']
            for i in range(204):
                path=base/('%03d.jpg'%i); Image.new('RGB',(40,30),(i%256,30,40)).save(path)
                split=('train','val','test')[i%3]
                splitrows[split].append({'image':str(path),'split':split,'group_id':str(i),'source':sources[i%6],
                                        'labels':labels(landscape=1,sports=1,outdoor=1),'evidence':{}})
            for s,rr in splitrows.items():
                (source/(s+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rr))
            before={p.name:sha(p) for p in source.iterdir()}
            prepare_run(argparse.Namespace(input_root=source,output_root=out,seed=20260917))
            root,plan,items=load_run(out)
            self.assertEqual(len(items),204); self.assertEqual(sum(r['preview'] for r in items),200)
            meta={'selection':plan['items_digest']}
            folder,sig=stage_lock(out,'mobile',meta); store=RecordStore(folder/'results.sqlite',sig)
            for r in items:
                y=labels(landscape=1,sports=1,outdoor=1)
                store.put(r,{'labels':y,'legacy_labels':labels(landscape=0,sports=0,outdoor=1),
                  'display_scores':{k:.9 for k in LABELS},'cosine_margins':{k:[.03]*3 for k in LABELS},'status':'OK'})
            store.close()
            for shard in (0,1):
                folder,sig=stage_lock(out,'qwen_%02d'%shard,dict(meta,shards=2,shard=shard))
                store=RecordStore(folder/'results.sqlite',sig)
                for i,r in enumerate(items):
                    if i%2==shard: store.put(r,{'labels':labels(landscape=1,sports=1,outdoor=1),'evidence':{k:'test' for k in LABELS},'status':'OK'})
                store.close()
            merge_run(argparse.Namespace(output_root=out,shards=2,preview_only=True))
            self.assertEqual(read(out/'preview200/summary.json')['records'],200)
            merge_run(argparse.Namespace(output_root=out,shards=2,preview_only=False))
            self.assertEqual(read(out/'merged/summary.json')['records'],204)
            self.assertEqual(len(list((out/'merged/review200').glob('*.jpg'))),200)
            self.assertEqual(before,{p.name:sha(p) for p in source.iterdir()})
            self.assertTrue((out/'merged/original_evaluation/test.jsonl').exists())
            merge_run(argparse.Namespace(output_root=out,shards=2,preview_only=False))
            # Corrupt source => resumability refuses silently changed labels.
            (source/'summary.json').write_text('{}')
            with self.assertRaises(ValueError): load_run(out)


if __name__=='__main__': unittest.main()
