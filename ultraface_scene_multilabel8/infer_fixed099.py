#!/usr/bin/env python3
"""Fresh TEST inference with fixed .99 confidence, top1 fallback and indoor/outdoor exclusion."""
import argparse,json,time,zipfile
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader
from rev10_data import LABELS,dump,sha
from train_rev10 import Scenes,evaluate,metrics,strict_gt,save_csv,torch_load,worker_init
from model import create_ultraface_slim_scene8
from decision_policy import decide
from visualize_test_rev10 import read_inputs,page
from infer_precision_rules import make_gallery


def run(run_dir,data_root,out,device='cuda:0',batch=128,workers=2,n=10,seed=20260917):
    run_dir=Path(run_dir).resolve();data_root=Path(data_root).resolve();out=Path(out).resolve()
    archive=out.with_suffix('.zip')
    if out.exists() or archive.exists() or out==run_dir or out==data_root or data_root in out.parents or out in run_dir.parents:
        raise ValueError('Use a new output outside the label root')
    if batch<1 or workers<0 or n<1:raise ValueError('Invalid sizes')
    if device.startswith('cuda') and not torch.cuda.is_available():raise ValueError('CUDA unavailable; CPU supported')
    # Read TEST only. No validation selection and no dependency on VAL predictions.
    rows,gt,cached,original,config,paths,hashes=read_inputs(run_dir,data_root)
    checkpoint=run_dir/'best_ultraface_scene8.pth';paths['checkpoint']=checkpoint;hashes['checkpoint']=sha(checkpoint)
    ck=torch_load(checkpoint);wh=config['input_wh']
    if ck.get('labels')!=LABELS or ck.get('input_hw')!=[wh[1],wh[0]]:raise ValueError('Checkpoint metadata mismatch')
    if not np.array_equal(np.array([ck['thresholds'][k] for k in LABELS]),original):raise ValueError('Original threshold mismatch')
    torch.set_num_threads(2)
    import cv2
    cv2.setNumThreads(0)
    model=create_ultraface_slim_scene8().eval();model.load_state_dict(ck['model'],strict=True);model.to(device)
    loader=DataLoader(Scenes(rows,wh),batch_size=batch,shuffle=False,num_workers=workers,
                      pin_memory=device.startswith('cuda'),worker_init_fn=worker_init,
                      **({'multiprocessing_context':'spawn'} if workers else {}))
    thresholds=np.full(8,.99)
    policy={'labels':LABELS,'thresholds':dict.fromkeys(LABELS,.99),'mode':'fixed_confidence',
            'validation_threshold_selection':False,'minimum_output_labels':1,
            'top1_fallback':'only if no probability >=0.99; stable LABELS order breaks ties',
            'exclusive_pair':['indoor','outdoor'],'conflict_resolution':'keep higher probability; exact ties keep indoor',
            'other_hard_exclusions':[],'close_score_gap_flag_only':.05,
            'confidence_notice':'0.99 is the model score threshold, not measured precision or guaranteed accuracy'}
    out.mkdir(parents=True,mode=0o700)
    try:
        dump(out/'decision_policy.json',policy)
        print('FRESH_TEST_INFERENCE',len(rows),device,'FP32',flush=True);start=time.time()
        actual,scores=evaluate(model,loader,torch.device(device),False)
        if not np.array_equal(actual,gt):raise ValueError('GT ordering mismatch')
        print('INFERENCE_SECONDS',time.time()-start,flush=True)
        final,notes=decide(scores,thresholds)
        stages={'original_calibrated':scores>=original,'fixed_099_only':scores>=thresholds,'fixed_099_context':final}
        strict=strict_gt(rows);summaries={};comparison=[]
        for name,decisions in stages.items():
            th=original if name=='original_calibrated' else thresholds
            per,summary=metrics(gt,scores,th,decisions=decisions)
            summary.update(empty_prediction_images=int(sum(decisions.sum(1)==0)),
                           indoor_outdoor_overlap=int(sum(decisions[:,1]&decisions[:,4])),
                           mean_predicted_labels=float(decisions.sum(1).mean()),
                           fp_total=sum(r['fp'] for r in per),fn_total=sum(r['fn'] for r in per))
            summaries[name]=summary;comparison.extend({'stage':name,**r} for r in per)
            save_csv(out/(name+'_per_class.csv'),per);dump(out/(name+'_summary.json'),summary)
            sp,ss=metrics(strict,scores,th,decisions=decisions)
            save_csv(out/(name+'_strict_per_class.csv'),sp);dump(out/(name+'_strict_summary.json'),ss)
        save_csv(out/'comparison.csv',comparison)
        np.savez_compressed(out/'test_inference.npz',image=np.array([r['image'] for r in rows]),gt=gt,scores=scores,**stages)
        with (out/'predictions.jsonl').open('w',encoding='utf-8') as f:
            for i,r in enumerate(rows):
                record={'image':r['image'],'source':r['source'],'gt':r['labels'],'evidence':r.get('evidence',{}),
                        'probabilities':dict(zip(LABELS,map(float,scores[i]))),'context':notes[i]}
                record.update({k:[label for label,yes in zip(LABELS,decisions[i]) if yes] for k,decisions in stages.items()})
                f.write(json.dumps(record,ensure_ascii=False)+'\n')
        transitions={};before=stages['fixed_099_only']
        for j,label in enumerate(LABELS):
            removed=before[:,j]&~final[:,j];added=~before[:,j]&final[:,j]
            transitions[label]={'suppressed_fp':int(sum(removed&(gt[:,j]==0))),
                                'suppressed_tp_became_fn':int(sum(removed&(gt[:,j]==1))),
                                'fallback_added_tp':int(sum(added&(gt[:,j]==1))),
                                'fallback_added_fp':int(sum(added&(gt[:,j]==0))),
                                'fallback_added_unknown':int(sum(added&(gt[:,j]==-1)))}
        selected=make_gallery(out,rows,gt,scores,thresholds,stages,notes,wh,n,seed,fixed_confidence=.99)
        for k,p in paths.items():
            if sha(p)!=hashes[k]:raise ValueError('Input changed: '+k)
        audit={'status':'INFERENCE_COMPLETE','input_unchanged':True,'input_sha256':hashes,'input_wh':wh,
               'input_files':{k:str(v) for k,v in paths.items()},'test_images':len(rows),'device':device,
               'inference_precision':'FP32','policy':policy,'summaries':summaries,'rule_transitions':transitions,
               'fallback_top1_images':sum(note['fallback_top1'] for note in notes),
               'suppressed_labels':dict(Counter(k for note in notes for k in note['suppressed'])),
               'ambiguous_events':dict(Counter(k for note in notes for k in note['ambiguities'])),
               'visualization_records':len(selected),'fresh_vs_cached_max_abs_difference':float(np.max(np.abs(scores-cached))),
               'notice':'TEST diagnostic; partial/weak GT remain; strict weather negatives unsupported; raw AP unchanged across policies'}
        if summaries['fixed_099_context']['empty_prediction_images'] or summaries['fixed_099_context']['indoor_outdoor_overlap']:
            raise AssertionError('Nonempty and exclusive output invariants failed')
        dump(out/'summary.json',audit)
        body='<p>固定confidence≥0.99；无达标类别时最高分兜底；indoor/outdoor同时达标只保留较高分。无validation阈值选择。</p><p><a href="gallery/index.html">每类10正确／10错误GT对照图</a></p><table><tr><th>方案</th><th>macro F1</th><th>FP</th><th>FN</th><th>空输出</th><th>室内户外共存</th></tr>'
        for name,s in summaries.items():
            body+='<tr>'+''.join('<td>'+str(v)+'</td>' for v in (name,s['macro_f1'],s['fp_total'],s['fn_total'],s['empty_prediction_images'],s['indoor_outdoor_overlap']))+'</tr>'
        body+='</table><p>兜底次数：%d。0.99分数不代表99%%准确率。未知GT不判对错，弱标签证据单独标注。</p>'%audit['fallback_top1_images']
        (out/'index.html').write_text(page('NAS8 固定0.99推理',body),encoding='utf-8')
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out.rglob('*')):
                if p.is_file():z.write(p,str(Path(out.name)/p.relative_to(out)))
        print('INFERENCE_COMPLETE',out,'ZIP',archive,flush=True);return audit
    except Exception as e:
        dump(out/'BLOCKED.json',{'error':str(e)});raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch-size',type=int,default=128);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--per-group',type=int,default=10);p.add_argument('--seed',type=int,default=20260917)
    a=p.parse_args();run(a.run_dir,a.data_root,a.output_dir,a.device,a.batch_size,a.workers,a.per_group,a.seed)
