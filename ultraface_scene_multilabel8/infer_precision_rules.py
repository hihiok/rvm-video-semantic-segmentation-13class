#!/usr/bin/env python3
"""Fresh VAL/TEST inference with precision-oriented NAS8 decisions and mandatory nonempty output."""
import argparse,html,json,math,time,zipfile
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from rev10_data import LABELS,dump,sha
from model import create_ultraface_slim_scene8
from train_rev10 import Scenes,evaluate,metrics,strict_gt,save_csv,torch_load,worker_init
from visualize_test_rev10 import read_inputs,select_cases,render_card,detail_table,result,page,NAMES
from decision_policy import tune_thresholds,decide,top1_099

STAGES=('original_calibrated','all_099','top1_plus_099','precision_only','precision_context')

def make_gallery(out,rows,gt,scores,thresholds,stages,notes,wh,n,seed):
 folder=out/'gallery';folder.mkdir();th=thresholds;final=stages[STAGES[-1]]
 cases,stats=select_cases(gt,scores,th,n,seed,decisions=final);selected=[]
 home=['<p>根据最终分类别阈值＋有限冲突处理，分类别抽取正确和错误。GT未知不计对错。每张卡片展示各策略在同一图片的输出。</p><p>各类先使用独立阈值；无类别达标才用最高分兜底，标记未达阈值。室内/户外只保留分数较高者，分数接近时标记歧义。室内与夜景、雨雪不互斥。弱GT不等于人工确认。</p>']
 for j,label in enumerate(LABELS):
  home.append('<h2>'+NAMES[label]+'</h2><ul>')
  for group in ('correct','error'):
   groupdir=folder/label/group;groupdir.mkdir(parents=True);cards=[];paths=[]
   for number,i in enumerate(cases[label][group],1):
    r=rows[i];status=result(int(gt[i,j]),bool(final[i,j]));name='%02d_%s.jpg'%(number,status)
    policy_note='VAL thresholds | * = context suppression | '+('FALLBACK TOP1: below threshold' if notes[i]['fallback_top1'] else 'Threshold-qualified output')
    render_card(groupdir/name,r,scores[i],th,label,status,wh,decisions=final[i],rule_notes=notes[i]['suppressed'],policy_note=policy_note);paths.append(groupdir/name)
    body='<table><tr><th>方案</th><th>预测正类别</th></tr>'
    for stage in STAGES:body+='<tr><td>'+stage+'</td><td>'+html.escape(', '.join(k for k,p in zip(LABELS,stages[stage][i]) if p) or '(none)')+'</td></tr>'
    body+='</table><p class="warn">规则压制：'+html.escape(json.dumps(notes[i],ensure_ascii=False))+'</p>'
    body+=detail_table(r,scores[i],th,decisions=final[i])
    cards.append('<div class="card"><a href="'+name+'"><img loading="lazy" src="'+name+'"></a>'+body+'<p>'+html.escape(r['image'])+'</p></div>')
    selected.append({'focus':label,'group':group,'outcome':status,'image':r['image'],'card':str((groupdir/name).relative_to(out))})
   if paths:
    sheet=Image.new('RGB',(1360,530*((len(paths)+1)//2)),'white')
    for k,p in enumerate(paths):
     with Image.open(p) as im:sheet.paste(im.resize((680,530)),((k%2)*680,(k//2)*530))
    sheet.save(groupdir/'contact.jpg',quality=90)
   info=stats[label][group]
   (groupdir/'index.html').write_text(page(label+' '+group,'<a href="../../index.html">返回分类总览</a><p>抽取%d，缺%d（不重复凑数）</p><div class="grid">'%(info['selected'],info['shortfall'])+''.join(cards)+'</div>'),encoding='utf-8')
   home.append('<li><a href="%s/%s/index.html">%s：%d张，缺%d</a></li>'%(label,group,group,info['selected'],info['shortfall']))
  home.append('</ul>')
 (folder/'index.html').write_text(page('测试GT与预测策略对比', ''.join(home)),encoding='utf-8')
 dump(out/'gallery_audit.json',{'by_class':stats,'display_records':len(selected),'unique_images':len({r['image'] for r in selected}),'selected':selected})
 return selected


def run(run_dir, data_root, out, device='cuda:0', batch=128, workers=2, n=10,
        seed=20260917, target_precision=.99, min_predictions=30, conflict_margin=.05):
    run_dir = Path(run_dir).resolve(); data_root = Path(data_root).resolve(); out = Path(out).resolve()
    archive = out.with_suffix('.zip')
    if out.exists() or archive.exists() or out == run_dir or out == data_root or data_root in out.parents or out in run_dir.parents:
        raise ValueError('Use a new output outside the label root and not above existing artifacts')
    if batch < 1 or workers < 0 or n < 1:
        raise ValueError('Invalid sizes')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; --device cpu is supported')
    inputs = {split: read_inputs(run_dir, data_root, split) for split in ('val','test')}
    val_rows, val_gt, val_cached, original, config, _, _ = inputs['val']
    rows, gt, cached, original_test, config_test, _, _ = inputs['test']
    if not np.array_equal(original, original_test) or config != config_test:
        raise ValueError('VAL/TEST shared configuration changed')
    if {r['image'] for r in rows} & {r['image'] for r in val_rows}:
        raise ValueError('VAL/TEST image overlap')
    paths = {}; hashes = {}
    for split, data in inputs.items():
        paths.update({split+'_'+k: v for k,v in data[5].items()})
        hashes.update({split+'_'+k: v for k,v in data[6].items()})
    checkpoint = run_dir/'best_ultraface_scene8.pth'
    paths['checkpoint'] = checkpoint; hashes['checkpoint'] = sha(checkpoint)
    ck = torch_load(checkpoint); wh = config['input_wh']
    if ck.get('labels') != LABELS or ck.get('input_hw') != [wh[1],wh[0]]:
        raise ValueError('Checkpoint metadata mismatch')
    if not np.array_equal(np.array([ck['thresholds'][k] for k in LABELS]), original):
        raise ValueError('Checkpoint/JSON calibration mismatch')
    torch.set_num_threads(2)
    import cv2
    cv2.setNumThreads(0)
    model = create_ultraface_slim_scene8().eval()
    model.load_state_dict(ck['model'], strict=True); model.to(device)

    def infer(records, expected, split):
        loader = DataLoader(Scenes(records,wh), batch_size=batch, shuffle=False,
                            num_workers=workers, pin_memory=device.startswith('cuda'),
                            worker_init_fn=worker_init,
                            **({'multiprocessing_context':'spawn'} if workers else {}))
        print('FRESH_INFERENCE',split,len(records),device,'FP32',flush=True)
        started = time.time()
        actual_gt, scores = evaluate(model, loader, torch.device(device), False)
        if not np.array_equal(actual_gt, expected):
            raise ValueError('Inference GT ordering mismatch')
        print('INFERENCE_SECONDS',split,time.time()-started,flush=True)
        return scores

    out.mkdir(parents=True, mode=0o700)
    try:
        val_scores = infer(val_rows, val_gt, 'val')
        th, tuning = tune_thresholds(val_gt, val_scores, original, target_precision, min_predictions)
        # This artifact is frozen BEFORE TEST forward inference and is never tuned on TEST.
        policy = {'labels':LABELS, 'thresholds':dict(zip(LABELS,map(float,th))),
                  'tuning_split':'val', 'target_precision':target_precision,
                  'min_predictions':min_predictions, 'threshold_floor':'max(original_threshold,0.5)',
                  'fallback':'disable threshold-based emission when target unmet or support insufficient; image-level top1 fallback remains',
                  'top1_fallback':'if no class passes its threshold, emit highest raw score; stable LABELS tie order', 'max_label_count':None,
                  'conflict_pair':['indoor','outdoor'], 'conflict_margin':conflict_margin,
                  'conflict_measure':'higher raw probability wins; exact ties favor indoor; margin only flags ambiguity',
                  'close_conflict_action':'keep winner, flag ambiguous when score gap <= margin',
                  'dominant_conflict_action':'withhold weaker indoor/outdoor label; never emit both',
                  'flag_only_pairs':['indoor+landscape','objective_image+any_scene'],
                  'allowed_cooccurrences':['indoor+night','indoor+rain_snow','sports+other_scenes'],
                  'scope':'experimental output policy; never changes GT/raw probability/weights',
                  'precision_notice':'Empirical on known VAL labels, including weak labels; not a deployment guarantee'}
        dump(out/'decision_policy.json', policy); dump(out/'threshold_selection.json', tuning)
        policy_hash = sha(out/'decision_policy.json')
        scores = infer(rows, gt, 'test')
        summaries = {}; comparisons = []; final_notes = None; test_stages = None
        for split, records, labels, sc in [('val',val_rows,val_gt,val_scores),('test',rows,gt,scores)]:
            final, notes = decide(sc, th, conflict_margin)
            stages = dict(zip(STAGES,[sc >= original, sc >= .99, top1_099(sc), sc >= th, final]))
            strict = strict_gt(records)
            for stage, decisions in stages.items():
                thresholds = original if stage == STAGES[0] else (np.full(8,.99) if stage in STAGES[1:3] else th)
                per, summary = metrics(labels, sc, thresholds, decisions=decisions)
                summary.update(mean_predicted_labels=float(decisions.sum(1).mean()),
                               empty_prediction_images=int(sum(decisions.sum(1)==0)),
                               fp_total=sum(r['fp'] for r in per), fn_total=sum(r['fn'] for r in per),
                               ap_notice='AP uses unchanged scores; identical across policies. Withheld positive GT counts as FN.')
                save_csv(out/(split+'_'+stage+'_per_class.csv'),per)
                dump(out/(split+'_'+stage+'_summary.json'),summary)
                if split == 'test': summaries[stage] = summary
                comparisons.extend({'split':split,'stage':stage,**r} for r in per)
                strict_per, strict_summary = metrics(strict,sc,thresholds,decisions=decisions)
                save_csv(out/(split+'_'+stage+'_strict_per_class.csv'),strict_per)
                dump(out/(split+'_'+stage+'_strict_summary.json'),strict_summary)
            np.savez_compressed(out/(split+'_inference.npz'),image=np.array([r['image'] for r in records]),
                                gt=labels,scores=sc,**stages)
            if split == 'test': test_stages = stages; final_notes = notes
        save_csv(out/'comparison.csv',comparisons)
        with (out/'predictions.jsonl').open('w',encoding='utf-8') as f:
            for i,r in enumerate(rows):
                rec = {'image':r['image'],'source':r['source'],'gt':r['labels'],'evidence':r.get('evidence',{}),
                       'probabilities':dict(zip(LABELS,map(float,scores[i]))), 'context':final_notes[i],
                       'empty_output':not bool(test_stages[STAGES[-1]][i].any())}
                rec.update({stage:[k for k,p in zip(LABELS,decisions[i]) if p] for stage,decisions in test_stages.items()})
                f.write(json.dumps(rec,ensure_ascii=False)+'\n')
        # Expose the accuracy cost of suppressions; never hide withheld labels from metrics.
        transitions = {}
        before = test_stages['precision_only']; after = test_stages['precision_context']
        for j,k in enumerate(LABELS):
            removed = before[:,j] & ~after[:,j]
            transitions[k] = {'removed_predictions':int(sum(removed)),
                              'removed_known_fp':int(sum(removed & (gt[:,j]==0))),
                              'removed_known_tp_became_fn':int(sum(removed & (gt[:,j]==1))),
                              'removed_unknown_gt':int(sum(removed & (gt[:,j]==-1))),
                              'fallback_added_predictions':int(sum(~before[:,j] & after[:,j])),
                              'fallback_added_known_fp':int(sum(~before[:,j] & after[:,j] & (gt[:,j]==0))),
                              'fallback_added_known_tp':int(sum(~before[:,j] & after[:,j] & (gt[:,j]==1)))}
        audit = {'status':'INFERENCE_COMPLETE','device':device,'precision':'FP32','test_images':len(rows),
                 'validation_images':len(val_rows),'input_wh':wh,'input_files':{k:str(v) for k,v in paths.items()},
                 'input_sha256':hashes,'decision_policy_sha256':policy_hash,'policy':policy,
                 'summaries':summaries,'threshold_selection':tuning,'rule_transitions':transitions,
                 'ambiguous_events':dict(Counter(k for v in final_notes for k in v['ambiguities'])),
                 'fallback_top1_images':sum(v['fallback_top1'] for v in final_notes),
                 'indoor_outdoor_output_overlap':int(sum(after[:,1] & after[:,4])),
                 'gt_indoor_outdoor_both_positive':int(sum((gt[:,1]==1)&(gt[:,4]==1))),
                 'raw_ap_unchanged_across_policies':True,
                 'fresh_vs_cached_max_abs_difference':{'val':float(np.max(np.abs(val_scores-val_cached))),
                                                       'test':float(np.max(np.abs(scores-cached)))},
                 'fresh_vs_cached_note':'FP32 vs previous AMP may differ; all current comparisons use the same fresh scores',
                 'test_tuning_notice':'Design informed by already viewed TEST; diagnostic comparison, not untouched generalization',
                 'weak_weather_notice':'Rule-based weather negatives remain unverified; strict metrics exclude these labels',
                 'prediction_zero_notice':'Output 0 means not emitted, not confirmed absent; GT -1 remains unknown'}
        selected = make_gallery(out, rows, gt, scores, th, test_stages, final_notes, wh, n, seed)
        for k,p in paths.items():
            if sha(p) != hashes[k]: raise ValueError('Input changed during inference: '+k)
        if sha(out/'decision_policy.json') != policy_hash:
            raise ValueError('Frozen policy modified after TEST')
        audit['input_unchanged'] = True; audit['visualization_records'] = len(selected)
        dump(out/'summary.json',audit)
        body = '<p><a href="gallery/index.html">打开每类10正确／10错误对照图</a></p><p>分类别验证阈值；无类别达标才Top1兜底（至少一个输出）。室内/户外只留较高分，接近则标歧义。其他类别独立判断。GT未知不评分，规则造成的漏报正常计入。</p><table><tr><th>策略</th><th>macro F1</th><th>FP</th><th>FN</th><th>平均输出类别</th><th>空输出图</th></tr>'
        for stage,s in summaries.items():
            body += '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (stage,s['macro_f1'],s['fp_total'],s['fn_total'],s['mean_predicted_labels'],s['empty_prediction_images']))+'</tr>'
        body += '</table><p class="warn">validation 的99% precision是选择目标而非真实准确率承诺；弱GT和已查看的test会限制结论。兜底标签虽未达阈值仍输出，相关误报会单列报告。</p>'
        (out/'index.html').write_text(page('NAS8 保守推理与GT对照',body),encoding='utf-8')
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out.rglob('*')):
                if p.is_file(): z.write(p,str(Path(out.name)/p.relative_to(out)))
        print('INFERENCE_COMPLETE',out,'ZIP',archive,flush=True)
        return audit
    except Exception as e:
        dump(out/'BLOCKED.json',{'error':str(e)}); raise

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch-size',type=int,default=128);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--per-group',type=int,default=10);p.add_argument('--seed',type=int,default=20260917)
    p.add_argument('--target-precision',type=float,default=.99);p.add_argument('--min-predictions',type=int,default=30)
    p.add_argument('--conflict-margin',type=float,default=.05)
    a = p.parse_args()
    run(a.run_dir,a.data_root,a.output_dir,a.device,a.batch_size,a.workers,a.per_group,a.seed,
        a.target_precision,a.min_predictions,a.conflict_margin)
