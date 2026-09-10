#!/usr/bin/env python3
"""Evaluate frozen MobileCLIP2-S0 against existing NAS8 clean V3 manifests."""
import argparse
import csv
import importlib.metadata as md
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from core import LABELS, NAMES, PROMPTS, audit, sha, protocol


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--split', choices=['val','test'], default='test')
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--limit', type=int, default=0, help='Smoke only; limited results never count as full benchmark')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--precision', choices=['fp32','fp16'], default='fp32')
    p.add_argument('--insecure-downloads', action='store_true', help='Explicit user-authorized process-local HF TLS bypass')
    p.add_argument('--checkpoint', type=Path, help='Optional locally supplied OpenCLIP safetensors checkpoint')
    p.add_argument('--revision', default='main')
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--runs', type=int, default=100)
    args = p.parse_args()
    if args.limit < 0 or args.batch_size < 1 or args.warmup < 0 or args.runs < 1:
        p.error('Invalid size/limit/timing arguments')
    return args


def run(a):
    source_root, output = a.label_root.resolve(), a.output_dir.resolve()
    if output == source_root or source_root in output.parents or output in source_root.parents:
        raise ValueError('Output must not overlap the source label directory')
    a.output_dir.mkdir(parents=True, exist_ok=False)
    rows, strict, data = audit(a.label_root, a.split)
    dump(a.output_dir/'data_audit.json', data)
    if a.limit:
        rows = rows[:a.limit]
    # Decode all selected input images before expensive model load. Never silently skip failures.
    for i,r in enumerate(rows):
        with Image.open(r['image']) as im:
            im.convert('RGB').load()
        if (i+1)%1000==0:
            print('DATA_PREFLIGHT',i+1,len(rows),flush=True)
    coverage = protocol(rows, strict, np.full((len(rows),8),0.5))
    # No dummy-score metrics are persisted or presented as results.
    dump(a.output_dir/'coverage.json', {k:[{f:r[f] for f in ['label','positive','negative','unknown','status']} for r in v['per_class']] for k,v in coverage.items()})
    if a.preflight_only:
        dump(a.output_dir/'STATUS.json', {'status':'DATA_PREFLIGHT_PASS', 'gpu_tested':False, **data})
        print('DATA_PREFLIGHT_PASS',flush=True)
        return

    import torch
    import torchvision
    import open_clip
    import timm
    from timm.utils import reparameterize_model
    from huggingface_hub import hf_hub_download, HfApi, configure_http_backend
    if not torch.cuda.is_available():
        raise RuntimeError('BLOCKED_ENV: CUDA unavailable; CPU fallback is forbidden')
    if md.version('open_clip_torch')!='3.2.0' or timm.__version__!='1.0.20':
        raise RuntimeError('Expected pinned OpenCLIP 3.2.0 and timm 1.0.20')
    torch.set_num_threads(2)
    torch.manual_seed(20260910)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if a.insecure_downloads:
        import requests
        def factory():
            session=requests.Session()
            session.verify=False
            return session
        configure_http_backend(backend_factory=factory)
        print('TLS verification disabled for this HF download process by explicit flag',flush=True)
    source='timm/MobileCLIP2-S0-OpenCLIP'
    if a.checkpoint:
        checkpoint=a.checkpoint.resolve()
        revision='manual-local; verify provenance separately'
    else:
        revision=HfApi().model_info(source,revision=a.revision).sha
        checkpoint=Path(hf_hub_download(source,'open_clip_model.safetensors',revision=revision))
    if checkpoint.suffix!='.safetensors' or not checkpoint.is_file():
        raise ValueError('Supply the OpenCLIP open_clip_model.safetensors checkpoint')
    cfg=open_clip.get_pretrained_cfg('MobileCLIP2-S0','dfndr2b')
    kwargs=dict(image_mean=cfg['mean'],image_std=cfg['std'])
    if cfg.get('interpolation'):
        kwargs['image_interpolation']=cfg['interpolation']
    if cfg.get('resize_mode'):
        kwargs['image_resize_mode']=cfg['resize_mode']
    # Keep dfndr2b preprocessing even when loading from a local safetensors file.
    model,_,preprocess=open_clip.create_model_and_transforms('MobileCLIP2-S0',pretrained=str(checkpoint),**kwargs)
    model=model.cuda().eval()
    tokenizer=open_clip.get_tokenizer('MobileCLIP2-S0')
    with Image.open(rows[0]['image']) as im:
        sample=preprocess(im.convert('RGB')).unsqueeze(0).cuda()
    if tuple(sample.shape[1:])!=(3,256,256):
        raise RuntimeError('Unexpected pretrained input shape')
    with torch.inference_mode():
        before=model.encode_image(sample).float()
        model=reparameterize_model(model).eval()
        after=model.encode_image(sample).float()
        torch.testing.assert_close(after,before,rtol=2e-3,atol=2e-3)
    device='cuda:0'
    amp=lambda:torch.autocast('cuda',dtype=torch.float16,enabled=a.precision=='fp16')
    normalize=lambda x:torch.nn.functional.normalize(x.float(),dim=-1)
    with torch.inference_mode(),amp():
        pos,neg=[],[]
        for l in LABELS:
            for prompts,dst in zip(PROMPTS[l],(pos,neg)):
                features=normalize(model.encode_text(tokenizer(prompts).cuda()))
                dst.append(normalize(features.mean(0,keepdim=True))[0])
        pos,neg=torch.stack(pos),torch.stack(neg)
    scale=float(model.logit_scale.exp().detach().clamp(max=100).cpu())
    scores=[]
    t0=time.perf_counter()
    for start in range(0,len(rows),a.batch_size):
        images=[]
        for r in rows[start:start+a.batch_size]:
            with Image.open(r['image']) as im:
                images.append(preprocess(im.convert('RGB')))
        x=torch.stack(images).cuda()
        with torch.inference_mode(),amp():
            feat=normalize(model.encode_image(x))
        s=torch.sigmoid(scale*(feat@pos.T-feat@neg.T))
        if s.shape[1]!=8 or not torch.isfinite(s).all():
            raise RuntimeError('Invalid eight-label scores')
        scores.append(s.cpu().numpy())
        if start==0 or start%(a.batch_size*25)==0:
            print('INFERENCE',min(start+a.batch_size,len(rows)),len(rows),flush=True)
    scores=np.concatenate(scores)
    elapsed=time.perf_counter()-t0
    results=protocol(rows,strict,scores)
    dump(a.output_dir/'metrics.json',results)
    smap={r['image']:r for r in strict}
    with (a.output_dir/'predictions.csv').open('w',newline='') as f:
        fields=['image','source','group_id','split']+[k+'_'+l for l in LABELS for k in ('gt','strict_gt','score','pred')]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r,s in zip(rows,scores):
            d={k:r[k] for k in fields[:4]}
            for j,l in enumerate(LABELS):
                d.update({f'gt_{l}':r['labels'][l],f'strict_gt_{l}':smap.get(r['image'],{}).get('labels',{}).get(l,-1),f'score_{l}':float(s[j]),f'pred_{l}':int(s[j]>=0.5)})
            w.writerow(d)
    for name,result in results.items():
        with (a.output_dir/(name+'_per_class.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(result['per_class'][0]));w.writeheader();w.writerows(result['per_class'])
    with torch.inference_mode():
        for _ in range(a.warmup):
            with amp():model.encode_image(sample)
        torch.cuda.synchronize()
        times=[]
        for _ in range(a.runs):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            with amp():model.encode_image(sample)
            end.record();torch.cuda.synchronize();times.append(start.elapsed_time(end))
    latency={'batch_size':1,'mean_ms':float(np.mean(times)),'p95_ms':float(np.percentile(times,95)),
             'encoder_fps':1000/float(np.mean(times)),'precision':a.precision,
             'scope':'GPU image encoder only; excludes decode, preprocess, H2D, text and scoring; NOT V516 NPU',
             'image_scoring_loop_seconds':elapsed,'image_scoring_loop_images_per_second':len(rows)/elapsed}
    dump(a.output_dir/'latency.json',latency)
    dump(a.output_dir/'prompts.json',{'labels':LABELS,'names':NAMES,'prompts':PROMPTS,'threshold':0.5,
                                   'scoring':'sigmoid(scale*(positive_similarity-negative_similarity))','logit_scale':scale})
    try:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,text=True).strip()
    except subprocess.CalledProcessError:commit='unavailable'
    dump(a.output_dir/'environment.json',{'python':sys.version,'git_commit':commit,
        'packages':{n:md.version(n) for n in ['torch','torchvision','open_clip_torch','timm','huggingface_hub','numpy','scikit-learn']},
        'cuda_runtime':torch.version.cuda,'cudnn_loaded':torch.backends.cudnn.version(),
        'gpu':torch.cuda.get_device_name(0),'input_shape':list(sample.shape),'preprocess':str(preprocess),
        'pretrained_config':cfg,'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
        'checkpoint_source':source,'revision':revision,'reparameterization_fp32_check':'PASS'})
    # Side panel leaves the entire image visible, including edges.
    vis=a.output_dir/'error_visualizations';vis.mkdir()
    errors=[]
    for mode in ('strict','mapped'):
        for j,l in enumerate(LABELS):
            for kind in ('fp','fn'):
                ix=[]
                for i,r in enumerate(rows):
                    y=(smap.get(r['image'],{}).get('labels',{}) if mode=='strict' else r['labels']).get(l,-1)
                    if (kind=='fp' and y==0 and scores[i,j]>=0.5) or (kind=='fn' and y==1 and scores[i,j]<0.5):ix.append(i)
                ix=sorted(ix,key=lambda i:float(scores[i,j]),reverse=kind=='fp')[:8]
                for rank,i in enumerate(ix):
                    with Image.open(rows[i]['image']) as im:im=ImageOps.contain(im.convert('RGB'),(512,384))
                    canvas=Image.new('RGB',(850,420),'white');canvas.paste(im,(0,0));d=ImageDraw.Draw(canvas)
                    d.text((520,10),mode+' '+l+' '+kind,fill='black')
                    for k,label in enumerate(LABELS):
                        gt=(smap.get(rows[i]['image'],{}).get('labels',{}) if mode=='strict' else rows[i]['labels']).get(label,-1)
                        d.text((520,40+k*35),f'{label}: GT={gt}  score={scores[i,k]:.3f}',fill='black')
                    filename=f'{mode}_{l}_{kind}_{rank:02d}.jpg';canvas.save(vis/filename)
                    errors.append({'file':filename,'image':rows[i]['image'],'mode':mode,'label':l,'kind':kind,'evidence':rows[i]['evidence']})
    dump(vis/'index.json',errors)
    # Verify source manifests have not changed during inference.
    for name,digest in data['hashes'].items():
        if sha(a.label_root/name)!=digest:raise RuntimeError('Source labels changed during run')
    status='SMOKE_PASS' if a.limit else 'BENCHMARK_COMPLETE_DIAGNOSTIC'
    dump(a.output_dir/'STATUS.json',{'status':status,'evaluated_images':len(rows),'full_split_images':data['images'],
        'source_labels_modified':False,'coverage_strict':results['strict']['summary'],
        'human_action_required':bool(data['dataset_human_action_required'] or results['strict']['summary']['covered_count']<8),
        'note':'Missing strict labels and weak GT require review; execution completion is not product acceptance.'})
    lines=['# MobileCLIP2-S0 NAS8 clean V3 zero-shot', '',status,'',
           'Labels: '+', '.join(LABELS),'', 'Frozen prompts, threshold 0.5; unknown=-1 masked. No test tuning.',
           'Strict is native/reviewed GT; mapped includes weak rules. Not a fully audited eight-class acceptance test.',
           'Single-class labels retain recall/specificity only; F1/AP/AUC/accuracy are N/A. Macro covered != macro all8.',
           'Latest label semantics do not guarantee model correctness: inspect night, rain/snow and objective-image errors.',
           'Pretraining overlap cannot be excluded. GPU encoder latency is not NPU throughput.','']
    for mode in ('strict','mapped','real_photo_mapped_subset'):
        lines+=['## '+mode,'','|Label|Pos|Neg|Unknown|P|R|F1|AP|Status|','|---|---:|---:|---:|---:|---:|---:|---:|---|']
        fmt=lambda v:'N/A' if v is None else f'{v:.4f}'
        for r in results[mode]['per_class']:
            lines.append(f"|{r['label']}|{r['positive']}|{r['negative']}|{r['unknown']}|{fmt(r['precision'])}|{fmt(r['recall'])}|{fmt(r['f1'])}|{fmt(r['ap'])}|{r['status']}|")
        lines+=['',json.dumps(results[mode]['summary'],ensure_ascii=False),'']
    lines+=['## Dataset quality blockers','',json.dumps(data['training_quality_blockers'],ensure_ascii=False),'',
            '## Latency','',json.dumps(latency,ensure_ascii=False)]
    (a.output_dir/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(status, str(a.output_dir/'REPORT.md'),flush=True)


if __name__=='__main__':
    args=arguments()
    # Never overwrite a prior run, including its failure report.
    if args.output_dir.exists():raise SystemExit('Choose a NEW output directory')
    try:run(args)
    except Exception as e:
        import traceback
        if args.output_dir.is_dir():dump(args.output_dir/'FAILED.json',{'error':str(e),'traceback':traceback.format_exc(),'HUMAN_ACTION_REQUIRED':True})
        raise
