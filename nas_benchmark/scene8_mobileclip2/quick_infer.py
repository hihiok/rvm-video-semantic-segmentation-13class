#!/usr/bin/env python3
"""Small NAS8 MobileCLIP2-S0 image preview; no GT or prepared dataset required."""
import argparse
import csv
import html
import importlib.metadata as md
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from core import LABELS, NAMES, PROMPTS, sha

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
EXCLUDED_DIRS = {'mask', 'masks', 'label', 'labels', 'annotation', 'annotations',
                 'segmentation', 'segmentations', 'gt', 'groundtruth',
                 'semantic', 'semantic_masks', 'segmentation_masks'}


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def select_images(root, count, seed):
    """Reservoir sample paths with bounded memory; decode only sampled images."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError('Missing input image directory: '+str(root))
    if root.name.lower() in EXCLUDED_DIRS:
        raise ValueError('Input points to a mask/label directory')
    rng = random.Random(seed)
    selected, seen = [], 0
    def walk_error(error):
        raise error
    for folder, dirs, files in os.walk(root, followlinks=False, onerror=walk_error):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d.lower() not in EXCLUDED_DIRS)
        for name in sorted(files):
            p = Path(folder)/name
            stem = p.stem.lower()
            if p.suffix.lower() not in IMAGE_EXTENSIONS or stem.endswith(('_mask', '_label', '_seg', '_trainids', '_labelids')):
                continue
            seen += 1
            if len(selected) < count:
                selected.append(p)
            else:
                index = rng.randrange(seen)
                if index < count:
                    selected[index] = p
    if not selected:
        raise ValueError('No candidate photos; select the real image subtree')
    rows = []
    for p in selected:
        with Image.open(p) as im:
            if im.mode not in ('RGB', 'RGBA', 'CMYK'):
                raise ValueError('Sample is not a color photo (possible mask): '+str(p))
            if min(im.size) < 32:
                raise ValueError('Sample too small (possible mask): '+str(p))
            im.convert('RGB').load()
            rows.append({'image':str(p), 'width':im.width, 'height':im.height, 'sha256':sha(p)})
    return rows, seen


def render(rows, scores, output):
    output = Path(output)
    vis = output/'visualizations'
    vis.mkdir()
    cards = []
    with (output/'predictions.csv').open('w', newline='', encoding='utf-8-sig') as f:
        fields = ['image', 'visualization']+[k+'_'+l for l in LABELS for k in ('score','pred')]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i,(r,s) in enumerate(zip(rows,scores)):
            name = f'{i+1:03d}.jpg'
            with Image.open(r['image']) as im:
                photo = ImageOps.contain(im.convert('RGB'), (640,480))
            panel = Image.new('RGB',(1050, max(420,photo.height)), 'white')
            panel.paste(photo,(0,0))
            draw = ImageDraw.Draw(panel)
            draw.text((650,15), 'MobileCLIP2-S0 | eight independent labels', fill='black')
            draw.text((650,38), 'Preview threshold 0.5 | NO GT', fill='black')
            record = {'image':r['image'], 'visualization':'visualizations/'+name}
            descriptions = []
            for j,label in enumerate(LABELS):
                value = float(s[j]); pred = int(value>=0.5)
                draw.text((650,75+36*j), f'{label:16s} {value:.4f}  pred={pred}',
                          fill='darkgreen' if pred else 'gray')
                record['score_'+label]=value
                record['pred_'+label]=pred
                descriptions.append(f'{NAMES[j]}: {value:.4f} ({pred})')
            panel.save(vis/name, quality=92)
            writer.writerow(record)
            cards.append('<article><img src="visualizations/'+name+'"><p>'+
                         html.escape(r['image'])+'</p><p>'+html.escape(' | '.join(descriptions))+'</p></article>')
    # Contact sheets contain eight previews each; originals are not cropped.
    previews = sorted(vis.glob('*.jpg'))
    for start in range(0,len(previews),8):
        sheet = Image.new('RGB',(1050, 840),'#eeeeee')
        for j,p in enumerate(previews[start:start+8]):
            with Image.open(p) as im:
                tile=ImageOps.contain(im,(525,210))
                sheet.paste(tile,((j%2)*525,(j//2)*210))
        sheet.save(output/f'contact_{start//8+1:02d}.jpg',quality=92)
    (output/'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>MobileCLIP2-S0 NAS8 preview</title>'
        '<style>body{font:16px sans-serif;margin:24px;background:#f5f5f5}article{background:white;'
        'padding:16px;margin:16px 0}img{max-width:100%}p{overflow-wrap:anywhere}</style>'
        '<h1>MobileCLIP2-S0 八类预览</h1><p>无 GT；得分不是校准概率，阈值 0.5 仅供查看。'
        '本页面不提供准确率或正式 benchmark 指标。</p>'+''.join(cards),encoding='utf-8')


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,required=True,help='Confirmed original color photo directory; not masks')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--count',type=int,default=32)
    p.add_argument('--seed',type=int,default=20260914)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--revision',default='main')
    p.add_argument('--insecure-downloads',action='store_true')
    p.set_defaults(precision='fp32')
    a=p.parse_args()
    if not 1<=a.count<=100 or not 1<=a.batch_size<=32:
        p.error('Use 1..100 images and batch size 1..32')
    return a


def run(a):
    root,out=a.input_dir.resolve(),a.output_dir.resolve()
    if out==root or root in out.parents or out in root.parents:
        raise ValueError('Output must be separate from input dataset')
    out.mkdir(parents=True,exist_ok=False)
    rows,candidates=select_images(root,a.count,a.seed)
    dump(out/'selected_images.json',{'input_root':str(root),'seed':a.seed,
         'candidate_count':candidates,'selected_count':len(rows),'images':rows,
         'note':'Filename/color checks do not prove that an image is a photo; visually confirm samples.'})
    # Allows input verification and selection contact sheets before model downloads.
    selections=out/'selected_previews';selections.mkdir()
    for i,r in enumerate(rows):
        with Image.open(r['image']) as im:
            ImageOps.contain(im.convert('RGB'),(640,480)).save(selections/f'{i+1:03d}.jpg')
    if a.preflight_only:
        dump(out/'STATUS.json',{'status':'INPUT_PREFLIGHT_PASS','gpu_tested':False,'images':len(rows)})
        print('INPUT_PREFLIGHT_PASS',str(out),flush=True)
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
    try:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,text=True).strip()
    except subprocess.CalledProcessError:commit='unavailable'
    dump(a.output_dir/'environment.json',{'python':sys.version,'git_commit':commit,
        'packages':{n:md.version(n) for n in ['torch','torchvision','open_clip_torch','timm','huggingface_hub','numpy','scikit-learn']},
        'cuda_runtime':torch.version.cuda,'cudnn_loaded':torch.backends.cudnn.version(),
        'gpu':torch.cuda.get_device_name(0),'input_shape':list(sample.shape),'preprocess':str(preprocess),
        'pretrained_config':cfg,'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
        'checkpoint_source':source,'revision':revision,'reparameterization_fp32_check':'PASS'})

    render(rows,scores,a.output_dir)
    dump(a.output_dir/'prompts.json',{'labels':LABELS,'names':NAMES,'prompts':PROMPTS,
         'threshold':0.5,'logit_scale':scale,
         'scoring':'sigmoid(scale*(positive_similarity-negative_similarity)); not calibrated probability'})
    for row in rows:
        if sha(row['image'])!=row['sha256']:
            raise RuntimeError('Source image changed during run: '+row['image'])
    dump(a.output_dir/'STATUS.json',{'status':'QUICK_INFERENCE_COMPLETE','images':len(rows),
         'requested_images':a.count,'gpu_tested':True,'precision':'fp32','gt_used':False,
         'source_images_modified':False,'prediction_threshold':0.5,
         'inference_loop_seconds':elapsed,'human_action_required':False,
         'note':'Execution check and qualitative preview only; no accuracy/F1/mAP claims.'})
    (a.output_dir/'REPORT.md').write_text(
        '# MobileCLIP2-S0 八类快速推理\n\nQUICK_INFERENCE_COMPLETE\n\n'
        f'图片：{len(rows)}；FP32；输入：256×256（官方预处理）。\n\n'
        '查看 index.html、contact_*.jpg 和 visualizations/，逐图得分见 predictions.csv。\n\n'
        '无八类 GT，没有计算准确率/F1/mAP；0.5 为预览阈值，得分不是校准概率。\n'
        '请重点查看夜景误报、自然风景主体、办公与显示器/客观图混淆。\n',encoding='utf-8')
    print('QUICK_INFERENCE_COMPLETE',str(a.output_dir/'index.html'),flush=True)


if __name__=='__main__':
    args=arguments()
    if args.output_dir.exists():
        raise SystemExit('Choose a NEW output directory')
    try:
        run(args)
    except Exception as e:
        import traceback
        if args.output_dir.is_dir():
            dump(args.output_dir/'FAILED.json',{'error':str(e),'traceback':traceback.format_exc(),
                 'HUMAN_ACTION_REQUIRED':True})
        raise
