"""Full MobileCLIP2-S0 vote, using the user's proven OpenCLIP environment/cache."""
import argparse
import importlib.metadata as md
from pathlib import Path
import time
from common import load_run, stage_lock, RecordStore, sha, mobile_vote, dump
from prompts import LABELS, PAIRS
from legacy_prompts import LEGACY_PROMPTS


def run(a):
    import torch
    import open_clip
    from PIL import Image, ImageOps
    from huggingface_hub import hf_hub_download
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
    for package, version in [('open_clip_torch','3.2.0'),('timm','1.0.20')]:
        if md.version(package) != version: raise RuntimeError('Use the existing pinned MobileCLIP environment')
    torch.set_num_threads(2)
    torch.manual_seed(20260917)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root, plan, items = load_run(a.output_root)
    checkpoint = a.checkpoint or Path(hf_hub_download('timm/MobileCLIP2-S0-OpenCLIP', 'open_clip_model.safetensors', local_files_only=True))
    if checkpoint.suffix != '.safetensors': raise ValueError('Expected OpenCLIP safetensors checkpoint')
    meta = {'selection':plan['items_digest'], 'checkpoint_sha256':sha(checkpoint), 'model':'MobileCLIP2-S0',
            'checkpoint_source':'timm/MobileCLIP2-S0-OpenCLIP', 'checkpoint':str(checkpoint.resolve()),
            'pairs':PAIRS, 'legacy_prompts':LEGACY_PROMPTS, 'margin':a.margin,
            'precision':'fp32', 'packages':{n:md.version(n) for n in ('torch','torchvision','open_clip_torch','timm')},
            'scoring':'per-label cosine positive-minus-negative; sigmoid display score is NOT calibrated probability'}
    folder, signature = stage_lock(root, 'mobile', meta)
    store = RecordStore(folder/'results.sqlite', signature)
    work = items[:a.limit] if a.limit else items
    work = [r for r in work if store.get(r) is None]
    if not work:
        store.close(); print('MOBILE_REUSED', flush=True); return
    cfg = open_clip.get_pretrained_cfg('MobileCLIP2-S0','dfndr2b')
    kwargs = dict(image_mean=cfg['mean'], image_std=cfg['std'])
    if cfg.get('interpolation'): kwargs['image_interpolation'] = cfg['interpolation']
    if cfg.get('resize_mode'): kwargs['image_resize_mode'] = cfg['resize_mode']
    model, _, preprocess = open_clip.create_model_and_transforms('MobileCLIP2-S0', pretrained=str(checkpoint), **kwargs)
    model = model.cuda().eval()  # No reparameterization dependency; identical trained checkpoint.
    tokenizer = open_clip.get_tokenizer('MobileCLIP2-S0')
    norm = lambda x: torch.nn.functional.normalize(x.float(), dim=-1)
    with torch.inference_mode():
        positive, negative, oldpos, oldneg = [], [], [], []
        for k in LABELS:
            positive.append(norm(model.encode_text(tokenizer([p[0] for p in PAIRS[k]]).cuda())))
            negative.append(norm(model.encode_text(tokenizer([p[1] for p in PAIRS[k]]).cuda())))
            for prompts, dst in zip(LEGACY_PROMPTS[k], (oldpos,oldneg)):
                dst.append(norm(norm(model.encode_text(tokenizer(prompts).cuda())).mean(0,keepdim=True))[0])
        positive, negative = torch.stack(positive), torch.stack(negative)
        oldpos, oldneg = torch.stack(oldpos), torch.stack(oldneg)
        scale = float(model.logit_scale.exp().clamp(max=100).item())
    t0 = time.monotonic()
    for start in range(0, len(work), a.batch_size):
        rr = work[start:start+a.batch_size]
        images = []
        for item in rr:
            if sha(item['image']) != item['image_sha256']: raise ValueError('Image changed since prepare')
            with Image.open(item['image']) as im:
                # Match existing quick-infer orientation (no implicit EXIF transpose).
                images.append(preprocess(im.convert('RGB')))
        x = torch.stack(images).cuda()
        if tuple(x.shape[1:]) != (3,256,256): raise ValueError('Unexpected MobileCLIP input shape')
        with torch.inference_mode():
            feat = norm(model.encode_image(x))
            margins = torch.einsum('bd,kpd->bkp', feat, positive-negative)
            legacy_scores = torch.sigmoid(scale*(feat@oldpos.T-feat@oldneg.T))
            scores = torch.sigmoid(scale*margins.mean(-1))
        if not torch.isfinite(margins).all(): raise ValueError('Nonfinite MobileCLIP output')
        for i, item in enumerate(rr):
            mm = margins[i].cpu().tolist()
            labels = {k:mobile_vote(mm[j], a.margin) for j,k in enumerate(LABELS)}
            store.put(item, {'labels':labels, 'cosine_margins':dict(zip(LABELS,mm)),
                'display_scores':dict(zip(LABELS,scores[i].cpu().tolist())),
                'legacy_scores':dict(zip(LABELS,legacy_scores[i].cpu().tolist())),
                'legacy_labels':{k:int(legacy_scores[i,j]>=0.5) for j,k in enumerate(LABELS)},
                'logit_scale':scale, 'status':'OK'})
        if start % 100 == 0:
            print('MOBILE', start+len(rr), '/', len(work), 'seconds', round(time.monotonic()-t0), flush=True)
    store.close()
    dump(folder/'progress.json', {'phase':'preview' if a.limit else 'full', 'processed_this_invocation':len(work), 'seconds':time.monotonic()-t0})
    load_run(root)  # Verify source manifests remain unchanged.


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--margin',type=float,default=.02)
    p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    if a.batch_size<1 or not 0<a.margin<1 or a.limit<0: p.error('Invalid batch/margin/limit')
    run(a)
