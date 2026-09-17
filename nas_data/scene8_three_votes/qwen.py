"""Blind structured Qwen3-VL annotation; one GPU/process, sharded atomic resume."""
import argparse
import importlib.metadata as md
from pathlib import Path
import time
from common import load_run, stage_lock, RecordStore, sha, dump, read, parse_qwen
from prompts import LABELS, QWEN_PROMPT
MODEL_ID = 'Qwen/Qwen3-VL-8B-Instruct'


def model_snapshot(root, a):
    lockfile = root/'qwen_model.json'
    if lockfile.exists():
        lock = read(lockfile)
        if lock['model_id'] != MODEL_ID: raise ValueError('Wrong locked teacher')
        if a.model_path and str(a.model_path.resolve()) != lock['path']: raise ValueError('Changed local teacher path')
        path = Path(lock['path'])
        for name, h in lock['files'].items():
            if sha(path/name) != h: raise ValueError('Locked Qwen model changed or incomplete')
        return path, lock
    if not a.download_only:
        raise ValueError('Run --download-only once before starting parallel Qwen workers')
    if a.insecure_downloads:
        import requests
        from huggingface_hub import configure_http_backend
        def backend():
            session=requests.Session(); session.verify=False; return session
        configure_http_backend(backend_factory=backend)
    if a.model_path:
        path = a.model_path.resolve()
        revision = 'local-manual-fingerprinted'
    else:
        from huggingface_hub import HfApi, snapshot_download
        # Resolve mutable main ONCE; downloads and all workers use this immutable commit.
        revision = HfApi().model_info(MODEL_ID).sha
        path = Path(snapshot_download(MODEL_ID, revision=revision,
                    allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja']))
    if read(path/'config.json').get('model_type') != 'qwen3_vl': raise ValueError('Not Qwen3-VL weights')
    index = read(path/'model.safetensors.index.json')
    weights = set(index['weight_map'].values())
    files = sorted(set(p.name for p in path.iterdir() if p.is_file() and p.suffix in ('.json','.txt','.model','.jinja')) | weights)
    lock = {'model_id':MODEL_ID, 'revision':revision, 'path':str(path.resolve()), 'files':{n:sha(path/n) for n in files}}
    dump(lockfile, lock)
    return path, lock


def run(a):
    root, plan, items = load_run(a.output_root)
    path, model_lock = model_snapshot(root, a)
    if a.download_only:
        print('QWEN_SNAPSHOT_LOCKED', model_lock['revision'], flush=True); return
    import torch
    from PIL import Image
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
    torch.set_num_threads(2); torch.manual_seed(20260917)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    meta={'selection':plan['items_digest'], 'model':model_lock, 'prompt':QWEN_PROMPT,
          'shards':a.shards, 'shard':a.shard, 'dtype':'float16', 'attention':'eager',
          'max_pixels':a.max_pixels, 'max_new_tokens':640, 'do_sample':False,
          'packages':{n:md.version(n) for n in ('torch','torchvision','transformers','huggingface_hub')}}
    folder, signature=stage_lock(root, 'qwen_%02d'%a.shard, meta)
    store=RecordStore(folder/'results.sqlite', signature)
    work=items[:a.limit] if a.limit else items
    work=[r for i,r in enumerate(work) if i%a.shards==a.shard and store.get(r) is None]
    if not work:
        store.close(); print('QWEN_REUSED',a.shard,flush=True); return
    processor=AutoProcessor.from_pretrained(path, local_files_only=True)
    # Qwen checkpoint uses Qwen2VLImageProcessorFast; size values are pixel AREA bounds.
    processor.image_processor.size={'shortest_edge':65536, 'longest_edge':a.max_pixels}
    model=Qwen3VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float16,
          attn_implementation='eager', local_files_only=True, use_safetensors=True,
          device_map={'':'cuda:0'}).eval()
    t0=time.monotonic(); consecutive_errors=0
    for n,item in enumerate(work):
        if sha(item['image'])!=item['image_sha256']: raise ValueError('Image changed since prepare')
        with Image.open(item['image']) as im: image=im.convert('RGB')
        # Labels, filenames, source names, and the other teacher's output never enter the prompt.
        messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':QWEN_PROMPT}]}]
        text=processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs=processor(text=[text],images=[image],return_tensors='pt').to('cuda:0')
        inputs.pop('token_type_ids', None)
        raws=[]; result=None
        for attempt in range(2):
            with torch.inference_mode():
                generated=model.generate(**inputs, do_sample=False, max_new_tokens=640 if attempt==0 else 960)
            raw=processor.batch_decode(generated[:,inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
            raws.append(raw)
            try:
                parsed=parse_qwen(raw)
                result={'labels':{k:parsed['labels'][k] for k in LABELS}, 'weather_components':{k:parsed['labels'][k] for k in ('rain','snow')},
                        'evidence':parsed['evidence'], 'raw_outputs':raws, 'status':'OK',
                        'weather_or_corrected':parsed['weather_or_corrected'], 'spatial_conflict':parsed['spatial_conflict']}
                break
            except (ValueError, TypeError):
                pass
        if result is None:
            result={'labels':{k:-1 for k in LABELS},'raw_outputs':raws,'status':'PARSE_ERROR_ABSTAIN'}
            consecutive_errors+=1
        else:
            consecutive_errors=0
        store.put(item,result)
        if n%10==0:
            elapsed=time.monotonic()-t0
            print('QWEN',a.shard,n+1,'/',len(work),'seconds',round(elapsed),'ETA_hours',round(elapsed/(n+1)*(len(work)-n-1)/3600,2),flush=True)
            dump(folder/'progress.json',{'processed_this_invocation':n+1,'remaining':len(work)-n-1,'elapsed_seconds':elapsed})
        if consecutive_errors>=5:
            raise RuntimeError('Five consecutive malformed Qwen outputs; saved raw outputs, stopping for diagnosis')
    store.close(); load_run(root)
    print('QWEN_SHARD_COMPLETE',a.shard,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--download-only',action='store_true')
    p.add_argument('--model-path',type=Path)
    p.add_argument('--insecure-downloads',action='store_true')
    p.add_argument('--shards',type=int,default=2)
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--max-pixels',type=int,default=512*512)
    a=p.parse_args()
    if not 0<=a.shard<a.shards or a.limit<0 or a.max_pixels<65536: p.error('Invalid shards/limit/pixels')
    try:
        run(a)
    except Exception as e:
        # Network failures can include proxy URLs; never print exception text/credentials.
        import traceback, re
        safe_trace = re.sub(r'https?://[^\s]+', '[network URL redacted]', traceback.format_exc())
        print('QWEN_BLOCKED', type(e).__name__, safe_trace, flush=True)
        raise SystemExit(2)
