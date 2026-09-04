#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import torch
from model import create_ultraface_slim_scene8

ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--opset',type=int,default=13); a=ap.parse_args()
ck=torch.load(a.checkpoint,map_location='cpu'); m=create_ultraface_slim_scene8().eval(); m.load_state_dict(ck['model'],strict=True)
x=torch.zeros(1,3,360,640)
a.output.parent.mkdir(parents=True,exist_ok=True)
torch.onnx.export(m,x,str(a.output),input_names=['image'],output_names=['logits'],opset_version=a.opset,dynamic_axes={'image':{0:'batch'},'logits':{0:'batch'}},do_constant_folding=True)
meta={'input_shape':[1,3,360,640],'output_shape':[1,8],'labels':ck['labels'],'thresholds':ck['thresholds'],'preprocess':'RGB; resize 640x360; (x-127)/128','architecture':'original UltraFace Mb_Tiny backbone + AdaptiveAvgPool2d + Linear(256,8)'}
a.output.with_suffix('.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
print('ONNX_EXPORTED',a.output)
