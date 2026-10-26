import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch


def load_frame(frame,hw,device):
    src=[];target=[];ks=[];poses=[];ids=[]
    for v in frame['views']:
        imgs=[]
        for key in ('source','target'):
            with Image.open(v[key]) as im:
                a=np.asarray(im.convert('RGB').resize((hw[1],hw[0]),Image.Resampling.BILINEAR)).copy()
            imgs.append(torch.from_numpy(a).permute(2,0,1).float()/127.5-1)
        src.append(imgs[0]);target.append(imgs[1])
        k=np.asarray(v['K_native']).copy()
        k[0]*=hw[1]/v['native_hw'][1];k[1]*=hw[0]/v['native_hw'][0]
        ks.append(k);poses.append(v['T_reference_from_cam']);ids.append(v['camera_id'])
    return (torch.stack(src).to(device,dtype=torch.bfloat16),torch.stack(target).to(device),
            torch.tensor(np.array(ks),device=device,dtype=torch.float32),
            torch.tensor(poses,device=device,dtype=torch.float32),torch.tensor(ids,device=device))


def pack_history(current,history):
    if not history: return current[:,:,None]
    refs=list(reversed(history[-4:]))
    while len(refs)<4: refs.append(refs[-1])
    return torch.stack([current,*refs],dim=2)


def quantized_history(pred):
    # Match byte conversion used for emitted PNGs; never quantize loss inputs.
    return ((pred.detach().float().clamp(-1,1)+1)*127.5).to(torch.uint8).float().div(127.5).sub(1).to(pred.dtype)


def tensor_digest(t):
    return hashlib.sha256(t.detach().float().cpu().numpy().tobytes()).hexdigest()


def load_sequences(path,split):
    values=[json.loads(l) for l in Path(path).read_text().splitlines() if l]
    return [s for s in values if s['split']==split]
