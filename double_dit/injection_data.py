"""Explicit geometry input contract, independent of RGB supervision."""
from pathlib import Path
import hashlib
import numpy as np
import torch
import torch.nn.functional as F


def load_depth(frame, hw, device):
    values=[]
    for view in frame['views']:
        info=view.get('geometry_depth')
        if not info:
            raise ValueError('Each geometry view requires geometry_depth metadata')
        required={'kind':'camera_z_m', 'source':'current_composite', 'available_at_inference':True}
        for key,value in required.items():
            if info.get(key)!=value:
                raise ValueError(f'geometry_depth.{key} must be {value!r}')
        if info.get('source_image_sha256') != view.get('source_sha256') or not view.get('source_sha256'):
            raise ValueError('Depth must be linked to the current composite source hash')
        path=Path(info['path'])
        if path.suffix != '.npy':
            raise ValueError('Depth must be a non-pickled float .npy array in metres')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=info.get('sha256'):
            raise ValueError(f'Depth checksum mismatch: {path}')
        if hashlib.sha256(Path(view['source']).read_bytes()).hexdigest()!=view['source_sha256']:
            raise ValueError('Current composite checksum mismatch')
        a=np.load(path, allow_pickle=False)
        if a.ndim!=2 or a.dtype.kind!='f' or list(a.shape)!=info.get('image_hw'):
            raise ValueError('Depth shape/dtype differs from declared image_hw')
        # Same FoV as RGB/K; only resizing is permitted, no hidden crop.
        from PIL import Image
        with Image.open(view['source']) as image:
            if (a.shape[1],a.shape[0])!=image.size:
                raise ValueError('Depth must use the current composite pixel grid')
        depth=torch.from_numpy(a.copy()).float()[None,None]
        values.append(F.interpolate(depth,size=hw,mode='nearest-exact')[0,0])
    return torch.stack(values).to(device)


def model_inputs(frame, hw, device, cfg):
    if cfg['exchange'].get('mode')=='geometry':
        return {'depth':load_depth(frame,hw,device)}
    return {}
