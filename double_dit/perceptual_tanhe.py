"""Exact pretrained LPIPS/VGG trunk, exported without unused classifier tensors."""
import hashlib
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
EXPORT_SHA256='fac6c870cd464e242a7013124b4f02ea5e1741f583c4bf46c6c0ef6087b3cb91'
VGG_SHA256='397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0'


def load_perceptual(device):
    import lpips
    path=ROOT/'runtime/tanhe/lpips_vgg_state.pt'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=EXPORT_SHA256:raise RuntimeError('Perceptual export hash mismatch')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved['vgg_sha256']!=VGG_SHA256:raise RuntimeError('VGG source hash mismatch')
    linear_path=Path(lpips.__file__).parent/'weights/v0.1/vgg.pth'
    if hashlib.sha256(linear_path.read_bytes()).hexdigest()!=saved['lpips_linear_sha256']:
        raise RuntimeError('LPIPS linear checkpoint differs from original runtime')
    # Skip the network download; replace every VGG feature parameter strictly.
    model=lpips.LPIPS(net='vgg',pnet_rand=True)
    weights={name:saved['features']['features.'+name.split('.',1)[1]] for name in model.net.state_dict()}
    if len(weights)!=len(saved['features']):raise RuntimeError('Unexpected VGG feature architecture')
    model.net.load_state_dict(weights,strict=True)
    state=model.state_dict()
    for key,value in saved['linears'].items():
        if key not in state or not torch.equal(state[key],value):raise RuntimeError('LPIPS linear parameter mismatch: '+key)
    for key,value in model.net.state_dict().items():
        if not torch.equal(value,weights[key]):raise RuntimeError('VGG feature load mismatch: '+key)
    return model.to(device).eval().requires_grad_(False)
