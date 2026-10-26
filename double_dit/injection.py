"""Dependency-light information injection and explicit Cosmos block adapters.

No hooks or mutable forward caches: activation checkpoint recomputation is safe.
Cameras remain batch items; joint attention never concatenates all cameras.
"""
import torch
from torch import nn
import torch.nn.functional as F

MODES = ('bidirectional', 'one_way', 'joint', 'modulation', 'geometry')


def validate_exchange(cfg):
    e = cfg['exchange']
    mode = e.get('mode', 'bidirectional')
    if mode not in MODES:
        raise ValueError(f'Unknown exchange mode: {mode}')
    positions = e['condition_to_main_after_blocks']
    n = cfg['backbone']['num_blocks']
    if len(set(positions)) != len(positions) or any(type(i) is not int or not 0 <= i < n for i in positions):
        raise ValueError('Exchange positions must be unique valid block indices')
    expected_views = [] if mode == 'geometry' else positions
    if e['condition_cross_view_after_blocks'] != expected_views:
        raise ValueError('View exchange positions do not match this mode')
    if not set(e['main_to_condition_after_blocks']).issubset(positions):
        raise ValueError('Reverse positions must be a subset of exchange positions')
    if mode in ('one_way', 'modulation', 'geometry') and e['main_to_condition_after_blocks']:
        raise ValueError(f'{mode} must have no S -> M positions')
    if e['heads'] <= 0 or e['projection_dim'] <= 0 or e['projection_dim'] % e['heads']:
        raise ValueError('Projection width must be divisible by heads')
    if cfg['backbone']['width'] != 1280 or n != 20 or e['projection_dim'] != 320 or e['heads'] != 8:
        raise ValueError('Inherited Cosmos baseline requires width=1280, blocks=20, projection_dim=320, heads=8')
    if e.get('query_chunk_size', 128) < 1:
        raise ValueError('query_chunk_size must be positive')
    if cfg['condition_stream']['spatial_pool_factor_after_patch_embed'] != 2:
        raise ValueError('The inherited backbone embedding supports pool factor 2 only')
    return mode


def _norm(block, x, state, suffix):
    emb, _, _, adaln = state
    params = getattr(block, 'adaln_modulation_' + suffix)(emb)
    if block.use_adaln_lora:
        if adaln is None:
            raise ValueError('AdaLN-LoRA conditioning missing')
        params = params + adaln
    shift, scale, gate = [v[:, :, None, None, :] for v in params.chunk(3, dim=-1)]
    y = getattr(block, 'layer_norm_' + suffix)(x) * (1 + scale) + shift
    return y, gate


def _finish(block, x, state):
    y, gate = _norm(block, x, state, 'cross_attn')
    result = block.cross_attn(y.flatten(1, 3), state[1], rope_emb=state[2])
    x = x + gate * result.reshape_as(x)
    y, gate = _norm(block, x, state, 'mlp')
    return x + gate * block.mlp(y)


def _project(attn, x):
    # SDPA returns [B,L,heads,head_dim]; Cosmos output_proj takes [B,L,D].
    return attn.output_dropout(attn.output_proj(x.flatten(-2)))


def joint_attention(qs, ks, vs, qm, km, vm, current_tokens, chunk_size=128):
    """One concatenated S/M softmax, separate stream QKV, same-camera scope.

    Historical S queries keep S-only keys; current S and all M queries see both
    streams, with M reading only current S. Query chunking bounds mask memory.
    All inputs are [views, tokens, heads, head_dim].
    """
    ns, nm = qs.shape[1], qm.shape[1]
    if not 0 < current_tokens <= ns or chunk_size < 1:
        raise ValueError('Invalid current token count or chunk size')
    q, k, v = [torch.cat(pair, dim=1).transpose(1, 2) for pair in ((qs, qm), (ks, km), (vs, vm))]
    key = torch.arange(ns + nm, device=q.device)[None, :]
    chunks = []
    for start in range(0, ns + nm, chunk_size):
        query = torch.arange(start, min(start + chunk_size, ns + nm), device=q.device)[:, None]
        allowed = ((query < ns) & (key < ns)) | ((query < current_tokens) & (key >= ns))
        allowed = allowed | ((query >= ns) & ((key < current_tokens) | (key >= ns)))
        chunks.append(F.scaled_dot_product_attention(q[:, :, start:start + chunk_size], k, v, attn_mask=allowed))
    result = torch.cat(chunks, dim=2).transpose(1, 2)
    return result[:, :ns], result[:, ns:]


def joint_block(sblock, mblock, S, M, sstate, mstate, chunk_size=128):
    """Replace the two native self-attentions with a joint self-attention.

    Keep pretrained QKV/norm/output projections, AdaLN gates, text cross-attn,
    MLPs and standard block residuals. No post-block read_m/read_s adapters.
    """
    sy, sg = _norm(sblock, S, sstate, 'self_attn')
    my, mg = _norm(mblock, M, mstate, 'self_attn')
    sqkv = sblock.self_attn.compute_qkv(sy.flatten(1, 3), rope_emb=sstate[2])
    mqkv = mblock.self_attn.compute_qkv(my.flatten(1, 3), rope_emb=mstate[2])
    so, mo = joint_attention(*sqkv, *mqkv, S.shape[2] * S.shape[3], chunk_size)
    S = S + sg * _project(sblock.self_attn, so).reshape_as(S)
    M = M + mg * _project(mblock.self_attn, mo).reshape_as(M)
    return _finish(sblock, S, sstate), _finish(mblock, M, mstate)


class SpatialModulation(nn.Module):
    """Per-location scale/shift from M, avoiding a global spatial bottleneck."""
    def __init__(self, width, hidden=320):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, 2 * width))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, normalized, M):
        scale, shift = self.net(M[:, 0]).chunk(2, dim=-1)
        def resize(x):
            return F.interpolate(x.permute(0, 3, 1, 2).float(), size=normalized.shape[2:4], mode='bilinear', align_corners=False).to(normalized.dtype).permute(0, 2, 3, 1)[:, None]
        current = normalized[:, :1] * (1 + resize(scale)) + resize(shift)
        return torch.cat((current, normalized[:, 1:]), dim=1)


def modulated_block(block, S, state, modulator, M):
    y, gate = _norm(block, S, state, 'self_attn')
    y = modulator(y, M)
    q, k, v = block.self_attn.compute_qkv(y.flatten(1, 3), rope_emb=state[2])
    # Use the existing global attention operation; no new attention parameters.
    result = block.self_attn.compute_attention(q, k, v)
    return _finish(block, S + gate * result.reshape_as(S), state)


def geometry_correspondences(K, poses, depth, query_hw, image_hw, adjacency,
                             abs_tol=0.5, rel_tol=0.05):
    """Project destination camera-z depth to source images and reject occlusion.

    Returns grid [dst,src,Hq,Wq,2] and valid [dst,src,Hq,Wq]. Coordinates use
    pixel-edge convention, token centers (x+.5)*W/Wq, align_corners=False.
    Depth must describe the CURRENT COMPOSITE (including inserted assets), not
    the supervision target or an unmodified scene. Invalid depth is masked.
    """
    h, w = query_hw
    H, W = image_hw
    views = K.shape[0]
    if depth is None or tuple(depth.shape) != (views, H, W):
        raise ValueError(f'geometry requires camera-z depth [V,H,W] = {(views,H,W)}')
    if tuple(poses.shape) != (views, 4, 4) or tuple(adjacency.shape) != (views, views):
        raise ValueError('Invalid poses/adjacency shape')
    if abs_tol < 0 or rel_tol < 0:
        raise ValueError('Depth consistency tolerances must be nonnegative')
    K, poses, depth = K.float(), poses.float(), depth.float()
    if not torch.isfinite(K).all() or not torch.isfinite(poses).all():
        raise ValueError('Nonfinite calibration')
    y, x = torch.meshgrid((torch.arange(h, device=K.device) + .5) * H / h,
                          (torch.arange(w, device=K.device) + .5) * W / w, indexing='ij')
    pix = torch.stack((x, y, torch.ones_like(x)), dim=-1)
    qgrid = torch.stack((2*x/W-1, 2*y/H-1), dim=-1)
    d = F.grid_sample(depth[:, None], qgrid[None].expand(views,-1,-1,-1), mode='nearest', align_corners=False)[:, 0]
    dvalid = torch.isfinite(d) & (d > 0)
    d = torch.where(dvalid, d, torch.zeros_like(d))
    points = torch.einsum('vij,hwj->vhwi', torch.linalg.inv(K), pix) * d[..., None]
    ref = torch.einsum('vij,vhwj->vhwi', poses[:, :3, :3], points) + poses[:, None, None, :3, 3]
    inverse = torch.linalg.inv(poses)
    grids, masks = [], []
    for dst in range(views):
        srcpts = torch.einsum('vij,hwj->vhwi', inverse[:, :3, :3], ref[dst]) + inverse[:, None, None, :3, 3]
        projected = torch.einsum('vij,vhwj->vhwi', K, srcpts)
        z = srcpts[..., 2]
        uv = projected[..., :2] / projected[..., 2:].clamp_min(1e-8)
        grid = torch.stack((2*uv[...,0]/W-1, 2*uv[...,1]/H-1), dim=-1)
        finite = torch.isfinite(grid).all(-1)
        safe = torch.where(finite[...,None], grid, torch.full_like(grid, 2))
        source_d = F.grid_sample(depth[:,None], safe, mode='nearest', align_corners=False)[:,0]
        valid = dvalid[dst][None] & finite & (z > 0) & (uv[...,0] >= .5) & (uv[...,0] <= W-.5) & (uv[...,1] >= .5) & (uv[...,1] <= H-.5)
        valid &= torch.isfinite(source_d) & (source_d > 0) & ((source_d-z).abs() <= abs_tol + rel_tol*z.abs())
        valid &= adjacency[dst,:,None,None].bool()
        valid[dst] = False  # read other cameras only
        grids.append(safe)
        masks.append(valid)
    return torch.stack(grids), torch.stack(masks)


class GeometryRead(nn.Module):
    def __init__(self, width, dim=320, heads=8, chunk_size=128):
        super().__init__()
        if dim % heads:
            raise ValueError('dim must divide by heads')
        self.heads, self.dim, self.chunk_size = heads, dim, chunk_size
        self.qnorm, self.knorm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.q, self.kv = nn.Linear(width, dim), nn.Linear(width, 2*dim)
        self.out = nn.Linear(dim, width)
        self.gate = nn.Parameter(torch.zeros(()))
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, S, M, grids, valid):
        views, _, h, w, width = S[:, :1].shape
        features = M[:, 0].permute(0,3,1,2).float()
        results = []
        for dst in range(views):
            # [queries, sources, width]; each query reads only projected points.
            sampled = F.grid_sample(features, grids[dst], align_corners=False).to(S.dtype)
            context = sampled.permute(2,3,0,1).reshape(h*w, views, width)
            mask = valid[dst].permute(1,2,0).reshape(h*w, views)
            query = S[dst,0].reshape(h*w,width)
            chunks=[]
            for start in range(0,h*w,self.chunk_size):
                c = context[start:start+self.chunk_size]
                good = mask[start:start+self.chunk_size]
                b = c.shape[0]
                q = self.q(self.qnorm(query[start:start+self.chunk_size])).reshape(b,1,self.heads,-1).transpose(1,2)
                k,v = self.kv(self.knorm(c)).chunk(2,-1)
                k,v = [a.reshape(b,views,self.heads,-1).transpose(1,2) for a in (k,v)]
                # A zero fallback for empty rows avoids all-masked softmax NaNs.
                empty = ~good.any(-1)
                k = torch.cat((k,torch.zeros_like(k[:,:,:1])),dim=2)
                v = torch.cat((v,torch.zeros_like(v[:,:,:1])),dim=2)
                allowed = torch.cat((good,empty[:,None]),dim=1)[:,None,None,:]
                out = F.scaled_dot_product_attention(q,k,v,attn_mask=allowed).reshape(b,self.dim)
                out = self.out(out) * (~empty)[:,None]
                chunks.append(out)
            results.append(torch.cat(chunks).reshape(1,1,h,w,width))
        current = S[:,:1] + self.gate.tanh()*torch.cat(results)
        return torch.cat((current,S[:,1:]),dim=1)
