"""Opt-in alternatives. Original model.py and its checkpoint keys stay intact."""
import copy
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .model import DoubleDiT as BaselineDoubleDiT, GatedAttention, pool_grid
from .model import build_pose_overlap_adjacency, PoseRoutingConfig
from .injection import (validate_exchange, joint_block, modulated_block,
                        SpatialModulation, GeometryRead, geometry_correspondences)


class DoubleDiT(BaselineDoubleDiT):
    def __init__(self, cfg, output_dir, device='cuda'):
        mode = validate_exchange(cfg)
        super().__init__(copy.deepcopy(cfg), output_dir, device)
        self.mode = mode
        self.chunk_size = cfg['exchange'].get('query_chunk_size', 128)
        width = cfg['backbone']['width']
        dim, heads = cfg['exchange']['projection_dim'], cfg['exchange']['heads']
        if mode in ('joint', 'modulation', 'geometry'):
            self.read_m = nn.ModuleDict()
            self.read_s = nn.ModuleDict()
        if mode == 'modulation':
            self.modulators = nn.ModuleDict({str(i): SpatialModulation(width, dim) for i in self.positions})
        if mode == 'geometry':
            self.view_exchange = nn.ModuleDict()
            self.geometry_read = nn.ModuleDict({str(i): GeometryRead(width, dim, heads, self.chunk_size) for i in self.positions})
        if mode in ('joint', 'modulation'):
            for stream in (self.main, self.condition):
                for i in self.positions:
                    block = stream.blocks[i]
                    for attr in ('self_attn', 'cross_attn', 'mlp', 'use_adaln_lora',
                                 'layer_norm_self_attn', 'adaln_modulation_self_attn',
                                 'layer_norm_cross_attn', 'adaln_modulation_cross_attn',
                                 'layer_norm_mlp', 'adaln_modulation_mlp'):
                        if not hasattr(block, attr):
                            raise RuntimeError(f'Unsupported Cosmos block: missing {attr}')
                    if getattr(block, 'cp_size', None) not in (None, 1):
                        raise ValueError('Injection adapter does not support context parallelism')
                    if getattr(block.self_attn, 'backend', 'torch') == 'natten':
                        raise ValueError('Injection adapter requires global native self-attention')
        self.to(device=device, dtype=torch.bfloat16)

    def _views(self, M, adjacency, i):
        v, _, h, w, d = M.shape
        m = M.flatten(1, 3)
        updated=[]
        for dst in range(v):
            src=adjacency[0,dst].nonzero().flatten()
            context=m[src].reshape(1,-1,d)
            updated.append(m[dst:dst+1]+self.view_exchange[str(i)](m[dst:dst+1],context))
        return torch.cat(updated).reshape(v,1,h,w,d)

    def _custom_block(self, fn, *args):
        if self.training and torch.is_grad_enabled():
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(self, current, packed, K, poses, camera_ids, official_only=False, depth=None):
        if official_only or self.mode in ('bidirectional', 'one_way'):
            return super().forward(current, packed, K, poses, camera_ids, official_only=official_only)
        if current.ndim!=4 or packed.ndim!=5 or packed.shape[2] not in (1,5):
            raise ValueError('Require current [V,3,H,W] and temporal packed T1 or T5')
        v,_,H,W=current.shape
        if v<1 or len(set(camera_ids.tolist()))!=v:
            raise ValueError('Invalid view identities')
        if not torch.equal(packed[:,:,0],current):
            raise ValueError('Current slot mismatch')
        if self.mode == 'geometry' and self.exchange_enabled and depth is None:
            raise ValueError('geometry requires current-composite camera-z depth; no overlap-only fallback')
        valid=torch.ones(1,v,device=current.device,dtype=torch.bool)
        adjacency,_=build_pose_overlap_adjacency(K[None],poses[None],image_hw=(H,W),view_valid=valid,config=PoseRoutingConfig())
        self.last_edges=adjacency.detach().cpu().tolist()
        z=torch.cat([self.encode(packed[j:j+1]) for j in range(v)])
        sigma=torch.full((v,1,1,1,1),self.cfg['backbone']['training_timestep']/1000.,device=z.device)
        skip,out,cin,noise=self.pipeline.scaling(sigma=sigma)
        noise=noise.squeeze((1,3,4)).to(torch.bfloat16)
        fields=self.make_condition(v,H,W,z.device)
        S,s_state=self.embed(self.main,(cin*z).to(torch.bfloat16),noise,fields)
        zc=z if packed.shape[2]==1 else torch.cat([self.encode(current[j:j+1,:,None]) for j in range(v)])
        M,m_state=self.embed(self.condition,(cin*zc).to(torch.bfloat16),noise,fields,pool=True)
        mh,mw=M.shape[2:4]
        geom=self.ray_embed(self.rays(K,poses,mh,mw,(H,W))).reshape(v,1,mh,mw,-1)
        M=M+geom+self.camera_embed(camera_ids)[:,None,None,None,:]
        correspondence=None
        if self.mode=='geometry' and self.exchange_enabled:
            options=self.cfg['exchange'].get('geometry_read',{})
            correspondence=geometry_correspondences(K,poses,depth,S.shape[2:4],(H,W),adjacency[0],
                options.get('absolute_depth_tolerance_m',.5), options.get('relative_depth_tolerance',.05))
        for i in range(self.cfg['backbone']['num_blocks']):
            active=i in self.positions and self.exchange_enabled
            if active and self.mode=='joint':
                # Route M first, then replace BOTH native self-attentions at i.
                M=self._views(M,adjacency,i)
                # Bind i as a default: checkpoint recomputes after the loop ends.
                def call(s,m,ss,ms,index=i):
                    return joint_block(self.main.blocks[index],self.condition.blocks[index],s,m,ss,ms,self.chunk_size)
                S,M=self._custom_block(call,S,M,s_state,m_state)
            elif active and self.mode=='modulation':
                M=self.block(self.condition.blocks[i],M,m_state)
                M=self._views(M,adjacency,i)
                def call(s,m,state,index=i):
                    return modulated_block(self.main.blocks[index],s,state,self.modulators[str(index)],m)
                S=self._custom_block(call,S,M,s_state)
            else:
                S=self.block(self.main.blocks[i],S,s_state)
                M=self.block(self.condition.blocks[i],M,m_state)
                if active:  # geometry: direct reads from aligned OTHER-camera M
                    # Deliberately no unaligned view_exchange here: otherwise an
                    # occluded camera could bypass the visibility mask through M.
                    S=self.geometry_read[str(i)](S,M,*correspondence)
        emb,_,_,adaln=s_state
        net=self.main.unpatchify(self.main.final_layer(S,emb,adaln_lora_B_T_3D=adaln)).float()
        zhat=skip*z+out*net
        return torch.cat([self.decode(zhat[j:j+1].to(torch.bfloat16))[:,:,:1] for j in range(v)])[:,:,0]
