import copy
import json
import os
from pathlib import Path
import sys
from types import MethodType
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

SOURCE = Path('/mnt/hlk/workspace/Scheme_A&Scheme_B')
RECIPE = Path('/mnt/hlk/workspace/harmonizer_official_recipe_20261012')
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(RECIPE/'src'))
from multiview_harmonizer.factory import build_official_pipeline_config
from multiview_harmonizer.checkpoint import load_official_harmonizer_checkpoint, require_sha256
from multiview_harmonizer.constants import OFFICIAL_PATHS
from multiview_harmonizer.runtime_contract import require_official_text2image_dit_contract
from multiview_harmonizer.tokenizer_utils import inner_tokenizer_module
from multiview_harmonizer.b_tmv_routing import PoseRoutingConfig, build_pose_overlap_adjacency
from cosmos_predict2.pipelines.text2image import Text2ImagePipeline
from cosmos_predict2.conditioner import DataType
from cosmos_predict2.models.text2image_dit import CheckpointMode, SACConfig
from vae_memory import vae_call, set_vae_checkpointing


def pool_grid(x, factor=2):
    b,t,h,w,d=x.shape
    if h%factor or w%factor: raise ValueError('Token grid must divide by pool factor')
    return F.avg_pool2d(x.permute(0,1,4,2,3).reshape(b*t,d,h,w),factor).reshape(b,t,d,h//factor,w//factor).permute(0,1,3,4,2)


class GatedAttention(nn.Module):
    def __init__(self, width=1280, dim=320, heads=8):
        super().__init__()
        self.heads=heads;self.dim=dim
        self.qnorm=nn.LayerNorm(width);self.knorm=nn.LayerNorm(width)
        self.q=nn.Linear(width,dim);self.kv=nn.Linear(width,2*dim)
        self.out=nn.Linear(dim,width)
        self.gate=nn.Parameter(torch.zeros(()))
        nn.init.xavier_uniform_(self.out.weight);nn.init.zeros_(self.out.bias)

    def forward(self,q,context):
        if context.shape[1]==0: return torch.zeros_like(q)
        b,n,_=q.shape;m=context.shape[1];h=self.heads;d=self.dim//h
        Q=self.q(self.qnorm(q)).reshape(b,n,h,d).transpose(1,2)
        K,V=self.kv(self.knorm(context)).chunk(2,-1)
        K=K.reshape(b,m,h,d).transpose(1,2);V=V.reshape(b,m,h,d).transpose(1,2)
        attended=F.scaled_dot_product_attention(Q,K,V).transpose(1,2).reshape(b,n,self.dim)
        return self.gate.tanh()*self.out(attended)


class DoubleDiT(nn.Module):
    """One time step; cameras are batch items, never tokenizer time slots."""
    def __init__(self, cfg, output_dir, device='cuda'):
        super().__init__()
        assert os.getuid()==1005, 'Model experiments must run as hlk'
        self.cfg=cfg;self.device_name=device;self.exchange_enabled=True
        b=cfg['backbone']
        require_sha256(b['harmonizer_checkpoint'],b['expected_harmonizer_sha256'])
        require_sha256(Path(b['cosmos_checkpoint_dir'])/'model.pt',OFFICIAL_PATHS.cosmos_dit_sha256)
        require_sha256(Path(b['cosmos_checkpoint_dir'])/'tokenizer/tokenizer.pth',OFFICIAL_PATHS.tokenizer_sha256)
        pc=build_official_pipeline_config(b['cosmos_checkpoint_dir'],torch.bfloat16)
        # Explicit full block recomputation below; do not nest runtime wrappers.
        pc.net.sac_config=SACConfig(mode=CheckpointMode.NONE,every_n_blocks=1)
        self.pipeline=Text2ImagePipeline.from_config(pc,use_text_encoder=False,dit_path=pc.dit_path,device=device,torch_dtype=torch.bfloat16)
        self.pipeline.tensor_kwargs={'device':device,'dtype':torch.bfloat16}
        require_official_text2image_dit_contract(self.pipeline.dit)
        load_official_harmonizer_checkpoint(self.pipeline,b['harmonizer_checkpoint'],audit_output=Path(output_dir)/'official_load.json')
        self.vae=inner_tokenizer_module(self.pipeline.tokenizer)
        self.condition=copy.deepcopy(self.pipeline.dit)
        del self.condition.final_layer
        self.copy_verified=True
        originals=dict(self.pipeline.dit.named_parameters())
        for name,p in self.condition.named_parameters():
            assert torch.equal(p,originals[name]),name
            assert p.data_ptr()!=originals[name].data_ptr(),name
        self.positions=tuple(cfg['exchange']['condition_to_main_after_blocks'])
        self.reverse_positions=tuple(cfg['exchange']['main_to_condition_after_blocks'])
        self.read_m=nn.ModuleDict({str(i):GatedAttention() for i in self.positions})
        self.read_s=nn.ModuleDict({str(i):GatedAttention() for i in self.reverse_positions})
        self.view_exchange=nn.ModuleDict({str(i):GatedAttention() for i in self.positions})
        self.camera_embed=nn.Embedding(6,1280)
        self.ray_embed=nn.Sequential(nn.Linear(6,128),nn.SiLU(),nn.Linear(128,1280))
        self.to(device=device,dtype=torch.bfloat16)
        # Older tokenizer facades wrap VAE methods in no_grad for inference.
        facade=self.pipeline.tokenizer.model
        for name in ('encode','decode'):
            fn=getattr(facade,name)
            while hasattr(fn,'__wrapped__'):
                fn=MethodType(fn.__wrapped__,facade)
            setattr(facade,name,fn)
        self.train(True)
        self.last_edges=None

    @property
    def main(self): return self.pipeline.dit

    def train(self,mode=True):
        super().train(mode)
        self.requires_grad_(True)
        for name,p in self.vae.named_parameters():
            if 'time_conv' in name: p.requires_grad_(False)
        set_vae_checkpointing(self.vae,mode)
        return self

    def encode(self,x):
        return vae_call(self.pipeline.tokenizer.encode,self.vae,x)*self.pipeline.sigma_data

    def decode(self,z):
        return vae_call(self.pipeline.tokenizer.decode,self.vae,z/self.pipeline.sigma_data)

    def make_condition(self,views,h,w,device):
        # Keep the official unconditional fields; replicate only batch, not camera-as-time.
        batch={'dataset_name':'image_data','images':torch.zeros(views,3,h,w,device=device,dtype=torch.bfloat16),
            't5_text_embeddings':torch.zeros(views,512,1024,device=device,dtype=torch.bfloat16),
            'fps':torch.full((views,),24.,device=device),
            'padding_mask':torch.zeros(views,1,h,w,device=device,dtype=torch.bfloat16)}
        _,uncond=self.pipeline.conditioner.get_condition_uncondition(batch)
        return uncond.edit_data_type(DataType.IMAGE).to_dict()

    def embed(self,dit,z,noise,fields,pool=False):
        x,rope,extra=dit.prepare_embedded_sequence(z,fps=fields['fps'],padding_mask=fields['padding_mask'])
        if extra is not None: raise ValueError('Unexpected absolute positional embedding')
        if pool:
            _,t,h,w,_=x.shape
            # RoPE object contains unrotated phase angles. Their mean gives exact
            # cell-center coordinates before sin/cos, not averaged rotated keys.
            r=rope.reshape(1,t,h,w,-1)
            rope=pool_grid(r).reshape(-1,1,1,r.shape[-1]).contiguous()
            x=pool_grid(x)
        context=fields['crossattn_emb']
        if dit.crossattn_proj is not None: context=dit.crossattn_proj(context)
        emb,adaln=dit.t_embedder(noise)
        emb=dit.t_embedding_norm(emb)
        return x,(emb,context,rope,adaln)

    def block(self,block,x,state):
        emb,ctx,rope,adaln=state
        def call(value,e,c,r,a):
            return block(value,e,c,rope_emb_L_1_1_D=r,adaln_lora_B_T_3D=a,extra_per_block_pos_emb=None)
        if self.training and torch.is_grad_enabled():
            return checkpoint(call,x,emb,ctx,rope,adaln,use_reentrant=False)
        return call(x,emb,ctx,rope,adaln)

    def rays(self,K,poses,h,w,image_hw):
        H,W=image_hw;dev=K.device
        yy,xx=torch.meshgrid((torch.arange(h,device=dev)+.5)*H/h,
                            (torch.arange(w,device=dev)+.5)*W/w,indexing='ij')
        pixels=torch.stack([xx,yy,torch.ones_like(xx)],-1).reshape(-1,3).float()
        direction=torch.einsum('vij,pj->vpi',torch.linalg.inv(K.float()),pixels)
        direction=F.normalize(torch.einsum('vij,vpj->vpi',poses[:,:3,:3].float(),direction),dim=-1)
        origin=poses[:,:3,3].float()[:,None].expand_as(direction)
        moment=torch.cross(origin,direction,dim=-1)/10.0
        return torch.cat([direction,moment],-1).to(torch.bfloat16)

    def forward(self,current,packed,K,poses,camera_ids,official_only=False):
        # current [V,3,H,W], packed [V,3,T,H,W], same history length within a group.
        if current.ndim!=4 or packed.ndim!=5 or packed.shape[2] not in (1,5):
            raise ValueError('Require current [V,3,H,W] and temporal packed T1 or T5')
        v,_,H,W=current.shape
        if v<1 or len(set(camera_ids.tolist()))!=v: raise ValueError('Invalid view identities')
        if not torch.equal(packed[:,:,0],current): raise ValueError('Current slot mismatch')
        valid=torch.ones(1,v,device=current.device,dtype=torch.bool)
        adjacency,scores=build_pose_overlap_adjacency(K[None],poses[None],image_hw=(H,W),view_valid=valid,config=PoseRoutingConfig())
        self.last_edges=adjacency.detach().cpu().tolist()
        # VAE calls are per-camera so temporal caches never mix cameras.
        z=torch.cat([self.encode(packed[j:j+1]) for j in range(v)])
        sigma=torch.full((v,1,1,1,1),self.cfg['backbone']['training_timestep']/1000.,device=z.device)
        skip,out,cin,noise=self.pipeline.scaling(sigma=sigma)
        noise=noise.squeeze((1,3,4)).to(torch.bfloat16)
        fields=self.make_condition(v,H,W,z.device)
        if official_only:
            net=self.main((cin*z).to(torch.bfloat16),noise,**fields).float()
        else:
            S,s_state=self.embed(self.main,(cin*z).to(torch.bfloat16),noise,fields)
            if packed.shape[2]==1:
                zc=z
            else:
                zc=torch.cat([self.encode(current[j:j+1,:,None]) for j in range(v)])
            M,m_state=self.embed(self.condition,(cin*zc).to(torch.bfloat16),noise,fields,pool=True)
            mh,mw=M.shape[2:4]
            geom=self.ray_embed(self.rays(K,poses,mh,mw,(H,W))).reshape(v,1,mh,mw,-1)
            M=M+geom+self.camera_embed(camera_ids)[:,None,None,None,:]
            for i in range(20):
                S=self.block(self.main.blocks[i],S,s_state)
                M=self.block(self.condition.blocks[i],M,m_state)
                if i in self.positions and self.exchange_enabled:
                    m=M.flatten(1,3)
                    updated=[]
                    for dst in range(v):
                        src=adjacency[0,dst].nonzero().flatten()
                        context=m[src].reshape(1,-1,m.shape[-1])
                        updated.append(m[dst:dst+1]+self.view_exchange[str(i)](m[dst:dst+1],context))
                    m=torch.cat(updated)
                    sc=S[:,:1].flatten(1,3)
                    ds=self.read_m[str(i)](sc,m)
                    if i in self.reverse_positions:
                        dm=self.read_s[str(i)](m,pool_grid(S[:,:1]).flatten(1,3))
                        M=(m+dm).reshape(v,1,mh,mw,-1)
                    else:
                        M=m.reshape(v,1,mh,mw,-1)
                    cur=(sc+ds).reshape_as(S[:,:1])
                    S=torch.cat([cur,S[:,1:]],dim=1)
            emb,_,_,adaln=s_state
            net=self.main.unpatchify(self.main.final_layer(S,emb,adaln_lora_B_T_3D=adaln)).float()
        zhat=skip*z+out*net
        # Preserve temporal normalization and decode context; publish slot zero only.
        return torch.cat([self.decode(zhat[j:j+1].to(torch.bfloat16))[:,:,:1] for j in range(v)])[:,:,0]


class OfficialLoss(nn.Module):
    def __init__(self,device):
        super().__init__()
        import lpips
        self.perceptual=lpips.LPIPS(net='vgg').to(device).eval().requires_grad_(False)

    def forward(self,pred,target):
        mse=(pred.float()-target.float()).square().mean()
        perceptual=self.perceptual(pred.float(),target.float()).clamp(0,.3).mean()
        return mse+.3*perceptual,{'train_mse':mse.detach().item(),'train_perceptual_loss':perceptual.detach().item()}
