"""CPU tests: python -m unittest discover -s tests -p 'test_injection.py' -v."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import torch
from torch import nn
import torch.nn.functional as F
from double_dit.injection import (joint_attention, joint_block, modulated_block,
    SpatialModulation, GeometryRead, geometry_correspondences, validate_exchange)
from double_dit.injection_data import load_depth

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


class ToyAttention(nn.Module):
    # Same public tensor contract as inspected Cosmos Attention, no external deps.
    def __init__(self,d=8):
        super().__init__()
        self.n_heads=2; self.head_dim=d//2
        self.q=nn.Linear(d,d); self.k=nn.Linear(d,d); self.v=nn.Linear(d,d)
        self.output_proj=nn.Linear(d,d); self.output_dropout=nn.Identity()
    def compute_qkv(self,x,context=None,rope_emb=None):
        c=x if context is None else context
        return tuple(layer(a).reshape(a.shape[0],a.shape[1],2,-1) for layer,a in ((self.q,x),(self.k,c),(self.v,c)))
    def compute_attention(self,q,k,v,video_size=None):
        r=F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)).transpose(1,2).flatten(-2)
        return self.output_dropout(self.output_proj(r))
    def forward(self,x,context=None,rope_emb=None,video_size=None):
        return self.compute_attention(*self.compute_qkv(x,context,rope_emb))


class ToyBlock(nn.Module):
    def __init__(self,d=8,lora=False):
        super().__init__(); self.use_adaln_lora=lora
        self.self_attn=ToyAttention(d); self.cross_attn=ToyAttention(d)
        self.mlp=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d))
        for suffix in ('self_attn','cross_attn','mlp'):
            setattr(self,'layer_norm_'+suffix,nn.LayerNorm(d,elementwise_affine=False))
            setattr(self,'adaln_modulation_'+suffix,nn.Sequential(nn.SiLU(),nn.Linear(d,3*d)))
    def forward(self,x,emb,context,rope_emb_L_1_1_D=None,adaln_lora_B_T_3D=None,extra_per_block_pos_emb=None):
        for suffix in ('self_attn','cross_attn','mlp'):
            p=getattr(self,'adaln_modulation_'+suffix)(emb)
            if self.use_adaln_lora:p=p+adaln_lora_B_T_3D
            shift,scale,gate=[a[:,:,None,None,:] for a in p.chunk(3,-1)]
            y=getattr(self,'layer_norm_'+suffix)(x)*(1+scale)+shift
            if suffix=='mlp':r=self.mlp(y)
            else:r=getattr(self,suffix)(y.flatten(1,3),context if suffix=='cross_attn' else None,rope_emb=rope_emb_L_1_1_D).reshape_as(x)
            x=x+gate*r
        return x


def state(v=2,d=8,lora=False):
    return (torch.randn(v,1,d),torch.randn(v,3,d),None,torch.randn(v,1,3*d) if lora else None)


class InjectionTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(7)

    def test_config_contracts(self):
        for path in (ROOT/'configs').glob('injection_*.json'):
            cfg=json.loads(path.read_text());self.assertEqual(validate_exchange(cfg),cfg['exchange']['mode'])
        cfg=json.loads((ROOT/'configs/injection_one_way.json').read_text())
        cfg['exchange']['main_to_condition_after_blocks']=[3]
        with self.assertRaises(ValueError):validate_exchange(cfg)
        cfg['exchange']['main_to_condition_after_blocks']=[];cfg['exchange']['heads']=0
        with self.assertRaises(ValueError):validate_exchange(cfg)

    def test_joint_matches_dense_masked_attention(self):
        qs,ks,vs=[torch.randn(2,8,2,4) for _ in range(3)]
        qm,km,vm=[torch.randn(2,3,2,4) for _ in range(3)]
        a,b=joint_attention(qs,ks,vs,qm,km,vm,4,3)
        mask=torch.zeros(11,11,dtype=torch.bool)
        mask[:8,:8]=True;mask[:4,8:]=True;mask[8:,:4]=True;mask[8:,8:]=True
        q,k,v=[torch.cat(t,1).transpose(1,2) for t in ((qs,qm),(ks,km),(vs,vm))]
        expected=F.scaled_dot_product_attention(q,k,v,attn_mask=mask).transpose(1,2)
        torch.testing.assert_close(torch.cat((a,b),1),expected)
        changed,_=joint_attention(qs,ks,vs,qm,km,vm+1,4,3)
        torch.testing.assert_close(changed[:,4:],a[:,4:])
        self.assertGreater(float((changed[:,:4]-a[:,:4]).abs().max()),0)
        _,changed_m=joint_attention(qs,ks,vs+1,qm,km,vm,4,3)
        self.assertGreater(float((changed_m-b).abs().max()),0)

    def test_joint_block_gradients_and_checkpoint(self):
        from torch.utils.checkpoint import checkpoint
        sb,mb=ToyBlock(lora=True),ToyBlock(lora=True)
        S=torch.randn(2,3,4,4,8,requires_grad=True);M=torch.randn(2,1,2,2,8,requires_grad=True)
        ss,ms=state(lora=True),state(lora=True)
        direct=joint_block(sb,mb,S,M,ss,ms,7)
        loss=sum(x.square().mean() for x in direct)
        grads=torch.autograd.grad(loss,(S,M,*sb.parameters(),*mb.parameters()))
        recomputed=checkpoint(lambda s,m:joint_block(sb,mb,s,m,ss,ms,7),S,M,use_reentrant=False)
        other=torch.autograd.grad(sum(x.square().mean() for x in recomputed),(S,M,*sb.parameters(),*mb.parameters()))
        for a,b in zip(grads,other):
            self.assertTrue(torch.isfinite(a).all());torch.testing.assert_close(a,b)
        self.assertGreater(float(grads[1].abs().max()),0)
        # No cross-camera leakage in the joint operation itself.
        altered=M.detach().clone();altered[1]+=10
        a,_=joint_block(sb,mb,S,altered,ss,ms,7)
        torch.testing.assert_close(a[0],direct[0][0])

    def test_modulation_identity_and_spatial_response(self):
        for lora in (False,True):
            block=ToyBlock(lora=lora);mod=SpatialModulation(8,4)
            S=torch.randn(2,3,4,4,8);M=torch.randn(2,1,2,2,8,requires_grad=True);st=state(lora=lora)
            actual=modulated_block(block,S,st,mod,M)
            native=block(S,st[0],st[1],adaln_lora_B_T_3D=st[3])
            torch.testing.assert_close(actual,native,rtol=0,atol=0)
            actual.square().mean().backward()
            self.assertGreater(float(mod.net[-1].weight.grad.abs().max()),0)
            with torch.no_grad():mod.net[-1].weight.normal_(0,.1)
            normalized=torch.randn_like(S)
            adjusted=mod(normalized,M)
            torch.testing.assert_close(adjusted[:,1:],normalized[:,1:],rtol=0,atol=0)
            adjusted.sum().backward();self.assertGreater(float(M.grad.abs().max()),0)
            self.assertGreater(float((adjusted[:,:1]-normalized[:,:1]).detach().abs().max()),0)

    def geometry(self,views=2,h=4,w=4):
        K=torch.eye(3)[None].repeat(views,1,1)
        P=torch.eye(4)[None].repeat(views,1,1)
        depth=torch.full((views,h,w),2.)
        adjacency=~torch.eye(views,dtype=torch.bool)
        return K,P,depth,adjacency

    def test_geometry_identity_and_occlusion(self):
        K,P,d,a=self.geometry()
        grid,valid=geometry_correspondences(K,P,d,(4,4),(4,4),a,0,0)
        self.assertTrue(valid[0,1].all());self.assertFalse(valid[0,0].any())
        expected=(torch.arange(4)+.5)*2/4-1
        torch.testing.assert_close(grid[0,1,0,:,0],expected)
        d[1]=1
        _,valid=geometry_correspondences(K,P,d,(4,4),(4,4),a,0,0)
        self.assertFalse(valid.any())
        d[:]=float('nan')
        grid,valid=geometry_correspondences(K,P,d,(4,4),(4,4),a)
        self.assertTrue(torch.isfinite(grid).all());self.assertFalse(valid.any())
        with self.assertRaises(ValueError):geometry_correspondences(K,P,None,(4,4),(4,4),a)

    def test_geometry_translation_and_behind_camera(self):
        K,P,d,a=self.geometry()
        P[1,0,3]=2  # z=2 => projected source x = destination x - 1 pixel
        g,v=geometry_correspondences(K,P,d,(4,4),(4,4),a,0,0)
        self.assertFalse(v[0,1,:,0].any());self.assertTrue(v[0,1,:,1:].all())
        torch.testing.assert_close(g[0,1,0,1:,0],(torch.arange(3)+.5)*2/4-1)
        P[1,2,3]=10
        _,v=geometry_correspondences(K,P,d,(4,4),(4,4),a)
        self.assertFalse(v[0,1].any())

    def test_geometry_read_mask_gradients_permutation(self):
        K,P,d,a=self.geometry(3)
        grid,valid=geometry_correspondences(K,P,d,(4,4),(4,4),a)
        S=torch.randn(3,3,4,4,8,requires_grad=True);M=torch.randn(3,1,2,2,8,requires_grad=True)
        read=GeometryRead(8,4,2,3)
        torch.testing.assert_close(read(S,M,grid,valid),S,rtol=0,atol=0)
        with torch.no_grad():read.gate.fill_(.2);read.out.bias.fill_(1)
        out=read(S,M,grid,valid)
        torch.testing.assert_close(out[:,1:],S[:,1:],rtol=0,atol=0)
        # Masked/occluded source features must not influence target.
        mask=valid.clone();mask[:,2]=False
        altered=M.detach().clone();altered[2]+=100
        torch.testing.assert_close(read(S,M,grid,mask)[0],read(S,altered,grid,mask)[0])
        empty=read(S,M,grid,torch.zeros_like(valid))
        torch.testing.assert_close(empty,S,rtol=0,atol=0)
        out.square().mean().backward()
        self.assertTrue(torch.isfinite(M.grad).all());self.assertGreater(float(M.grad.abs().max()),0)
        perm=torch.tensor([2,0,1])
        g,v=geometry_correspondences(K[perm],P[perm],d[perm],(4,4),(4,4),a[perm][:,perm])
        torch.testing.assert_close(read(S[perm],M[perm],g,v),out[perm])

    def test_depth_contract(self):
        import numpy as np
        import hashlib
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);src=p/'src.png';depth=p/'depth.npy'
            Image.new('RGB',(4,4)).save(src);np.save(depth,np.ones((4,4),dtype='float32'))
            sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
            info={'kind':'camera_z_m','source':'current_composite','available_at_inference':True,
                'source_image_sha256':sha(src),'path':str(depth),'sha256':sha(depth),'image_hw':[4,4]}
            frame={'views':[{'source':str(src),'source_sha256':sha(src),'geometry_depth':info}]}
            result=load_depth(frame,[2,2],'cpu');self.assertEqual(tuple(result.shape),(1,2,2))
            info['source']='target'
            with self.assertRaises(ValueError):load_depth(frame,[2,2],'cpu')




class FullForwardTests(unittest.TestCase):
    """Run production forward methods with tiny backbone/VAE stand-ins.

    This checks branch wiring and checkpoint recomputation, not pretrained
    Cosmos compatibility or image quality. Original model.py is AST-loaded
    without importing the external runtime or running its heavyweight init.
    """
    @staticmethod
    def build(mode):
        import ast, importlib.util, sys, types
        from torch.utils.checkpoint import checkpoint
        tree=ast.parse((ROOT/'double_dit/model.py').read_text())
        selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('pool_grid','GatedAttention','DoubleDiT')]
        module=types.ModuleType('double_dit.model')
        def adjacency(K,P,**kwargs):
            v=K.shape[1];a=~torch.eye(v,device=K.device,dtype=torch.bool)[None]
            return a,a.float()
        module.__dict__.update(torch=torch,nn=nn,F=F,checkpoint=checkpoint,
            build_pose_overlap_adjacency=adjacency,PoseRoutingConfig=lambda:None)
        exec(compile(ast.Module(body=selected,type_ignores=[]),'baseline_ast','exec'),module.__dict__)
        old=sys.modules.get('double_dit.model');sys.modules['double_dit.model']=module
        try:
            spec=importlib.util.spec_from_file_location('double_dit._test_model_injection',ROOT/'double_dit/model_injection.py')
            variant=importlib.util.module_from_spec(spec);spec.loader.exec_module(variant)
        finally:
            if old is None:sys.modules.pop('double_dit.model',None)
            else:sys.modules['double_dit.model']=old
        class Final(nn.Module):
            def __init__(self):super().__init__();self.proj=nn.Linear(8,3)
            def forward(self,x,emb,adaln_lora_B_T_3D=None):return self.proj(x)
        class DiT(nn.Module):
            def __init__(self):
                super().__init__();self.embedder=nn.Linear(3,8)
                self.blocks=nn.ModuleList([ToyBlock() for _ in range(20)])
                self.final_layer=Final()
            def unpatchify(self,x):return x.permute(0,4,1,2,3)
        model=variant.DoubleDiT.__new__(variant.DoubleDiT);nn.Module.__init__(model)
        model.cfg=json.loads((ROOT/f'configs/injection_{mode}.json').read_text())
        model.mode=mode;model.chunk_size=7;model.exchange_enabled=True
        model.positions=(3,7,11,15,19)
        model.reverse_positions=tuple(model.cfg['exchange']['main_to_condition_after_blocks'])
        model.pipeline=nn.Module();model.pipeline.dit=DiT();model.condition=DiT()
        model.pipeline.sigma_data=1.
        model.pipeline.scaling=lambda sigma:(0.,1.,1.,sigma)
        model.camera_embed=nn.Embedding(6,8);model.ray_embed=nn.Linear(6,8)
        model.read_m=nn.ModuleDict({str(i):module.GatedAttention(8,4,2) for i in model.positions}) if mode in ('one_way','bidirectional') else nn.ModuleDict()
        model.read_s=nn.ModuleDict({str(i):module.GatedAttention(8,4,2) for i in model.reverse_positions}) if mode=='bidirectional' else nn.ModuleDict()
        model.view_exchange=nn.ModuleDict({str(i):module.GatedAttention(8,4,2) for i in model.positions}) if mode!='geometry' else nn.ModuleDict()
        if mode=='modulation':model.modulators=nn.ModuleDict({str(i):SpatialModulation(8,4) for i in model.positions})
        if mode=='geometry':model.geometry_read=nn.ModuleDict({str(i):GeometryRead(8,4,2,7) for i in model.positions})
        model.encode=lambda x:x;model.decode=lambda z:z
        model.make_condition=lambda *args:{}
        def embed(dit,z,noise,fields,pool=False):
            x=dit.embedder(z.permute(0,2,3,4,1))
            if pool:x=module.pool_grid(x)
            emb=torch.ones(x.shape[0],1,8,dtype=x.dtype)
            context=torch.zeros(x.shape[0],2,8,dtype=x.dtype)
            return x,(emb,context,None,None)
        model.embed=embed
        return model.to(dtype=torch.bfloat16)

    def test_real_forward_modes_t1_t5_and_checkpoint(self):
        for mode in ('bidirectional','one_way','joint','modulation','geometry'):
            with self.subTest(mode=mode):
                torch.manual_seed(12);model=self.build(mode)
                if mode=='one_way':self.assertEqual(len(model.read_s),0)
                current=torch.randn(2,3,4,4,dtype=torch.bfloat16)
                K=torch.eye(3).repeat(2,1,1);P=torch.eye(4).repeat(2,1,1);ids=torch.tensor([0,1])
                kwargs={'depth':torch.full((2,4,4),2.)} if mode=='geometry' else {}
                for t in (1,5):
                    packed=torch.cat((current[:,:,None],torch.randn(2,3,t-1,4,4,dtype=current.dtype)),2)
                    model.training=False
                    out=model(current,packed,K,P,ids,**kwargs)
                    self.assertEqual(tuple(out.shape),tuple(current.shape));self.assertTrue(torch.isfinite(out).all())
                    # Compare the complete checkpointed path to noncheckpointed.
                    model.zero_grad(set_to_none=True)
                    out.float().square().mean().backward()
                    grads={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
                    model.zero_grad(set_to_none=True);model.training=True
                    checked=model(current,packed,K,P,ids,**kwargs)
                    torch.testing.assert_close(checked,out,rtol=0,atol=0)
                    checked.float().square().mean().backward()
                    for n,p in model.named_parameters():
                        if n in grads:torch.testing.assert_close(p.grad,grads[n],rtol=0,atol=0)
                    if mode!='joint':
                        model.exchange_enabled=False
                        model.training=False
                        disabled=model(current,packed,K,P,ids,**kwargs)
                        torch.testing.assert_close(disabled,out,rtol=0,atol=0)
                        model.exchange_enabled=True


if __name__=='__main__':unittest.main()
