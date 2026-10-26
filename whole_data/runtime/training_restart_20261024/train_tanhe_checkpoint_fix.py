"""Official objective, full main/VAE updates, generated-history rollouts."""
import argparse
import collections
import zipfile
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from double_dit.model_tanhe import OfficialLoss
from double_dit.model_injection_tanhe import DoubleDiT
from double_dit.sixcam_contract import validate_sixcam_manifest
from double_dit.injection_data import model_inputs
from double_dit.data import load_frame,load_sequences,pack_history,quantized_history,tensor_digest


def save_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def gradients(model):
    stats=collections.defaultdict(lambda:{'with_gradient':0,'nonzero':0,'finite':True})
    for n,p in model.named_parameters():
        if not p.requires_grad:continue
        if n.startswith('pipeline.dit.blocks.'):
            kind='main_block_'+n.split('.')[3]
        elif n.startswith('pipeline.dit.'):kind='main_other'
        elif n.startswith('vae.encoder'):kind='vae_encoder'
        elif n.startswith('vae.decoder'):kind='vae_decoder'
        else:kind=n.split('.')[0]
        s=stats[kind]
        if p.grad is not None:
            s['with_gradient']+=1
            s['finite'] &= bool(torch.isfinite(p.grad).all())
            s['nonzero']+=int(bool(p.grad.abs().max()>0))
    return dict(stats)


def check_gradients(model,step):
    stats=gradients(model)
    assert all(v['finite'] for v in stats.values()), 'Nonfinite gradient'
    for key in ['main_block_0','main_block_10','main_block_19','main_other','vae_encoder','vae_decoder']:
        assert stats.get(key,{}).get('nonzero',0)>0,(key,stats)
    if step>=2:
        assert stats.get('condition',{}).get('nonzero',0)>0,stats
    return stats


def smoke(model,criterion,seq,hw,out):
    device=next(model.parameters()).device
    frame=seq['frames'][0]
    src,tgt,K,P,ids=load_frame(frame,hw,device)
    inputs=model_inputs(frame,hw,device,model.cfg)
    packed=pack_history(src,[])
    model.eval()
    with torch.no_grad():
        reference=model(src,packed,K,P,ids,official_only=True)
        actual=model(src,packed,K,P,ids,**inputs)
        zero_error=float((reference.float()-actual.float()).abs().max())
        if model.mode!='joint':
            assert torch.allclose(reference.float(),actual.float(),atol=.002,rtol=.002),zero_error
        initial_history=quantized_history(actual)
        second=load_frame(seq['frames'][1],hw,device)
        s2,t2,k2,p2,i2=second
        pack2=pack_history(s2,[initial_history])
        ref2=model(s2,pack2,k2,p2,i2,official_only=True)
        inputs2=model_inputs(seq['frames'][1],hw,device,model.cfg)
        out2=model(s2,pack2,k2,p2,i2,**inputs2)
        temporal_zero_error=float((ref2.float()-out2.float()).abs().max())
        if model.mode!='joint':
            assert torch.allclose(ref2.float(),out2.float(),atol=.002,rtol=.002),temporal_zero_error
        perturbed_pack=pack_history(s2,[initial_history.flip(-1)])
        changed=model(s2,perturbed_pack,k2,p2,i2,**inputs2)
        history_effect=float((changed.float()-out2.float()).abs().max())
        assert history_effect>0,'History does not influence current prediction'
        del ref2,out2,changed,pack2,perturbed_pack
    del reference,actual
    model.train()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    history=[];journal=[];audits=[]
    for step in range(4):
        f=seq['frames'][min(step,len(seq['frames'])-1)]
        # One genuine T1->T5 rollout; subsequent optimizer diagnostics reuse
        # the current single-frame training sample, not invented future frames.
        if step>=2:history=[]
        src,tgt,K,P,ids=load_frame(f,hw,device)
        packed=pack_history(src,history)
        t0=time.monotonic()
        pred=model(src,packed,K,P,ids,**model_inputs(f,hw,device,model.cfg))
        loss,parts=criterion(pred,tgt)
        assert torch.isfinite(loss)
        loss.backward()
        stats=check_gradients(model,step)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        # Verify at least one actual early-main and VAE weight changes.
        watched={n:p for n,p in model.named_parameters() if (n.startswith('pipeline.dit.blocks.0.') or n.startswith('vae.encoder.')) and p.requires_grad}
        before={n:p.detach().clone() for n,p in watched.items()} if step==0 else {}
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        if step==0:
            changes={key:sum(not torch.equal(before[n],p) for n,p in watched.items() if n.startswith(key)) for key in ['pipeline.dit.blocks.0.','vae.encoder.']}
            assert all(changes.values()),changes
        history_digests=[tensor_digest(h) for h in history]
        history.append(quantized_history(pred))
        journal.append({'step':step,'source_frame':f['frame_index'],'T':packed.shape[2],'history':history_digests,'prediction':tensor_digest(history[-1]),'kind':'autoregressive_rollout' if step<2 else 'single_frame_gradient_diagnostic'})
        audits.append(stats)
        print('SMOKE_STEP',step,parts,'seconds',time.monotonic()-t0,flush=True)
        del pred,loss,packed,before
    assert journal[1]['history']==[journal[0]['prediction']]
    save_json(out/'history_audit.json',journal)
    save_json(out/'gradient_audit.json',audits)
    # Exercise the interaction topology at deliberately opened diagnostic gates.
    # These values are never saved as training initialization.
    model.eval()
    saved={n:p.detach().clone() for n,p in model.named_parameters() if n.endswith('.gate')}
    with torch.no_grad():
        for n,p in model.named_parameters():
            if n in saved:p.fill_(.2)
        src,tgt,K,P,ids=load_frame(frame,hw,device)
        one=pack_history(src,[])
        original=model(src,one,K,P,ids,**inputs)
        permutation=torch.arange(len(ids)-1,-1,-1,device=device)
        permuted_inputs={k:v[permutation] for k,v in inputs.items()}
        permuted=model(src[permutation],one[permutation],K[permutation],P[permutation],ids[permutation],**permuted_inputs)[permutation]
        permutation_error=float((original.float()-permuted.float()).abs().max())
        assert torch.allclose(original.float(),permuted.float(),atol=.005,rtol=.005),permutation_error
        alternate=src.clone();alternate[1]=src[1].flip(-1)
        affected=model(alternate,pack_history(alternate,[]),K,P,ids,**inputs)
        cross_view_effect=float((affected[0].float()-original[0].float()).abs().max())
        assert cross_view_effect>0,'No routed cross-view influence on target camera'
        for n,p in model.named_parameters():
            if n in saved:p.copy_(saved[n])
    result={'mode':model.mode,'initialization_equivalent_to_official':model.mode!='joint','status':'PASS','uid':os.getuid(),'world_size':1,'zero_gate_max_error':zero_error,
      'T5_zero_gate_max_error':temporal_zero_error,'generated_history_output_effect':history_effect,
      'diagnostic_gate_value':.2,'permutation_max_error':permutation_error,'cross_view_output_effect':cross_view_effect,
      'early_main_and_vae_update':changes,'generated_history_verified':True,
      'image_hw':hw,'peak_memory_bytes':torch.cuda.max_memory_allocated()}
    save_json(out/'RESULT.json',result)


def checkpoint_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        result = type(value)((key, checkpoint_to_cpu(item)) for key, item in value.items())
        if hasattr(value, '_metadata'):
            result._metadata = value._metadata
        return result
    if isinstance(value, list):
        return [checkpoint_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(checkpoint_to_cpu(item) for item in value)
    return value


def main():
    assert os.getuid()==Path('/data/juziwei/hlk/double_DiT').stat().st_uid
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--phase',required=True,choices=['probe','smoke','native','ddp_smoke','train'])
    p.add_argument('--steps',type=int);p.add_argument('--resume');args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text());out=Path(args.output)
    if args.phase in ('ddp_smoke','train'):
        # Validate every frame before creating a CUDA context or loading weights.
        # Historical two-camera smoke runs remain reproducible, but cannot be
        # mistaken for the final six-camera training path.
        contract=validate_sixcam_manifest(cfg,require_validation_split=args.phase=='train')
        if args.phase=='train' and cfg.get('training_enabled') is not True:
            raise ValueError('Formal training requires training_enabled=true')
        if int(os.environ.get('RANK',0))==0:
            print('SIXCAM_DATA_CONTRACT',json.dumps(contract),flush=True)
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    if world>1:dist.init_process_group('nccl',timeout=timedelta(minutes=30))
    checkpoint_group=dist.new_group(backend='gloo',timeout=timedelta(minutes=30)) if world>1 else None
    torch.manual_seed(42);random.seed(42);np.random.seed(42)
    if args.phase=='probe':
        import cosmos_predict2.tokenizers.tokenizer as tokenizer
        print({'uid':os.getuid(),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'tokenizer':tokenizer.__file__,
          'encode_decorated':hasattr(tokenizer.VAE.encode,'__wrapped__')},flush=True)
        return
    auditdir=out/f'rank{rank}';auditdir.mkdir(exist_ok=True)
    model=DoubleDiT(cfg,auditdir,str(device))
    params=[p for p in model.parameters() if p.requires_grad]
    assert len({id(p) for p in params})==len(params)
    save_json(auditdir/'parameter_scope.json',{'uid':os.getuid(),'rank':rank,
      'trainable_numel':sum(p.numel() for p in params),
      'trainable_names':[n for n,p in model.named_parameters() if p.requires_grad],
      'frozen_names':[n for n,p in model.named_parameters() if not p.requires_grad]})
    criterion=OfficialLoss(device)
    sequences=load_sequences(cfg['data']['manifest'],'train')
    assert sequences
    if args.phase in ('smoke','native'):
        seq=next(s for s in sequences if len(s['frames'])>=2)
        smoke(model,criterion,seq,[64,128] if args.phase=='smoke' else cfg['data']['image_hw'],out)
        return
    assert world in (4,8)
    if args.phase=='ddp_smoke':sequences=[s for s in sequences if len(s['frames'])>=2]
    wrapped=DDP(model,device_ids=[local],broadcast_buffers=False,find_unused_parameters=True,gradient_as_bucket_view=True)
    optimizer=torch.optim.AdamW(params,lr=2e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    total=args.steps or (4 if args.phase=='ddp_smoke' else cfg['training']['max_optimizer_steps'])
    accum=1 if args.phase=='ddp_smoke' else 8//world
    order=list(range(len(sequences)));random.Random(42).shuffle(order)
    start_step=0
    if args.resume:
        if args.phase!='train':raise ValueError('Resume is only supported for formal training')
        saved=torch.load(args.resume,map_location='cpu',weights_only=False)
        assert saved['config']['training_recipe_id']==cfg['training_recipe_id']
        assert saved['config']['implementation_sha256']==cfg['implementation_sha256']
        assert saved['data_sha256']==hashlib.sha256(Path(cfg['data']['manifest']).read_bytes()).hexdigest()
        assert saved['config']['training']==cfg['training']
        assert saved['config']['exchange']==cfg['exchange'], 'Cannot resume across injection modes'
        model.load_state_dict(saved['model'],strict=True)
        optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        order=saved['sequence_order'];start_step=saved['step']
        rng=saved['rng_by_rank'][rank]
        torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda'],device)
        random.setstate(rng['python']);np.random.set_state(rng['numpy'])
        del saved
    started=time.monotonic();journal=(out/f'history_rank{rank}.jsonl').open('a',buffering=1)
    for step in range(start_step,total):
        selected=[[sequences[order[((step*accum+a)*world+r)%len(order)]] for a in range(accum)] for r in range(world)]
        # Same arithmetic on all ranks, including variable sequence lengths.
        image_count=sum(len(f['views']) for batch in selected for s in batch for f in s['frames'][:5])
        localframes=[(s,fidx,f) for s in selected[rank] for fidx,f in enumerate(s['frames'][:5])]
        losses=[];history=[]
        for j,(seq,fidx,frame) in enumerate(localframes):
            if fidx==0:history=[]
            src,tgt,K,P,ids=load_frame(frame,cfg['data']['image_hw'],device)
            packed=pack_history(src,history)
            sync=nullcontext() if j==len(localframes)-1 else wrapped.no_sync()
            with sync:
                pred=wrapped(src,packed,K,P,ids,**model_inputs(frame,cfg['data']['image_hw'],device,cfg))
                loss,parts=criterion(pred,tgt)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                (loss*(len(ids)*world/image_count)).backward()
            losses.append(parts)
            old=[tensor_digest(h) for h in history[-4:]]
            history.append(quantized_history(pred))
            journal.write(json.dumps({'step':step+1,'sequence':seq['sequence_id'],'frame':frame['frame_index'],
              'timestamp_us':frame['timestamp_us'],'T':packed.shape[2],'history_hashes_oldest_first':old,
              'prediction_hash':tensor_digest(history[-1]),'camera_ids':ids.tolist()})+'\n')
            del loss,pred,packed
        if step<4:
            grad=check_gradients(model,step)
            save_json(auditdir/f'gradient_step{step+1}.json',grad)
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
        if rank==0:
            status={'status':'TRAINING','uid':os.getuid(),'world_size':world,'step':step+1,'total_steps':total,
              'elapsed_seconds':time.monotonic()-started,'rank0_losses':losses,'gradient_norm':float(norm),
              'valid_images_this_update':image_count,'peak_memory_bytes':torch.cuda.max_memory_allocated(),
              'training_recipe_id':cfg['training_recipe_id']}
            save_json(out/'STATUS.json',status);print('TRAIN_STEP',json.dumps(status),flush=True)
        if args.phase=='train' and ((step+1)%500==0 or step+1 in (5,total)):
            rng={'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(),'python':random.getstate(),'numpy':np.random.get_state()}
            torch.cuda.synchronize(device)
            all_rng=[None]*world;dist.all_gather_object(all_rng,rng,group=checkpoint_group)
            if rank==0:
                payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                  'step':step+1,'rng_by_rank':all_rng,'config':cfg,'sequence_order':order,
                  'data_sha256':hashlib.sha256(Path(cfg['data']['manifest']).read_bytes()).hexdigest()}
                checkpoint_path=out/'latest.pt';tmp=out/'latest.tmp.pt'
                print('CHECKPOINT_CPU_STAGING',step+1,flush=True)
                payload=checkpoint_to_cpu(payload)
                print('CHECKPOINT_CPU_READY',step+1,flush=True)
                with tmp.open('wb') as checkpoint_stream:
                    torch.save(payload,checkpoint_stream)
                    checkpoint_stream.flush();os.fsync(checkpoint_stream.fileno())
                if not zipfile.is_zipfile(tmp):raise RuntimeError('Incomplete checkpoint archive')
                tmp.replace(checkpoint_path)
                print('CHECKPOINT_SAVED',step+1,checkpoint_path.stat().st_size,flush=True)
                del payload
            dist.barrier(group=checkpoint_group)
    journal.close()
    dist.barrier(group=checkpoint_group)
    if rank==0:
        save_json(out/'RESULT.json',{'status':'PASS' if args.phase=='ddp_smoke' else 'TRAIN_COMPLETE',
          'world_size':world,'steps':total,'uid':os.getuid(),'generated_history':True,
          'implementation_sha256':cfg.get('implementation_sha256')})
        status=json.loads((out/'STATUS.json').read_text());status['status']='COMPLETE';save_json(out/'STATUS.json',status)
    dist.destroy_process_group()


if __name__=='__main__':main()
