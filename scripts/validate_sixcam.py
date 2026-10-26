"""Bounded six-camera diagnostics only: never save a training checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from contextlib import nullcontext
from datetime import timedelta
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from train_tanhe import smoke, check_gradients, save_json
from double_dit.model_tanhe import OfficialLoss
from double_dit.model_injection_tanhe import DoubleDiT
from double_dit.injection_data import model_inputs
from double_dit.data import load_sequences, load_frame, pack_history, quantized_history, tensor_digest


def parameter_digest(model, gradients=False):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        value=p.grad if gradients else p.detach()
        h.update(name.encode())
        if value is None:
            h.update(b'none');continue
        h.update(value.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def memory():
    torch.cuda.synchronize()
    free,total=torch.cuda.mem_get_info()
    return dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                device_total_bytes=total,device_free_bytes_at_end=free)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--kind',choices=['single','ddp4'],required=True)
    a=p.parse_args();out=Path(a.output);cfg=json.loads(Path(a.config).read_text())
    root=Path(__file__).resolve().parents[1]
    assert os.getuid()==root.stat().st_uid and os.getuid()!=0
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    assert world==(1 if a.kind=='single' else 4)
    torch.cuda.set_device(local)
    if world>1:dist.init_process_group('nccl',timeout=timedelta(minutes=3))
    torch.manual_seed(42);np.random.seed(42);random.seed(42)
    sequences=load_sequences(cfg['data']['manifest'],'train')
    assert len(sequences)>=4
    for s in sequences:
        assert len(s['frames'])==2
        for f in s['frames']:
            assert len(f['views'])==6
            assert sorted(v['camera_id'] for v in f['views'])==list(range(6))
    audit=out/f'rank{rank}';audit.mkdir(exist_ok=True)
    device=torch.device('cuda',local)
    model=DoubleDiT(cfg,audit,str(device));criterion=OfficialLoss(device)
    if a.kind=='single':
        smoke(model,criterion,sequences[0],cfg['data']['image_hw'],out)
        result=json.loads((out/'RESULT.json').read_text())
        result.update(cameras_per_frame=6,diagnostic_optimizer_steps=4,memory_by_rank=[memory()],
                      sequence_ids=[sequences[0]['sequence_id']],formal_training=False)
        save_json(out/'RESULT.json',result)
        return
    model.train()
    wrapped=DDP(model,device_ids=[local],broadcast_buffers=False,find_unused_parameters=True,gradient_as_bucket_view=True)
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=2e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    histories=[];gradients=[];syncs=[];selections=[];started=time.monotonic()
    watched={n:p for n,p in model.named_parameters() if p.requires_grad and (n.startswith('pipeline.dit.blocks.0.') or n.startswith('vae.encoder.'))}
    changes=None
    for step in range(4):
        seq=sequences[(rank+step)%len(sequences)];history=[];journal=[];parts_list=[]
        for j,frame in enumerate(seq['frames']):
            src,tgt,K,P,ids=load_frame(frame,cfg['data']['image_hw'],device)
            packed=pack_history(src,history)
            with (wrapped.no_sync() if j==0 else nullcontext()):
                pred=wrapped(src,packed,K,P,ids,**model_inputs(frame,cfg['data']['image_hw'],device,cfg))
                loss,parts=criterion(pred,tgt)
                assert bool(torch.isfinite(loss))
                # All ranks have exactly 2 frames * 6 cameras. DDP averages ranks.
                (loss/2).backward()
            journal.append({'frame_index':frame['frame_index'],'T':packed.shape[2],
                'camera_ids':ids.tolist(),'history':[tensor_digest(h) for h in history]})
            history.append(quantized_history(pred));journal[-1]['prediction']=tensor_digest(history[-1])
            parts_list.append(parts)
            del pred,loss,packed
        assert [x['T'] for x in journal]==[1,5]
        assert journal[1]['history']==[journal[0]['prediction']]
        histories.append(journal);selections.append(seq['sequence_id'])
        gradients.append(check_gradients(model,step))
        grad_hash=parameter_digest(model,gradients=True)
        hashes=[None]*world;dist.all_gather_object(hashes,grad_hash)
        assert len(set(hashes))==1,('gradient_sync_failed',hashes)
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        before={n:p.detach().clone() for n,p in watched.items()} if step==0 else {}
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        if step==0:
            changes={prefix:sum(not torch.equal(before[n],p) for n,p in watched.items() if n.startswith(prefix)) for prefix in ['pipeline.dit.blocks.0.','vae.encoder.']}
            assert all(changes.values()),changes
        del before
        syncs.append({'step':step+1,'gradient_sha256_by_rank':hashes})
        print('DIAGNOSTIC_STEP',json.dumps({'rank':rank,'step':step+1,'loss_parts':parts_list,
              'gradient_norm':float(norm),'peak_allocated_bytes':torch.cuda.max_memory_allocated()}),flush=True)
    digest=parameter_digest(model);all_digest=[None]*world;dist.all_gather_object(all_digest,digest)
    assert len(set(all_digest))==1,('parameter_sync_failed',all_digest)
    save_json(audit/'history_audit.json',histories);save_json(audit/'gradient_audit.json',gradients)
    save_json(audit/'synchronization_audit.json',syncs)
    local_result={'rank':rank,'sequence_ids':selections,'memory':memory(),'main_vae_parameter_updates':changes,
                  'generated_history_verified':True,'final_parameters_sha256':digest}
    save_json(audit/'RESULT.json',local_result)
    all_results=[None]*world;dist.all_gather_object(all_results,local_result)
    if rank==0:
        save_json(out/'RESULT.json',{'status':'PASS','mode':model.mode,'kind':a.kind,'world_size':world,
            'cameras_per_frame':6,'image_hw':cfg['data']['image_hw'],'diagnostic_optimizer_steps':4,
            'global_images_per_optimizer_step':48,'generated_history_verified':True,
            'all_step_gradients_identical_across_ranks':True,'final_parameters_identical_across_ranks':True,
            'memory_by_rank':[r['memory'] for r in all_results],'ranks':all_results,
            'elapsed_seconds':time.monotonic()-started,'formal_training':False})
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
