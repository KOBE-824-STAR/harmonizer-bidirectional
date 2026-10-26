"""Opt-in launcher; original runs and source remain untouched.

--prepare-only resolves configs without importing the unavailable backbone.
Execution still requires the original Cosmos runtime and UID contract.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
IMAGE='harmonizer-cosmos-env:cuda128'


def precheck(gpus):
    assert os.getuid()==os.getgid()==1005
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    for gpu in gpus:
        uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        if any(line.split(',')[0].strip()==uuid for line in apps.splitlines()):
            raise RuntimeError(f'GPU {gpu} has active compute processes')
    names=subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()
    if any(n.startswith('double-dit-') for n in names): raise RuntimeError('Another double DiT container is running')


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['probe','smoke','native','ddp_smoke','train'])
    p.add_argument('--mode',required=True,choices=['bidirectional','one_way','joint','modulation','geometry'])
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--manifest',type=Path)
    p.add_argument('--steps',type=int);p.add_argument('--resume');a=p.parse_args()
    if a.phase=='train' and a.steps is None:
        a.steps=1000
    if a.steps is not None and a.steps<=0:
        p.error('--steps must be positive')
    gpus=[0,1,2,3] if a.phase in ('ddp_smoke','train') else [0]
    runroot=ROOT/'runs'/('injection_'+a.mode)
    out=runroot/a.phase
    out.mkdir(parents=True,exist_ok=True)
    if a.phase=='train' and (out/'STATUS.json').exists() and not a.resume:
        raise RuntimeError('Formal run exists; use an explicit matching checkpoint with --resume')
    cfg=json.loads((ROOT/'configs'/('injection_'+a.mode+'.json')).read_text())
    # Same official bytes, restored previously under a different local root.
    # Model construction still recomputes all three registered SHA256 hashes.
    restored=Path('/mnt/hlk/workspace/harmonizer_official_recipe_20261012/weights')
    cfg['backbone']['harmonizer_checkpoint']=str(restored/'diffusion_harmonizer.pkl')
    cfg['backbone']['cosmos_checkpoint_dir']=str(restored/'cosmos')
    cfg['data']['manifest']=str(a.manifest.resolve() if a.manifest else ROOT/'runs/data_v1/sequences.jsonl')
    cfg['geometry']['max_exposure_delta_us']=50000
    cfg['autoregressive']['max_history_gap_us']=1000000
    cfg['autoregressive']['protocol']='ar_generated_repeat_oldest_native_rgb_v1'
    cfg['autoregressive']['history_representation']='in_memory_uint8_roundtrip_equivalent_to_lossless_native_resolution_png'
    cfg['autoregressive']['history_image_hw']=cfg['data']['image_hw']
    cfg['autoregressive']['evaluation_resize_is_not_history_feedback']=True
    cfg['training_enabled']=not a.prepare_only
    cfg['run_control']={'effective_total_steps':a.steps or (4 if a.phase=='ddp_smoke' else None),
      'step_budget_source':'launcher_override',
      'note':'Training recipe retained for checkpoint compatibility; --steps is the actual stop limit.'}
    cfg['status']='prepared_not_executed' if a.prepare_only else ('implementation_validation' if a.phase!='train' else 'formal_training')
    cfg['implementation_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [ROOT/p for p in ['double_dit/model.py','double_dit/data.py','double_dit/injection.py',
            'double_dit/model_injection.py','double_dit/injection_data.py','scripts/train_injection.py','scripts/run_injection.py']]}
    if a.prepare_only:
        dest=out/'prepared_config.json'
        dest.write_text(json.dumps(cfg,indent=2)+'\n')
        print(json.dumps({'status':'PREPARED_NOT_EXECUTED','config':str(dest),
          'runtime_requirements':['UID/GID 1005','harmonizer-cosmos-env:cuda128 Docker image',
            '/mnt/hlk/workspace/Scheme_A&Scheme_B', '/mnt/hlk/workspace/harmonizer_official_recipe_20261012',
            'accessible manifest source/target images', 'depth metadata for geometry mode']}))
        return
    if os.getuid()!=1005 or os.getgid()!=1005:
        raise RuntimeError('Runtime not migrated: inherited Cosmos model requires UID/GID 1005. See INFORMATION_INJECTION.md. No training started.')
    precheck(gpus)
    if a.phase=='train':
        audit=json.loads((runroot/'ddp_smoke/RESULT.json').read_text())
        assert audit['status']=='PASS' and audit['world_size']==4
        assert audit['implementation_sha256']==cfg['implementation_sha256'],'Implementation changed after DDP validation'
        native=json.loads((runroot/'native/RESULT.json').read_text())
        assert native['status']=='PASS' and native['generated_history_verified']
        assert native['mode']==a.mode
        native_cfg=json.loads((runroot/'native/config.json').read_text())
        assert native_cfg['implementation_sha256']==cfg['implementation_sha256'], 'Rerun native validation after code changes'
        assert native_cfg['exchange']==cfg['exchange'], 'Rerun native validation for this mode'
        assert native_cfg['data']==cfg['data'], 'Native data configuration differs'
        assert native['image_hw']==cfg['data']['image_hw'], 'Rerun native validation at the configured training resolution'
        validated=json.loads((runroot/'ddp_smoke/config.json').read_text())
        assert validated['exchange']==cfg['exchange'], 'DDP injection configuration differs'
        assert validated['data']==cfg['data'], 'DDP data configuration differs'
        assert validated['data']['image_hw']==cfg['data']['image_hw'], 'Rerun DDP validation at the configured training resolution'
        assert validated['autoregressive']['history_image_hw']==cfg['autoregressive']['history_image_hw'], 'DDP history resolution differs'
        if a.resume and (out/'config.json').exists():
            previous=json.loads((out/'config.json').read_text())
            assert previous['data']['image_hw']==cfg['data']['image_hw'], 'A resolution change requires a separate run; preserve the original run and checkpoint'
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    if (out/'FAILED.json').exists():
        import time
        (out/'FAILED.json').rename(out/f'FAILED_{time.time_ns()}.json')
    name='double-dit-'+a.mode+'-'+a.phase
    cmd=['docker','run','--rm','--name',name,'--user','1005:1005','--userns=host',
      '--gpus','"device='+','.join(map(str,gpus))+'"','--network=none','--ipc=host',
      '-v','/mnt:/mnt:ro','-v',f'{ROOT}:{ROOT}:rw',
      '-v','/etc/passwd:/etc/passwd:ro','-v','/etc/group:/etc/group:ro']
    env={'XDG_CACHE_HOME':str(ROOT/'runtime/.cache'),
      'TORCHINDUCTOR_CACHE_DIR':str(ROOT/'runtime/inductor'),'TRITON_CACHE_DIR':str(ROOT/'runtime/triton'),
      'TORCH_HOME':'/mnt/ojc/models/torch_cache','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1',
      'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
      'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','WANDB_MODE':'offline',
      'NCCL_P2P_DISABLE':'1','NCCL_CUMEM_ENABLE':'0','NCCL_IB_DISABLE':'1',
      'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True',
      'CUDA_VISIBLE_DEVICES':','.join(map(str,range(len(gpus)))),
      'PYTHONPATH':str(ROOT)}
    for k,v in env.items(): cmd+=['-e',k+'='+v]
    cmd+=['-w',str(ROOT),'--entrypoint','python',IMAGE]
    if len(gpus)>1:cmd+=['-m','torch.distributed.run','--nproc_per_node',str(len(gpus)),'--master_port','29630']
    cmd += [str(ROOT/'scripts/train_injection.py'),'--config',str(out/'config.json'),'--output',str(out),'--phase',a.phase]
    if a.steps:cmd+=['--steps',str(a.steps)]
    if a.resume:cmd+=['--resume',a.resume]
    (ROOT/'logs').mkdir(exist_ok=True)
    log=ROOT/'logs'/f'injection_{a.mode}_{a.phase}.log'
    with (ROOT/'logs/launches.jsonl').open('a') as f:f.write(json.dumps({'uid':os.getuid(),'phase':a.phase,'gpus':gpus,'argv':cmd})+'\n')
    with log.open('a',buffering=1) as f:
        f.write(json.dumps({'phase':a.phase,'launcher_uid':os.getuid(),'gpus':gpus})+'\n')
        result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:
        (out/'FAILED.json').write_text(json.dumps({'exit_code':result.returncode,'log':str(log)},indent=2))
        raise SystemExit(result.returncode)


if __name__=='__main__':main()
