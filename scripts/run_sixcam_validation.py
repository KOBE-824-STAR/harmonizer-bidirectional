"""Launch bounded single-GPU or four-GPU six-camera validation."""
import argparse,fcntl,hashlib,json,os,subprocess,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['one_way','joint','modulation'])
    p.add_argument('--kind',required=True,choices=['single','ddp4']);p.add_argument('--gpus',required=True)
    p.add_argument('--tag',default='sixcam_v1');p.add_argument('--disable-p2p',action='store_true')
    p.add_argument('--legacy-host-shm',action='store_true',help='NCCL 2.26.2 host-allocation compatibility')
    a=p.parse_args();gpus=[int(g) for g in a.gpus.split(',')]
    assert len(gpus)==len(set(gpus))==(1 if a.kind=='single' else 4)
    assert a.tag.replace('_','').replace('-','').isalnum()
    assert os.getuid()==ROOT.stat().st_uid and os.getuid()!=0
    locks=[]
    for gpu in sorted(gpus):
        f=(ROOT/f'runtime/tanhe/gpu{gpu}.validation.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    info=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total','--format=csv,noheader,nounits'],text=True)
    uuids={int(line.split(',')[0]):line.split(',')[1].strip() for line in info.splitlines()}
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert not any(line.split(',')[0].strip() in {uuids[g] for g in gpus} for line in apps.splitlines()),'Selected GPU occupied'
    out=ROOT/'runs'/f'tanhe_{a.mode}'/f'{a.tag}_{a.kind}'
    out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((ROOT/f'runs/tanhe_{a.mode}/native/config.json').read_text())
    cfg['data']['manifest']=str(ROOT/'data/tanhe_sixcam_validation/sequences.jsonl')
    cfg['data']['manifest_sha256']=sha(cfg['data']['manifest'])
    cfg['data'].update(allow_variable_views=False,max_views=6)
    cfg['runtime'].update(gpus=gpus,validation_kind=a.kind)
    cfg['training_enabled']=False
    cfg['validation']={'steps':4,'formal_training':False,'camera_count':6,'kind':a.kind,'scope':'asynchronous_real_exposure_engineering_diagnostic'}
    cfg['implementation_sha256']={rel:sha(ROOT/rel) for rel in cfg['implementation_sha256']}
    cfg['implementation_sha256'].update({str(f.relative_to(ROOT)):sha(f) for f in [ROOT/'scripts/validate_sixcam.py',Path(__file__),ROOT/'scripts/prepare_sixcam_validation.py']})
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpus,PYTHONPATH=os.pathsep.join([str(ROOT),str(ROOT/'runtime/tanhe/vendor')]),
        PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TORCH_HOME=str(ROOT/'runtime/tanhe/torch_cache'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',TORCH_NCCL_ASYNC_ERROR_HANDLING='1',NCCL_DEBUG='WARN')
    if a.disable_p2p:env['NCCL_P2P_DISABLE']='1'
    if a.legacy_host_shm:env['NCCL_CUMEM_HOST_ENABLE']='0'
    py=str(ROOT/'runtime/tanhe/venv/bin/python')
    cmd=[py]
    if a.kind=='ddp4':cmd+=['-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=4']
    cmd += [str(ROOT/'scripts/validate_sixcam.py'),'--config',str(out/'config.json'),'--output',str(out),'--kind',a.kind]
    log=ROOT/f'logs/{a.tag}_{a.mode}_{a.kind}.log'
    state={'state':'RUNNING','uid':os.getuid(),'started':time.time(),'command':cmd,'gpus':gpus,'p2p_disabled':a.disable_p2p,
           'nccl_environment':{k:v for k,v in env.items() if k.startswith('NCCL_')},'log':str(log)}
    (out/'LAUNCH.json').write_text(json.dumps(state,indent=2))
    stop=threading.Event();samples=[];errors=[]
    def monitor():
        while not stop.is_set():
            try:
                raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10)
                rows=[]
                for line in raw.splitlines():
                    index,mem,util=[int(v.strip()) for v in line.split(',')]
                    if index in gpus:rows.append({'gpu':index,'used_mib':mem,'utilization_percent':util})
                samples.append({'time':time.time(),'gpus':rows})
            except Exception as e:errors.append(str(e))
            stop.wait(1)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    with log.open('w') as f:result=subprocess.run(cmd,env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    stop.set();thread.join()
    memory={'sample_interval_seconds':1,'samples':samples,'errors':errors,
            'peak_device_used_mib':{str(g):max((r['used_mib'] for s in samples for r in s['gpus'] if r['gpu']==g),default=0) for g in gpus}}
    (out/'device_memory.json').write_text(json.dumps(memory,indent=2))
    passed=result.returncode==0 and (out/'RESULT.json').is_file() and json.loads((out/'RESULT.json').read_text())['status']=='PASS'
    state.update(state='PASS' if passed else 'FAILED',exit_code=result.returncode,finished=time.time())
    (out/'LAUNCH.json').write_text(json.dumps(state,indent=2));print(json.dumps(state),flush=True)
    raise SystemExit(0 if passed else 1)

if __name__=='__main__':main()
