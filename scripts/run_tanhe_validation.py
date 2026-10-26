"""Native, project-local validation launcher; no Docker and no formal training."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/'runtime/tanhe'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',required=True,choices=['one_way','joint','modulation'])
    p.add_argument('--phase',default='native',choices=['smoke','native'])
    p.add_argument('--gpu',type=int,required=True)
    args=p.parse_args()
    if os.getuid()!=ROOT.stat().st_uid or os.getuid()==0:
        raise RuntimeError('Must run as the non-root project owner')
    out=ROOT/'runs'/('tanhe_'+args.mode)/args.phase;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'validation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'RESULT.json').exists():raise RuntimeError('Completed validation exists; retain it, choose a new version for reruns')
    uuid=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    if any(line.split(',')[0].strip()==uuid for line in apps.splitlines()):raise RuntimeError(f'GPU {args.gpu} is occupied')
    cfg=json.loads((ROOT/f'configs/injection_{args.mode}.json').read_text())
    weights=ROOT.parent/'harmonizer'
    cfg['backbone']['harmonizer_checkpoint']=str(weights/'models/diffusion_harmonizer.pkl')
    cfg['backbone']['cosmos_checkpoint_dir']=str(weights/'src/checkpoints/nvidia/Cosmos-Predict2-0.6B-Text2Image')
    cfg['data']['manifest']=str(ROOT/'data/tanhe_restored/sequences.jsonl')
    if not Path(cfg['data']['manifest']).is_file():raise FileNotFoundError('Run verified data restoration first')
    cfg['data']['manifest_sha256']=sha(cfg['data']['manifest'])
    cfg['runtime']={'uid':os.getuid(),'gid':os.getgid(),'host':'tanhe-server','python':str(RUNTIME/'venv/bin/python'),
        'vendor':str(RUNTIME/'vendor'),'device_index':args.gpu,'original_runtime_uid':1005}
    cfg['implementation_sha256']={rel:sha(ROOT/rel) for rel in ['double_dit/model_tanhe.py','double_dit/model_injection_tanhe.py',
        'double_dit/injection.py','double_dit/perceptual_tanhe.py','double_dit/data.py','double_dit/injection_data.py','scripts/train_tanhe.py','scripts/run_tanhe_validation.py']}
    cfg['autoregressive'].update(protocol='ar_generated_repeat_oldest_native_rgb_v1',
        history_representation='in_memory_uint8_roundtrip_equivalent_to_lossless_native_resolution_png',history_image_hw=cfg['data']['image_hw'])
    cfg['status']='real_model_validation';cfg['training_enabled']=False
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    log=ROOT/f'logs/tanhe_{args.mode}_{args.phase}.log'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),PYTHONPATH=os.pathsep.join([str(ROOT),str(RUNTIME/'vendor')]),
        PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TORCH_HOME=str(RUNTIME/'torch_cache'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    cmd=[str(RUNTIME/'venv/bin/python'),str(ROOT/'scripts/train_tanhe.py'),'--config',str(out/'config.json'),'--output',str(out),'--phase',args.phase]
    status={'state':'RUNNING','uid':os.getuid(),'gpu':args.gpu,'mode':args.mode,'phase':args.phase,'started':time.time(),'log':str(log),'command':cmd}
    (out/'LAUNCH.json').write_text(json.dumps(status,indent=2))
    with log.open('a',buffering=1) as f:
        f.write(json.dumps(status)+'\n')
        result=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
    status.update(state='PASS' if result.returncode==0 else 'FAILED',exit_code=result.returncode,finished=time.time())
    (out/'LAUNCH.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
    raise SystemExit(result.returncode)

if __name__=='__main__':main()
