"""hlk-owned launcher; wait for a GPU with no existing compute process."""
import fcntl,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/mnt/hlk/demos/harmonizer_comparison/2026-10-21/double_dit_step1000')
def main():
 assert os.getuid()==os.getgid()==1005
 lock=(OUT/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert (OUT/'manifest.json').exists()
 names=subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()
 assert not any(n.startswith('double-dit-') for n in names)
 gpu=None
 while gpu is None:
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
  rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
  for row in rows:
   idx,uuid,mem=[x.strip() for x in row.split(',')]
   if uuid not in apps and int(mem)<1024:gpu=int(idx);break
  if gpu is None:
   (OUT/'LAUNCH.json').write_text(json.dumps(dict(status='WAITING_FOR_IDLE_GPU',uid=os.getuid(),tmux_socket='double_dit_comparison',tmux_session='comparison_20261021')))
   print('Waiting for a GPU with no compute processes',flush=True);time.sleep(5)
 state=dict(status='LAUNCHING',uid=os.getuid(),gid=os.getgid(),gpu=gpu,tmux_socket='double_dit_comparison',tmux_session='comparison_20261021',output=str(OUT),log=str(OUT/'inference.log'))
 (OUT/'LAUNCH.json').write_text(json.dumps(state,indent=2));print(state,flush=True)
 cmd=['docker','run','--rm','--name','double-dit-comparison-20261021','--user','1005:1005','--userns=host','--gpus','device='+str(gpu),'--network=none','--ipc=host',
  '-v','/mnt:/mnt:ro','-v',f'{ROOT}:{ROOT}:rw','-v',f'{OUT}:{OUT}:rw','-v','/etc/passwd:/etc/passwd:ro','-v','/etc/group:/etc/group:ro']
 env={'XDG_CACHE_HOME':str(ROOT/'runtime/.cache'),'TORCHINDUCTOR_CACHE_DIR':str(ROOT/'runtime/inductor'),'TRITON_CACHE_DIR':str(ROOT/'runtime/triton'),
  'TORCH_HOME':'/mnt/ojc/models/torch_cache','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
  'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','WANDB_MODE':'offline','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','CUDA_VISIBLE_DEVICES':'0','PYTHONPATH':str(ROOT)}
 for k,v in env.items():cmd+=['-e',k+'='+v]
 cmd+=['-w',str(ROOT),'--entrypoint','python','harmonizer-cosmos-env:cuda128',str(ROOT/'scripts/infer_comparison.py'),'--output',str(OUT)]
 with (OUT/'inference.log').open('a',buffering=1) as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 state.update(status='COMPLETE' if r.returncode==0 else 'FAILED',exit_code=r.returncode);(OUT/'LAUNCH.json').write_text(json.dumps(state,indent=2))
 print(state,flush=True);raise SystemExit(r.returncode)
if __name__=='__main__':main()
