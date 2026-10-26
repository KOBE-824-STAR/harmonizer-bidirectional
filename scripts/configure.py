"""Resolve local paths without changing the archived model/training recipe."""
import argparse,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--weights-root',type=Path,default=ROOT/'weights');p.add_argument('--output',type=Path,default=ROOT/'configs/local.json');a=p.parse_args()
 original=ROOT/'whole_data/runs/training_restart_20261024/bidirectional/config.json'
 cfg=json.loads(original.read_text());w=a.weights_root.resolve();manifest=a.manifest.resolve()
 actual=hashlib.sha256(manifest.read_bytes()).hexdigest()
 if actual!=cfg['data']['manifest_sha256']:raise ValueError('Manifest bytes must match the archived training manifest; use original server paths or restore the same dataset layout. Arbitrary data requires a separately documented experiment.')
 cfg['data']['manifest']=str(manifest)
 cfg['backbone']['harmonizer_checkpoint']=str(w/'models/diffusion_harmonizer.pkl')
 cfg['backbone']['cosmos_checkpoint_dir']=str(w/'src/checkpoints/nvidia/Cosmos-Predict2-0.6B-Text2Image')
 cfg['runtime'].update(uid=os.getuid(),gid=os.getgid(),python=sys.executable,vendor=str(ROOT/'runtime/tanhe/vendor'))
 cfg['portable_archive']={'original_config_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'runtime_only_trainer_change':'owner check resolves repository root','trainer_sha256':hashlib.sha256((ROOT/'scripts/train_bidirectional.py').read_bytes()).hexdigest()}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(cfg,indent=2)+'\n');print(a.output.resolve())
if __name__=='__main__':main()
