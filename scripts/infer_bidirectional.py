"""Strict six-camera autoregressive inference with the trained bidirectional model."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runtime/tanhe/vendor')]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--max-sequences',type=int);a=p.parse_args()
 cfg=json.loads(a.config.read_text())
 if cfg['exchange']['mode']!='bidirectional':raise ValueError('Wrong model mode')
 if a.max_sequences is not None and a.max_sequences<1:raise ValueError('max-sequences must be positive')
 a.output.mkdir(parents=True,exist_ok=False)
 import numpy as np
 import torch
 from PIL import Image
 from double_dit.data import load_frame,pack_history,quantized_history
 from double_dit.model_injection_tanhe import DoubleDiT
 torch.manual_seed(42);device=torch.device(a.device);torch.cuda.set_device(device)
 # Restricted load: these NumPy classes are needed by the original RNG record.
 with torch.serialization.safe_globals([np.core.multiarray._reconstruct,np.ndarray,np.dtype,type(np.dtype('uint32'))]):
  saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True,mmap=True)
 if saved['config']['exchange']!=cfg['exchange'] or saved['config']['training']!=cfg['training']:raise ValueError('Checkpoint/config recipe mismatch')
 if saved['data_sha256']!=cfg['data']['manifest_sha256']:raise ValueError('Training manifest mismatch')
 audit=a.output/'model_audit';audit.mkdir()
 model=DoubleDiT(cfg,audit,str(device));model.load_state_dict(saved['model'],strict=True)
 step=saved['step'];del saved
 model.eval().requires_grad_(False);images=a.output/'predictions';images.mkdir()
 views=frames=sequences=0;started=time.monotonic()
 with (a.output/'predictions_manifest.jsonl').open('w',buffering=1) as journal,torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  with a.manifest.open() as stream:
   for line in stream:
    if not line.strip():continue
    if a.max_sequences is not None and sequences>=a.max_sequences:break
    seq=json.loads(line)
    if seq.get('split')!='val':raise ValueError('Inference manifest must contain only val sequences')
    history=[];history_ids=[];times=[]
    for frame in seq['frames']:
     if [v['camera_id'] for v in frame['views']]!=list(range(6)):raise ValueError('Expected canonical six-camera frame')
     t=frame['timestamp_us']
     if times and t<=times[-1]:raise ValueError('Nonchronological sequence')
     times.append(t)
     src,_,k,poses,ids=load_frame(frame,cfg['data']['evaluation_hw'],device)
     packed=pack_history(src,history);pred=model(src,packed,k,poses,ids)
     if pred.shape!=src.shape or not torch.isfinite(pred).all():raise ValueError('Invalid prediction')
     feedback=quantized_history(pred)
     for i,view in enumerate(frame['views']):
      path=images/(frame['group_id']+'__'+view['camera']+'.png')
      if path.exists():raise ValueError('Duplicate view')
      pixels=((pred[i].float().clamp(-1,1)+1)*127.5).to(torch.uint8).permute(1,2,0).cpu().numpy();Image.fromarray(pixels).save(path)
      journal.write(json.dumps({'sequence_id':seq['sequence_id'],'group_id':frame['group_id'],'camera':view['camera'],'prediction':str(path.resolve()),'prediction_sha256':sha(path),'source':view['source'],'gt':view['target'],'T':int(packed.shape[2]),'history_groups_newest_first':list(reversed(history_ids[-4:]))})+'\n');views+=1
     history=(history+[feedback])[-4:];history_ids=(history_ids+[frame['group_id']])[-4:];frames+=1
     (a.output/'STATUS.json').write_text(json.dumps({'status':'GENERATING','frames':frames,'views':views,'elapsed_seconds':time.monotonic()-started}))
     del src,pred,packed,k,poses,ids
    sequences+=1
 result={'status':'COMPLETE','mode':'bidirectional','checkpoint_sha256':sha(a.checkpoint),'optimizer_step':step,'manifest_sha256':sha(a.manifest),'sequences':sequences,'six_camera_frames':frames,'predicted_views':views,'elapsed_seconds':time.monotonic()-started,'metrics_computed':False}
 (a.output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
