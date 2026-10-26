"""Strict step-1000 loading; chronological generated RGB feedback, six views."""
import argparse,fcntl,gc,hashlib,importlib.util,json,os,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from double_dit.model import DoubleDiT
from double_dit.data import pack_history,quantized_history,tensor_digest
ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/mnt/hlk/demos/harmonizer_comparison/2026-10-19')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 return h.hexdigest()
def write(path,obj):
 t=path.with_suffix('.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2));t.replace(path)
def render(out,manifest,audits):
 import shutil
 spec=importlib.util.spec_from_file_location('layout',OLD/'render_comparison.py');layout=importlib.util.module_from_spec(spec);spec.loader.exec_module(layout)
 report=json.loads((OLD/'REPORT.json').read_text());figures=[];full=[];details=[];records=[]
 number='10';name='双路DiT模型1000步五帧自回归'
 labels=['输入图像','原场景参考图','官方模型\n五帧自回归（未训练）','官方模型\n五帧自回归（训练二）','双路 DiT / 1000步\n原生分辨率自回归']
 bygroup={x['group_id']:x for x in audits}
 allnames=['00_input.png','00_gt.png']+[f'{n}_{s}.png' for n,s,_,_ in layout.MODELS]+[f'{number}_{name}.png']
 alllabels=layout.COLUMN_LABELS+['10 双路 DiT\n1000步 / 六视角\n五帧自回归']
 for s in report['scenes']:
  key=f"{s['scene']}_frame_{s['frame']:06d}";rows=[];visible=[];zoomrows=[];allrows=[]
  audit=bygroup[s['group_id']]
  for cam in layout.CAMERAS:
   dest=out/key/cam;dest.mkdir(parents=True,exist_ok=True)
   for fname in allnames[:-1]:shutil.copyfile(OLD/key/cam/fname,dest/fname)
   src=out/'predictions_native'/f"{s['group_id']}__{cam}.png"
   with Image.open(src) as im:im.resize((512,288),Image.Resampling.BILINEAR).save(dest/allnames[-1])
   pics=[layout.image_at(dest/f) for f in allnames];chosen=[pics[i] for i in (0,1,2,10,11)]
   rows.append((layout.CAMERA_LABELS[cam],chosen));allrows.append((layout.CAMERA_LABELS[cam],pics))
   records.append(dict(group_id=s['group_id'],camera=cam,native=str(src),published=str(dest/allnames[-1]),sha256=sha(dest/allnames[-1])))
   if cam in s['visible_cameras']:
    label=f"{s['scene']} / 第{s['frame']}帧\n{layout.CAMERA_LABELS[cam]} / T{audit['T']}"
    visible.append((label,chosen));full.append((label,chosen))
    z=[layout.zoom(p,s['crop_boxes'][cam]) for p in chosen];zoomrows.append((label,z));details.append((label,z))
  sub='双路DiT：1024×576生成，512×288展示；历史协议与旧模型不同，见INDEX.txt'
  layout.save_figure(out/key/'comparison_all_cameras.png',rows,labels,key,sub,figures)
  layout.save_figure(out/key/'comparison_all_models.png',allrows,alllabels,key,sub,figures)
  layout.save_figure(out/key/'comparison_detail_crops.png',zoomrows,labels,key,sub,figures)
 layout.save_figure(out/'overview_full_images.png',full,labels,'双路 DiT 与既有模型：相同场景、相同输入',sub,figures)
 layout.save_figure(out/'overview_detail_crops.png',details,labels,'双路 DiT 与既有模型：相同资产区域',sub,figures)
 (out/'INDEX.txt').write_text('双路 DiT step1000 对比\n\n入口：overview_full_images.png、overview_detail_crops.png。\n5组场景帧、30张六相机输出；中间自回归前缀保存在 predictions_native/。\n每个场景目录另含 comparison_all_models.png，覆盖原有九模型。\n\n本模型：576×1024网络分辨率，双路各20层，6相机同一时刻联合条件；首帧T1，此后T5，历史不足4帧时重复最早已生成输出。历史在原生分辨率uint8往返，未使用GT或原图作为历史。展示缩到288×512，展示图不用于反馈。\n旧模型：各自的历史训练/推理尺寸不同；既有T5模型前4次观测用T1，之后使用已发布288×512输出反馈。本次按double_DiT训练协议生成，所以是同输入定性比较，不是统一时序协议的量化评测。\n原场景参考图不含插入资产，不是和谐化GT。裁剪区域完全复用旧报告。\n本模型只在双视角数据上完成1000步训练，此处六视角推理属于视角数量扩展。\n未计算九项质量指标，本批次不提供量化质量结论。\n详情见manifest.json、history_audit.jsonl、REPORT.json。\n')
 return records,figures

def main():
 assert os.getuid()==os.getgid()==1005
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();out=Path(args.output)
 lock=(out/'inference.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (out/'REPORT.json').exists():raise RuntimeError('Completed run already exists')
 manifest=json.loads((out/'manifest.json').read_text());cfg=json.loads((ROOT/'runs/train/config.json').read_text())
 for rel,h in cfg['implementation_sha256'].items():assert sha(ROOT/rel)==h,rel
 assert cfg['autoregressive']['protocol']==manifest['protocol']
 torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(4)
 write(out/'STATUS.json',dict(status='LOADING',uid=os.getuid(),step=0,total_groups=manifest['total_groups']))
 checkpoint=ROOT/'runs/train/latest.pt';checkpoint_hash=sha(checkpoint)
 # Load tensors and the minimal NumPy RNG types with the restricted weights loader.
 with torch.serialization.safe_globals([np.core.multiarray._reconstruct,np.ndarray,np.dtype,type(np.dtype('uint32'))]):
  saved=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
 assert saved['step']==1000 and saved['config']['implementation_sha256']==cfg['implementation_sha256']
 (out/'audit').mkdir(exist_ok=True)
 model=DoubleDiT(cfg,out/'audit','cuda');model.load_state_dict(saved['model'],strict=True);del saved;gc.collect()
 model.eval();model.requires_grad_(False)
 write(out/'audit/checkpoint.json',dict(path=str(checkpoint),sha256=checkpoint_hash,step=1000,strict=True,uid=os.getuid(),config=cfg))
 audits=[];done=0;start=time.monotonic();(out/'predictions_native').mkdir(exist_ok=True)
 with (out/'history_audit.jsonl').open('w',buffering=1) as journal,torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  for seq in manifest['sequences']:
   history=[];history_ids=[]
   for f in seq['frames']:
    src=[];Ks=[];Ps=[];ids=[]
    H,W=manifest['network_hw']
    for v in f['views']:
     assert sha(v['source'])==v['source_sha256']
     with Image.open(v['source']) as im:a=np.asarray(im.convert('RGB').resize((W,H),Image.Resampling.BILINEAR)).copy()
     src.append(torch.from_numpy(a).permute(2,0,1).float()/127.5-1)
     K=np.array(v['K_native']);K[0]*=W/v['native_hw'][1];K[1]*=H/v['native_hw'][0]
     Ks.append(K);Ps.append(v['T_reference_from_cam']);ids.append(v['camera_id'])
    src=torch.stack(src).cuda().to(torch.bfloat16);K=torch.tensor(np.array(Ks),device='cuda',dtype=torch.float32)
    P=torch.tensor(Ps,device='cuda',dtype=torch.float32);ids=torch.tensor(ids,device='cuda')
    packed=pack_history(src,history);refs=list(reversed(history[-4:]));ref_ids=list(reversed(history_ids[-4:]))
    if refs:
     while len(refs)<4:refs.append(refs[-1]);ref_ids.append(ref_ids[-1])
     for j,h in enumerate(refs,1):assert torch.equal(packed[:,:,j],h)
    print('GENERATE',done+1,manifest['total_groups'],f['group_id'],'T',packed.shape[2],flush=True)
    pred=model(src,packed,K,P,ids);assert tuple(pred.shape)==(6,3,H,W) and torch.isfinite(pred).all()
    cached=quantized_history(pred);outputs={}
    for i,v in enumerate(f['views']):
     a=((pred[i].float().clamp(-1,1)+1)*127.5).to(torch.uint8).permute(1,2,0).cpu().numpy()
     dest=out/'predictions_native'/f"{f['group_id']}__{v['camera']}.png";Image.fromarray(a).save(dest);outputs[str(dest)]=sha(dest)
     recovered=torch.from_numpy(a.copy()).permute(2,0,1).to(device='cuda').float().div(127.5).sub(1).to(dtype=cached.dtype)
     assert torch.equal(cached[i],recovered)
    audit=dict(group_id=f['group_id'],scene=seq['scene'],frame=f['frame_index'],timestamp_us=f['timestamp_us'],T=packed.shape[2],
     history_groups_newest_first=ref_ids,history_hashes_newest_first=[tensor_digest(h) for h in refs],prediction_hash=tensor_digest(cached),
     outputs=outputs,camera_order=[v['camera'] for v in f['views']],adjacency=model.last_edges)
    audits.append(audit);journal.write(json.dumps(audit)+'\n')
    history.append(cached);history=history[-4:];history_ids.append(f['group_id']);history_ids=history_ids[-4:]
    done+=1
    write(out/'STATUS.json',dict(status='GENERATING',uid=os.getuid(),checkpoint_step=1000,step=done,total_groups=manifest['total_groups'],
      current_group=f['group_id'],T=packed.shape[2],elapsed_seconds=time.monotonic()-start,peak_memory_bytes=torch.cuda.max_memory_allocated()))
    del src,K,P,ids,packed,pred
 records,figures=render(out,manifest,audits)
 result=dict(status='COMPLETE',uid=os.getuid(),checkpoint_step=1000,checkpoint_sha256=checkpoint_hash,groups=done,native_outputs=done*6,
  selected_outputs=len(records),images=records,figures=figures,protocol=manifest['protocol'],network_hw=manifest['network_hw'],publish_hw=manifest['publish_hw'],
  metrics_status='NOT_COMPUTED',limitations=['trained on two views; six-view inference is an extension','historical baselines use different history startup and resize protocols'],
  elapsed_seconds=time.monotonic()-start,peak_memory_bytes=torch.cuda.max_memory_allocated())
 write(out/'REPORT.json',result);write(out/'STATUS.json',dict(result,step=done,total_groups=done));print('COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
