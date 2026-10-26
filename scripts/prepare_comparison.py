"""Prepare the established five demo frames plus their genuine AR prefixes."""
import hashlib,json,os,sys
from pathlib import Path
import numpy as np
from PIL import Image
from prepare_data import json_items,transform,ORDER,NUSC
ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/mnt/hlk/demos/harmonizer_comparison/2026-10-19')
OUT=Path('/mnt/hlk/demos/harmonizer_comparison/2026-10-21/double_dit_step1000')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert os.getuid()==os.getgid()==1005
 OUT.mkdir(parents=True,exist_ok=True)
 report=json.loads((OLD/'REPORT.json').read_text()); wanted={r['group_id'] for r in report['scenes']}
 planpath=ROOT.parent/'harmonizer_fiveway_512_ar_20261015/manifests/autoregressive_sequences.json'
 plan=json.loads(planpath.read_text()); selected=[]
 for s in plan['sequences']:
  indices=[i for i,g in enumerate(s['groups']) if g in wanted]
  if indices:selected.append(dict(s,groups=s['groups'][:max(indices)+1]))
 gids={g for s in selected for g in s['groups']}
 metapath=ROOT.parent/'harmonizer_autoregressive_20261015/manifests/source_val_groups.jsonl'
 meta={}
 for line in metapath.open():
  r=json.loads(line)
  if r['group_id'] in gids:meta[r['group_id']]=r
 assert set(meta)==gids
 tokens={v['exposure']['sample_data_token'] for m in meta.values() for v in m['views']}
 sd={r['token']:r for r in json_items(NUSC/'sample_data.json') if r['token'] in tokens};assert set(sd)==tokens
 poses_needed={r['ego_pose_token'] for r in sd.values()}
 poses={r['token']:r for r in json_items(NUSC/'ego_pose.json') if r['token'] in poses_needed}
 cals={r['token']:r for r in json_items(NUSC/'calibrated_sensor.json')}
 rows={}
 for line in (ROOT.parent/'harmonizer_512_eval_20261019/manifests/eval_5713.jsonl').open():
  r=json.loads(line)
  if r['group_id'] in gids:rows[(r['group_id'],r['camera_id'])]=r
 frames={}; missing=[]; recovered=[]
 for gid,m in meta.items():
  views=sorted(m['views'],key=lambda v:ORDER.index(v['camera_name']))
  assert len(views)==6 and {v['camera_name'] for v in views}==set(ORDER)
  ref=transform(poses[sd[views[0]['exposure']['sample_data_token']]['ego_pose_token']])
  frame=dict(group_id=gid,scene=m['scene_name'],frame_index=m['frame_index'],sample_token=m['sample_token'],views=[])
  timestamps=[]
  for v in views:
   cam=v['camera_name'];s=sd[v['exposure']['sample_data_token']];cal=cals[s['calibrated_sensor_token']]
   timestamps.append(s['timestamp']);role=v['role_sources']['input'];expected=role['content_sha256']
   row=rows[(gid,cam)];assert row['input_sha256']==expected
   candidates=[OUT/'recovered_inputs'/f'{gid}__{cam}__input.png',Path(row['input_path']),Path(role['path']),OLD/f"{m['scene_name']}_frame_{m['frame_index']:06d}"/cam/'00_input.png']
   source=next((p for p in candidates if p.is_file() and sha(p)==expected),None)
   if source is None:
    figure=OLD.parent/'2026-10-08/autoregressive_20261015/figures'/gid/'all_cameras.png'
    if figure.is_file():
     y=168+ORDER.index(cam)*292;box=(204,y,716,y+288)
     dest=OUT/'recovered_inputs'/f'{gid}__{cam}__input.png';dest.parent.mkdir(exist_ok=True)
     with Image.open(figure) as im:im.crop(box).save(dest)
     if sha(dest)==expected:
      source=dest;recovered.append(dict(group_id=gid,camera=cam,source_figure=str(figure),box=box,output=str(dest),sha256=expected,exact_original_png_bytes=True))
     else:dest.unlink()
   if source is None:missing.append(dict(group_id=gid,camera=cam,candidates=list(map(str,candidates))));continue
   with Image.open(source) as im:assert im.size==(512,288)
   T=np.linalg.inv(ref)@transform(poses[s['ego_pose_token']])@transform(cal)
   frame['views'].append(dict(camera=cam,camera_id=ORDER.index(cam),source=str(source),source_sha256=expected,
    K_native=cal['camera_intrinsic'],native_hw=[s['height'],s['width']],T_reference_from_cam=T.tolist(),
    timestamp_us=s['timestamp'],sample_data_token=s['token']))
  frame['timestamp_us']=min(timestamps);frame['exposure_delta_us']=max(timestamps)-min(timestamps)
  assert frame['exposure_delta_us']<=50000
  frames[gid]=frame
 if recovered:(OUT/'RECOVERED_INPUTS.json').write_text(json.dumps(recovered,indent=2))
 if missing:
  (OUT/'MISSING_INPUTS.json').write_text(json.dumps(missing,indent=2));raise RuntimeError(f'{len(missing)} inputs missing; see {OUT}/MISSING_INPUTS.json')
 sequences=[]
 for s in selected:
  fs=[frames[g] for g in s['groups']]
  for a,b in zip(fs,fs[1:]):assert 0<b['timestamp_us']-a['timestamp_us']<=1000000,(a['frame_index'],b['frame_index'])
  identity=sorted((a['global_uid'],a['canonical_asset_sha256']) for a in meta[s['groups'][0]]['asset_union'])
  contracts=[]
  for g in s['groups']:
   assert sorted((a['global_uid'],a['canonical_asset_sha256']) for a in meta[g]['asset_union'])==identity
   contracts.append(meta[g]['group_identity']['checkpoint_sha256'])
  assert all(x==contracts[0] for x in contracts)
  sequences.append(dict(scene=s['scene'],asset_identity=identity,render_checkpoints=contracts[0],frames=fs))
 manifest=dict(uid=os.getuid(),selection=report['scenes'],sequences=sequences,total_groups=len(gids),
   network_hw=[576,1024],publish_hw=[288,512],protocol='ar_generated_repeat_oldest_native_rgb_v1',
   historical_comparison_protocol='old baselines warm up T1 for four observations and feed back published 512x288 outputs; double_DiT repeats oldest generated history from observation two at native 1024x576',
   source_metadata=str(metapath),source_metadata_sha256=sha(metapath),sequence_plan=str(planpath),sequence_plan_sha256=sha(planpath))
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(status='READY',groups=len(gids),views=6*len(gids),selected_outputs=30,output=str(OUT))))
if __name__=='__main__':main()
