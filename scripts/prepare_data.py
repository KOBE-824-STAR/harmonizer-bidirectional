"""Audit exact-identity multi-camera pairs; never invent missing views."""
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NUSC = Path('/mnt/public_data/nuScenes/v1.0-trainval')
ORDER = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']


def json_items(path):
    """Stream a top-level JSON array with bounded memory using stdlib."""
    dec = json.JSONDecoder()
    with open(path) as f:
        buf = ''; eof = False; begun = False
        while True:
            if not eof:
                chunk = f.read(4 * 1024 * 1024)
                eof = not chunk
                buf += chunk
            pos = 0
            while True:
                while pos < len(buf) and buf[pos].isspace(): pos += 1
                if not begun:
                    if pos == len(buf): break
                    assert buf[pos] == '['
                    pos += 1; begun = True
                    continue
                if pos < len(buf) and buf[pos] == ',': pos += 1; continue
                if pos < len(buf) and buf[pos] == ']': return
                if pos == len(buf): break
                try: value, end = dec.raw_decode(buf, pos)
                except json.JSONDecodeError:
                    if eof: raise
                    break
                yield value
                pos = end
            buf = buf[pos:]
            if eof: raise ValueError('Truncated JSON array')


def transform(row):
    q = np.asarray(row['rotation'], dtype=np.float64)
    q /= np.linalg.norm(q)
    w,x,y,z = q
    T=np.eye(4)
    T[:3,:3]=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
               [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
               [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    T[:3,3]=row['translation']
    return T


def main():
    assert os.getuid() == os.getgid() == 1005
    groups=collections.defaultdict(list)
    with open('/mnt/hlk/workspace/nusc_pair_grouped6/pair_lineage.csv') as f:
        for r in csv.DictReader(f):
            key=tuple(r[k] for k in ('split','scene_token','sample_token','combination_id','frame_index'))
            groups[key].append(r)
    candidates=[rows for rows in groups.values() if 2 <= len(rows) <= 6 and len({r['camera_name'] for r in rows})==len(rows)]
    needed=collections.defaultdict(set)
    for rows in candidates:
        for r in rows:
            matches=re.findall(r'(/\S+\.jsonl):(\d+)',r['lineage_source'])
            assert len(matches)>=1, r['lineage_source']
            path,line=matches[-1]; r['_metadata_ref']=(path,int(line))
            needed[path].add(int(line))
    metadata={}
    for path,indices in needed.items():
        with open(path) as f:
            for i,line in enumerate(f,1):
                if i in indices: metadata[(path,i)]=json.loads(line)
                if i>=max(indices): break
    print('Metadata loaded',len(metadata),flush=True)
    tokens={r['sample_data_token'] for rows in candidates for r in rows}
    sd={r['token']:r for r in json_items(NUSC/'sample_data.json') if r['token'] in tokens}
    assert set(sd)==tokens
    pose_tokens={r['ego_pose_token'] for r in sd.values()}
    poses={r['token']:r for r in json_items(NUSC/'ego_pose.json') if r['token'] in pose_tokens}
    calibration={r['token']:r for r in json_items(NUSC/'calibrated_sensor.json')}
    print('Calibration loaded',len(sd),flush=True)
    accepted=[]; rejects=collections.Counter(); details=[]; exposure_deltas=[]
    for rows in candidates:
        rows=sorted(rows,key=lambda r:ORDER.index(r['camera_name']))
        try:
            signatures=[]
            for r in rows:
                m=metadata[r['_metadata_ref']]
                assert m['sample_id']==Path(r['input_path']).stem, 'metadata_sample_mismatch'
                assert str(m['frame_index'])==r['frame_index'], 'metadata_frame_mismatch'
                assets=sorted((a['instance_token'],a['canonical_asset_sha256']) for a in m['selected_exact_assets'])
                assert assets, 'missing_assets'
                contract=m['pair_contract']
                assert contract['raw_nuscenes_pixels_in_pair'] is False, 'unknown_pair_domain'
                signatures.append(json.dumps([assets,contract],sort_keys=True))
            assert len(set(signatures))==1, 'asset_or_render_contract_mismatch'
            timestamps=[int(r['timestamp']) for r in rows]
            dt=max(timestamps)-min(timestamps); exposure_deltas.append(dt)
            assert dt <= 50000, 'exposure_gap_gt_50ms'
            reference_pose=transform(poses[sd[rows[0]['sample_data_token']]['ego_pose_token']])
            views=[]
            for r in rows:
                s=sd[r['sample_data_token']]
                assert s['timestamp']==int(r['timestamp']), 'timestamp_mismatch'
                cal=calibration[s['calibrated_sensor_token']]
                camera_pose=np.linalg.inv(reference_pose) @ transform(poses[s['ego_pose_token']]) @ transform(cal)
                for key in ('input_path','gt_path'):
                    with Image.open(r[key]) as im:
                        assert im.size==(512,288), 'unexpected_image_size'
                        im.verify()
                K=np.asarray(cal['camera_intrinsic'],dtype=np.float64)
                # Store native intrinsics; loader applies resize from native sensor pixels.
                views.append({'camera':r['camera_name'],'camera_id':ORDER.index(r['camera_name']),
                  'source':r['input_path'],'target':r['gt_path'],'timestamp_us':s['timestamp'],
                  'sample_data_token':s['token'],'K_native':K.tolist(),
                  'native_hw':[s['height'],s['width']], 'T_reference_from_cam':camera_pose.tolist(),
                  'metadata_ref':list(r['_metadata_ref'])})
            signature=hashlib.sha256(signatures[0].encode()).hexdigest()
            accepted.append({'split':rows[0]['split'],'scene':rows[0]['scene_token'],
              'combination_id':rows[0]['combination_id'],'frame_index':int(rows[0]['frame_index']),
              'sample_token':rows[0]['sample_token'],'timestamp_us':min(timestamps),
              'asset_render_signature':signature,'exposure_delta_us':dt,'views':views})
        except (AssertionError,KeyError,FileNotFoundError,OSError) as e:
            reason=str(e); rejects[reason]+=1
            if len(details)<20: details.append({'sample':rows[0]['input_path'],'reason':reason})
    tracks=collections.defaultdict(list)
    for g in accepted:
        key=(g['split'],g['scene'],g['combination_id'],g['asset_render_signature'],tuple(v['camera'] for v in g['views']))
        tracks[key].append(g)
    sequences=[]; gaps=[]
    for key,frames in tracks.items():
        frames.sort(key=lambda f:f['timestamp_us'])
        chunks=[[]]
        for f in frames:
            if chunks[-1]:
                gap=f['timestamp_us']-chunks[-1][-1]['timestamp_us']; gaps.append(gap)
                assert gap>0
                if gap>1000000: chunks.append([])
            chunks[-1].append(f)
        for i,chunk in enumerate(chunks):
            sequences.append({'sequence_id':hashlib.sha256(repr((key,i)).encode()).hexdigest()[:24],
                 'split':key[0],'scene':key[1],'frames':chunk})
    scenes={s:{q['scene'] for q in sequences if q['split']==s} for s in {q['split'] for q in sequences}}
    split_names=list(scenes)
    for i,s in enumerate(split_names):
        for other in split_names[i+1:]: assert not scenes[s]&scenes[other], 'scene_split_leakage'
    out=ROOT/'runs'/'data_v1';out.mkdir(exist_ok=True)
    manifest=out/'sequences.jsonl'
    manifest.write_text(''.join(json.dumps(s)+'\n' for s in sequences))
    report={'uid':os.getuid(),'candidate_groups':len(candidates),'accepted_groups':len(accepted),
      'rejections':dict(rejects),'rejection_examples':details,
      'sequences_by_split':dict(collections.Counter(s['split'] for s in sequences)),
      'length_histogram':dict(collections.Counter(len(s['frames']) for s in sequences)),
      'multiframe_sequences_by_split':dict(collections.Counter(s['split'] for s in sequences if len(s['frames'])>1)),
      'max_exposure_delta_us':50000,'max_history_gap_us':1000000,
      'observed_exposure_delta_quantiles_us':np.quantile(exposure_deltas,[0,.5,.95,1]).tolist() if exposure_deltas else [],
      'observed_history_gap_quantiles_us':np.quantile(gaps,[0,.5,.95,1]).tolist() if gaps else [],
      'manifest':str(manifest),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
      'status':'READY_FOR_GEOMETRY_AND_MODEL_CHECK' if accepted else 'NO_VALID_GROUPS'}
    (ROOT/'reports'/'data_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
