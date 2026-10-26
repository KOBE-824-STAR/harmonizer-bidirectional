"""Restore hash-verified real six-camera inputs and shared-frame calibration."""
import hashlib,json,tarfile
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
ORDER=['CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_FRONT_LEFT']

def transform(r):
    q=np.asarray(r['rotation'],dtype=np.float64);q/=np.linalg.norm(q);w,x,y,z=q
    t=np.eye(4);t[:3,:3]=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    t[:3,3]=r['translation'];return t

def main():
    dst=ROOT/'data/tanhe_sixcam_validation';dst.mkdir(exist_ok=True)
    with tarfile.open(ROOT/'runtime/tanhe/sixcam_data.tar') as tar:
        plan=json.load(tar.extractfile('plan.json'))
        expected={'plan.json',*plan['files']}
        assert {m.name for m in tar.getmembers()}==expected
        for m in tar.getmembers():
            assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
            data=tar.extractfile(m).read()
            if m.name!='plan.json':assert hashlib.sha256(data).hexdigest()==plan['files'][m.name]['sha256']
            p=dst/m.name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    sequences=[]
    for gs in plan['tracks']:
        frames=[]
        for g in gs:
            vv=sorted(g['views'],key=lambda v:ORDER.index(v['camera_name']))
            first=plan['sample_data'][vv[0]['exposure']['sample_data_token']]
            ref=transform(plan['ego_pose'][first['ego_pose_token']])
            views=[]
            for v in vv:
                sd=plan['sample_data'][v['exposure']['sample_data_token']]
                assert sd['timestamp']==v['exposure']['sample_data_timestamp_us']
                cal=plan['calibration'][sd['calibrated_sensor_token']]
                T=np.linalg.inv(ref)@transform(plan['ego_pose'][sd['ego_pose_token']])@transform(cal)
                paths={k:str(dst/rel) for k,rel in v['restored_paths'].items()}
                for path in paths.values():
                    with Image.open(path) as im:assert im.size==(512,288);im.verify()
                views.append(dict(camera=v['camera_name'],camera_id=ORDER.index(v['camera_name']),
                    **paths,K_native=cal['camera_intrinsic'],native_hw=[sd['height'],sd['width']],
                    T_reference_from_cam=T.tolist(),timestamp_us=sd['timestamp'],sample_data_token=sd['token'],
                    no_op=v['no_op'],source_roles={'source':'target' if v['no_op'] else 'input','target':'gt'}))
            times=[v['timestamp_us'] for v in views];assert max(times)-min(times)<=50000
            frames.append(dict(group_id=g['group_id'],frame_index=g['frame_index'],timestamp_us=min(times),
                sample_token=g['sample_token'],asset_union=g['asset_union'],exposure_delta_us=max(times)-min(times),views=views))
        sid=hashlib.sha256(json.dumps([g['group_id'] for g in gs]).encode()).hexdigest()[:24]
        sequences.append(dict(sequence_id=sid,split='train',scene=gs[0]['scene_name'],frames=frames))
    manifest=dst/'sequences.jsonl';manifest.write_text(''.join(json.dumps(s)+'\n' for s in sequences))
    report={'status':'PASS','tracks':len(sequences),'frames':sum(len(s['frames']) for s in sequences),
        'cameras_per_frame':6,'unique_files':len(plan['files']),'all_image_hashes_verified':True,
        'manifest':str(manifest),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'source_groups_sha256':plan['groups_sha256'],'source_index_sha256':plan['source_index_sha256'],
        'scope':plan['scope'],'supervision':plan['supervision'],
        'selected':[{k:s[k] for k in ['sequence_id','scene']}|{'frame_indices':[f['frame_index'] for f in s['frames']],
                    'exposure_spans_us':[f['exposure_delta_us'] for f in s['frames']]} for s in sequences]}
    (ROOT/'reports/tanhe_sixcam_data.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
