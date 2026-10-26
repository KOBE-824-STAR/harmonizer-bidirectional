"""Read-only migration preflight. Does not start Docker, GPU jobs or training."""
import argparse
import json
import os
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def inspect(mode,manifest=None):
    manifest=Path(manifest or ROOT/'runs/data_v1/sequences.jsonl')
    sources=Path('/mnt/hlk/workspace/Scheme_A&Scheme_B')
    recipe=Path('/mnt/hlk/workspace/harmonizer_official_recipe_20261012')
    missing=[]
    if os.getuid()!=1005 or os.getgid()!=1005:missing.append('inherited_model_requires_uid_gid_1005')
    if not shutil.which('docker'):missing.append('docker_not_available')
    if not sources.is_dir():missing.append('multiview_harmonizer_source_not_migrated')
    if not (recipe/'src').is_dir():missing.append('recipe_source_not_migrated')
    for p in (recipe/'weights/diffusion_harmonizer.pkl',recipe/'weights/cosmos/model.pt',recipe/'weights/cosmos/tokenizer/tokenizer.pth'):
        if not p.is_file():missing.append('unresolved_weight_path:'+str(p))
    counts={'sequences':0,'views':0,'missing_source':0,'missing_target':0,'missing_geometry_depth':0}
    if not manifest.is_file():missing.append('manifest_missing')
    else:
        for line in manifest.read_text().splitlines():
            if not line.strip():continue
            sequence=json.loads(line);counts['sequences']+=1
            for frame in sequence['frames']:
                for view in frame['views']:
                    counts['views']+=1
                    for key in ('source','target'):
                        if not view.get(key) or not Path(view[key]).is_file():counts['missing_'+key]+=1
                    info=view.get('geometry_depth',{})
                    if not info.get('path') or not Path(info['path']).is_file():counts['missing_geometry_depth']+=1
        for key in ('missing_source','missing_target'):
            if counts[key]:missing.append(key)
        if mode=='geometry' and counts['missing_geometry_depth']:missing.append('missing_geometry_depth')
    return {'status':'BLOCKED_RUNTIME' if missing else 'PATHS_READY_RUNTIME_STILL_UNVALIDATED',
            'mode':mode,'root':str(ROOT),'uid':os.getuid(),'gid':os.getgid(),'manifest':str(manifest),
            'counts':counts,'missing':missing,
            'scope':'Filesystem preflight only; does not validate Docker image, CUDA, weight hashes or model execution.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',default='one_way',choices=['bidirectional','one_way','joint','modulation','geometry'])
    p.add_argument('--manifest');args=p.parse_args()
    result=inspect(args.mode,args.manifest);print(json.dumps(result,indent=2))
    raise SystemExit(2 if result['missing'] else 0)

if __name__=='__main__':main()
