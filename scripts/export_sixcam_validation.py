"""Export four real two-frame, six-camera diagnostic tracks; no generation."""
import collections,gzip,hashlib,json,os,tarfile
from pathlib import Path

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


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert os.getuid()==1005
 root=Path('/mnt/hlk/workspace/double_DiT/runs/tanhe_sixcam_export_20261021')
 root.mkdir(exist_ok=True)
 base=Path('/mnt/ojc/workplace2_harmonization_archive_20261020T135832Z/pipeline_manifests/sixcam_production_20260913/index_verified_lwh/train')
 tracks=collections.defaultdict(list)
 with gzip.open(base/'groups.jsonl.gz','rt') as f:
  for l in f:
   g=json.loads(l)
   ts=[v['exposure']['sample_data_timestamp_us'] for v in g['views']]
   if g['status']=='ready' and max(ts)-min(ts)<=50000:
    key=(g['context_id'],tuple(sorted(a['global_uid'] for a in g['asset_union'])))
    tracks[key].append(g)
 source={}
 with gzip.open(base/'source_index.jsonl.gz','rt') as f:
  for l in f:
   r=json.loads(l);source[r['sample_id']]=r
 chosen=[];files={};used_contexts=set()
 for key,gs in tracks.items():
  if key[0] in used_contexts:continue
  gs=sorted(gs,key=lambda g:g['frame_index'])
  if len(gs)<2:continue
  pair=gs[:2]
  if min(v['exposure']['sample_data_timestamp_us'] for v in pair[1]['views'])-min(v['exposure']['sample_data_timestamp_us'] for v in pair[0]['views'])>1000000:continue
  pending={};resolved=[]
  try:
   for g in pair:
    views=[]
    for v in g['views']:
     sid=v['baseline_source_sample_id'] if v['no_op'] else v['pair_source_sample_id']
     r=source[sid]
     assert r['exposure']==v['exposure']
     # A no-op uses the rendered, unedited baseline (source role target).
     # Supervision remains role gt, exactly as in the existing DoubleDiT recipe.
     roles={'source':'target' if v['no_op'] else 'input','target':'gt'}
     paths={}
     for field,role in roles.items():
      original=Path(r['paths'][role])
      p=Path(r['production_record']).parents[2]/'dataset'/role/original.name
      assert p.is_file(),str(p)
      expected=r['content_sha256'][role];assert sha(p)==expected,str(p)
      rel='images/'+role+'/'+p.name
      pending[rel]={'path':str(p),'sha256':expected,'bytes':p.stat().st_size}
      paths[field]=rel
     views.append(dict(v,source_record=r,restored_paths=paths))
    resolved.append(dict(g,views=views))
  except (KeyError,AssertionError) as e:
   print('skip',key,str(e),flush=True);continue
  chosen.append(resolved);files.update(pending);used_contexts.add(key[0])
  if len(chosen)==4:break
 assert len(chosen)==4,len(chosen)
 tokens={v['exposure']['sample_data_token'] for gs in chosen for g in gs for v in g['views']}
 nusc=Path('/mnt/public_data/nuScenes/v1.0-trainval')
 sd={r['token']:r for r in json_items(nusc/'sample_data.json') if r['token'] in tokens}
 assert set(sd)==tokens
 poses_needed={r['ego_pose_token'] for r in sd.values()}
 poses={r['token']:r for r in json_items(nusc/'ego_pose.json') if r['token'] in poses_needed}
 calibration={r['token']:r for r in json_items(nusc/'calibrated_sensor.json')}
 plan={'tracks':chosen,'files':files,'sample_data':sd,'ego_pose':poses,'calibration':calibration,'source_index_sha256':sha(base/'source_index.jsonl.gz'),'groups_sha256':sha(base/'groups.jsonl.gz'),'scope':'engineering_validation_only_asynchronous_real_camera_exposures','supervision':'gt (original DoubleDiT recipe); no-op input is rendered target baseline'}
 (root/'plan.json').write_text(json.dumps(plan,indent=2))
 with tarfile.open(root/'sixcam_data.tar','w') as tar:
  tar.add(root/'plan.json',arcname='plan.json')
  for rel,r in files.items():tar.add(r['path'],arcname=rel)
 print(json.dumps({'output':str(root),'tracks':len(chosen),'images':len(files),'archive_bytes':(root/'sixcam_data.tar').stat().st_size,'frames':[[g['frame_index'] for g in gs] for gs in chosen]}),flush=True)
if __name__=='__main__':main()
