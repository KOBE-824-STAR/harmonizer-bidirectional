"""Verify and restore private Release assets using the standard library only."""
import argparse,hashlib,json,shutil,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def extract(path,dest):
 dest.mkdir(parents=True,exist_ok=True);base=dest.resolve()
 with tarfile.open(path) as archive:
  for member in archive.getmembers():
   target=(base/member.name).resolve()
   if not target.is_relative_to(base) or not (member.isfile() or member.isdir()):raise ValueError('Unsafe archive entry: '+member.name)
  archive.extractall(base)
def main():
 p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--manifest',type=Path,default=ROOT/'release_manifest.json');a=p.parse_args()
 m=json.loads(a.manifest.read_text())
 for item in m['artifacts']:
  target=ROOT/item['target'];target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists():
   if target.stat().st_size!=item['bytes'] or sha(target)!=item['sha256']:raise ValueError('Existing output differs: '+str(target))
  else:
   sources=[]
   for part in item['parts']:
    source=a.assets/part['file']
    if source.stat().st_size!=part['bytes'] or sha(source)!=part['sha256']:raise ValueError('Asset checksum mismatch: '+source.name)
    sources.append(source)
   temp=target.with_suffix(target.suffix+'.restoring')
   with temp.open('xb') as out:
    for source in sources:
     with source.open('rb') as stream:shutil.copyfileobj(stream,out,8<<20)
   if temp.stat().st_size!=item['bytes'] or sha(temp)!=item['sha256']:raise ValueError('Restored checksum mismatch: '+item['name'])
   temp.replace(target)
  if item.get('extract_to'):extract(target,ROOT/item['extract_to'])
  print('VERIFIED',item['name'])
 for item in m['base_weight_files']:
  path=ROOT/'weights'/item['path']
  if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:raise ValueError('Base-weight checksum mismatch: '+item['path'])
 lpips=ROOT/'weights/runtime/tanhe/lpips_vgg_state.pt'
 destination=ROOT/'runtime/tanhe/lpips_vgg_state.pt'
 if destination.exists():
  if sha(destination)!=sha(lpips):raise ValueError('Existing LPIPS weights differ')
 else:shutil.copy2(lpips,destination)
 print('All checkpoint, base weights and manifests restored.')
if __name__=='__main__':main()
