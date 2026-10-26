"""Restore only hash-matched images from an authorized migration archive."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    plan=json.loads((ROOT/'runtime/tanhe/data_plan.json').read_text())
    files=plan['files']
    with tarfile.open(ROOT/'runtime/tanhe/data_restored.tar') as archive:
        members=archive.getmembers()
        if {m.name for m in members}!=set(files):raise RuntimeError('Archive membership mismatch')
        for member in members:
            rel=Path(member.name)
            if not member.isfile() or rel.is_absolute() or '..' in rel.parts:raise RuntimeError('Unexpected archive entry')
            data=archive.extractfile(member).read()
            expected=files[member.name]['sha256']
            if hashlib.sha256(data).hexdigest()!=expected:raise RuntimeError('Archive image hash mismatch')
            dest=ROOT/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():
                if hashlib.sha256(dest.read_bytes()).hexdigest()!=expected:raise RuntimeError('Existing image mismatch')
            else:
                with dest.open('xb') as output:output.write(data)
    for seq in plan['sequences']:
        for frame in seq['frames']:
            for view in frame['views']:
                for key in ('source','target'):
                    if hashlib.sha256(Path(view[key]).read_bytes()).hexdigest()!=view[key+'_sha256']:raise RuntimeError('Restored file checksum mismatch')
    out=ROOT/'data/tanhe_restored/sequences.jsonl'
    content=''.join(json.dumps(s)+'\n' for s in plan['sequences'])
    if out.exists() and out.read_text()!=content:raise RuntimeError('Existing manifest differs')
    out.write_text(content)
    report={'status':'PASS','files':len(files),'sequences':len(plan['sequences']),
        'manifest':str(out),'manifest_sha256':hashlib.sha256(content.encode()).hexdigest(),
        'split_preserved':True,'calibration_preserved':True,'image_sha256_all_verified':True,
        'source_role':'input','supervision_role':'gt','historical_manifest_modified':False}
    (ROOT/'reports/tanhe_data_restored.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
