"""Verify all asset masks referenced by the frozen 5,713-group evaluation set."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time


HERE = Path(__file__).resolve().parent
DATA = Path('/data/public_data/harmonizer_512_20261019')
OLD = '/mnt/ojc/workplace2/asset_mask_inventory/nusc_pair_insert_masks'


def sha(path: Path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    if os.getuid() == 0 or os.getuid() != HERE.stat().st_uid:
        raise PermissionError('Run as whole_data owner, never as root')
    cache = {}
    ready = 0
    empty = 0
    started = time.time()
    for line in (DATA / 'manifests/eval_5713.jsonl').open():
        row = json.loads(line)
        if row['mask_status'] == 'READY':
            ready += 1
            source = row['mask_path']
            if not source or not source.startswith(OLD + '/'):
                raise ValueError(f'Invalid asset mask source: {row["sample_key"]}')
            path = Path(source.replace(OLD, str(DATA / 'masks/nusc_pair_insert_masks'), 1))
            if path not in cache:
                cache[path] = sha(path)
            if cache[path] != row['mask_sha256']:
                raise ValueError(f'Asset mask SHA256 mismatch: {path}')
        elif row['mask_status'] == 'EMPTY_NO_ASSET_VIEW':
            empty += 1
            if row['mask_path'] is not None:
                raise ValueError('No-asset view has a mask path')
        else:
            raise ValueError(f'Unexpected mask status: {row["mask_status"]}')
    if (ready, empty) != (5863, 28415):
        raise ValueError(f'Evaluation mask counts changed: {(ready, empty)}')
    result = {'status': 'PASS', 'ready_mask_rows': ready,
              'empty_mask_rows': empty, 'unique_masks_hashed': len(cache),
              'elapsed_seconds': time.time() - started,
              'training_started': False}
    path = HERE / 'processed/MASK_INTEGRITY.json'
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(result, indent=2) + '\n')
    tmp.replace(path)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
