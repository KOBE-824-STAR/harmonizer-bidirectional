"""Verify every published six-camera PNG against its provenance SHA256."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import struct
import time


HERE = Path(__file__).resolve().parent
DATA = Path('/data/public_data/harmonizer_512_20261019')
ORDER = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
         'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT')


def inspect(path: Path) -> tuple[str, tuple[int, int]]:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        header = f.read(24)
        if header[:8] != b'\x89PNG\r\n\x1a\n':
            raise ValueError(f'Not a PNG: {path}')
        wh = struct.unpack('>II', header[16:24])
        if wh != (512, 288):
            raise ValueError(f'Unexpected image size {wh}: {path}')
        digest.update(header)
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest(), wh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DATA)
    args = parser.parse_args()
    if os.getuid() == 0 or os.getuid() != HERE.stat().st_uid:
        raise PermissionError('Run as the whole_data owner, never as root')
    started = time.time()
    cache = {}
    counts = Counter()
    for split, meta_rel, image_rel in (
        ('train', 'dataset/train_snapshot/metadata/train_groups.jsonl', 'dataset/train_snapshot/train'),
        ('val', 'dataset/published/metadata/val_groups.jsonl', 'dataset/published/val'),
    ):
        image_dir = args.data_root / image_rel
        for line in (args.data_root / meta_rel).open():
            group = json.loads(line)
            if group['split'] != split or group['review']['status'] != 'accepted':
                raise ValueError(f'Invalid accepted group: {group["group_id"]}')
            views = {v['camera_name']: v for v in group['views']}
            if set(views) != set(ORDER) or len(group['views']) != 6:
                raise ValueError(f'Not six cameras: {group["group_id"]}')
            counts[f'{split}_groups'] += 1
            for camera in ORDER:
                view = views[camera]
                for role in ('input', 'target', 'gt'):
                    path = image_dir / f'{group["group_id"]}__{camera}__{role}.png'
                    st = path.stat()
                    key = (st.st_dev, st.st_ino)
                    if key not in cache:
                        cache[key] = inspect(path)[0]
                        counts['unique_payload_files'] += 1
                        counts['unique_payload_bytes'] += st.st_size
                    if cache[key] != view['role_sources'][role]['content_sha256']:
                        raise ValueError(f'SHA256 mismatch: {path}')
                    counts[f'{split}_role_files'] += 1
            if counts[f'{split}_groups'] % 5000 == 0:
                print(split, counts[f'{split}_groups'], 'groups verified', flush=True)
    result = {'status': 'PASS', 'image_size_wh': [512, 288],
              'counts': dict(counts), 'elapsed_seconds': time.time() - started,
              'source_root': str(args.data_root), 'training_started': False}
    path = HERE / 'processed/DATA_INTEGRITY.json'
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(result, indent=2) + '\n')
    tmp.replace(path)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
