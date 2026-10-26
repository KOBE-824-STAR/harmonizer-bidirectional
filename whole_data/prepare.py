"""Prepare the published 512x288 six-camera data for Double DiT.

This reads the immutable public bundle and nuScenes metadata. It writes only
under this script's directory; it never copies or edits source images.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = Path('/data/juziwei/hlk/double_DiT')
DATA = Path('/data/public_data/harmonizer_512_20261019')
NUSCENES = Path('/data/public_data/nuScenes_trainval')
ORDER = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
         'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT')
PUBLISHED_IDS = {'CAM_FRONT': '0', 'CAM_FRONT_RIGHT': '2',
                 'CAM_BACK_RIGHT': '4', 'CAM_BACK': '5',
                 'CAM_BACK_LEFT': '3', 'CAM_FRONT_LEFT': '1'}
EXPECTED_META_SHA = {
    'sample.json': '6035ac58b6e971622be2bb1be15b917e7cb4e05ae984d6b339c27c1699c4ad9d',
    'sample_data.json': '6dcad49f0b9bd7b1cef04e2a0ff2ac2879b46b938e8d48153f00693cebb4de21',
    'calibrated_sensor.json': '67781a5dd7b2504b046ef89d6dcb267b12d1cc91af21f1ec5758624588e99865',
    'ego_pose.json': 'be12bd501f694b344628ba0680a37d6e1ec83e62b37f310c205c1dbe41cd09ff',
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def write_jsonl(path: Path, values: list[dict]) -> str:
    tmp = path.with_suffix(path.suffix + '.tmp')
    digest = hashlib.sha256()
    with tmp.open('wb') as f:
        for value in values:
            row = (json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
            f.write(row)
            digest.update(row)
    tmp.replace(path)
    return digest.hexdigest()


def transform(record: dict) -> list[list[float]]:
    q = [float(x) for x in record['rotation']]
    n = math.sqrt(sum(x * x for x in q))
    if not n or not math.isfinite(n):
        raise ValueError('Invalid pose quaternion')
    w, x, y, z = [v / n for v in q]
    r = [
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ]
    t = [float(v) for v in record['translation']]
    return [r[i] + [t[i]] for i in range(3)] + [[0., 0., 0., 1.]]


def rigid_inverse(m: list[list[float]]) -> list[list[float]]:
    r = [[m[j][i] for j in range(3)] for i in range(3)]
    return [r[i] + [-sum(r[i][j] * m[j][3] for j in range(3))] for i in range(3)] + [[0., 0., 0., 1.]]


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def read_groups(data_root: Path):
    groups = {'train': [], 'val': []}
    needed_sd = set()
    metadata_hashes = {}
    cameras_with_asset = Counter()
    exposure_over_50ms = Counter()
    for split, rel in (
        ('train', 'dataset/train_snapshot/metadata/train_groups.jsonl'),
        ('val', 'dataset/published/metadata/val_groups.jsonl'),
    ):
        meta = data_root / rel
        image_dir = data_root / ('dataset/train_snapshot/train' if split == 'train' else 'dataset/published/val')
        h = hashlib.sha256()
        seen = set()
        with meta.open('rb') as f:
            for line in f:
                h.update(line)
                row = json.loads(line)
                gid = row['group_id']
                if row['split'] != split or row['review']['status'] != 'accepted' or gid in seen:
                    raise ValueError(f'Invalid or duplicate accepted {split} group: {gid}')
                seen.add(gid)
                by_name = {v['camera_name']: v for v in row['views']}
                if set(by_name) != set(ORDER) or len(row['views']) != 6:
                    raise ValueError(f'Incomplete six-camera group: {gid}')
                views = []
                for name in ORDER:
                    v = by_name[name]
                    if str(v['camera_id']) != PUBLISHED_IDS[name]:
                        raise ValueError(f'Unexpected published camera ID for {name}')
                    role_hashes = {role: v['role_sources'][role]['content_sha256']
                                   for role in ('input', 'target', 'gt')}
                    if bool(v['no_op']) != (role_hashes['input'] == role_hashes['target']):
                        raise ValueError(f'Input/target role mismatch: {gid} {name}')
                    paths = {role: image_dir / f'{gid}__{name}__{role}.png'
                             for role in ('input', 'target', 'gt')}
                    if any(not p.is_file() for p in paths.values()):
                        raise FileNotFoundError(f'Missing image role: {gid} {name}')
                    exposure = v['exposure']
                    token = exposure['sample_data_token']
                    needed_sd.add(token)
                    views.append({
                        'camera': name, 'camera_id': ORDER.index(name),
                        'sample_data_token': token,
                        'timestamp_us': int(exposure['sample_data_timestamp_us']),
                        'source': str(paths['input']), 'target': str(paths['gt']),
                        'render_target': str(paths['target']),
                        'source_sha256': role_hashes['input'],
                        'target_sha256': role_hashes['gt'],
                        'no_op': bool(v['no_op']),
                    })
                times = [v['timestamp_us'] for v in views]
                if max(times) - min(times) > 50000:
                    exposure_over_50ms[split] += 1
                cameras_with_asset[(split, sum(not v['no_op'] for v in views))] += 1
                identity = tuple(sorted((a['global_uid'], a['canonical_asset_sha256'])
                                        for a in row['asset_union']))
                if not identity:
                    raise ValueError(f'No asset identity: {gid}')
                groups[split].append({
                    'group_id': gid, 'scene': row['scene_name'],
                    'context_id': row['context_id'], 'frame_index': int(row['frame_index']),
                    'sample_token': row['sample_token'], 'asset_identity': identity,
                    'timestamp_us': min(times), 'exposure_delta_us': max(times) - min(times),
                    'views': views,
                })
        metadata_hashes[split] = h.hexdigest()
    return groups, needed_sd, metadata_hashes, cameras_with_asset, exposure_over_50ms


def load_calibration(nuscenes_root: Path, tokens: set[str]):
    version = nuscenes_root / 'v1.0-trainval'
    for name, expected in EXPECTED_META_SHA.items():
        path = version / name
        if sha(path) != expected:
            raise ValueError(f'nuScenes metadata checksum mismatch: {path}')
    with (version / 'sample_data.json').open() as f:
        all_sd = json.load(f)
    sd = {x['token']: x for x in all_sd if x['token'] in tokens}
    del all_sd
    gc.collect()
    if set(sd) != tokens:
        raise ValueError(f'Missing {len(tokens - set(sd))} sample_data records')
    pose_tokens = {x['ego_pose_token'] for x in sd.values()}
    cal_tokens = {x['calibrated_sensor_token'] for x in sd.values()}
    with (version / 'ego_pose.json').open() as f:
        all_poses = json.load(f)
    poses = {x['token']: transform(x) for x in all_poses if x['token'] in pose_tokens}
    del all_poses
    gc.collect()
    with (version / 'calibrated_sensor.json').open() as f:
        all_cal = json.load(f)
    cal = {x['token']: x for x in all_cal if x['token'] in cal_tokens}
    if set(poses) != pose_tokens or set(cal) != cal_tokens:
        raise ValueError('Missing ego pose or calibrated sensor records')
    return sd, poses, cal


def enrich(groups: dict, sd: dict, poses: dict, cal: dict) -> dict:
    output = {'train': {}, 'val': {}}
    for split in ('train', 'val'):
        for g in groups[split]:
            front = sd[g['views'][0]['sample_data_token']]
            ref_inverse = rigid_inverse(poses[front['ego_pose_token']])
            views = []
            for v in g['views']:
                sample = sd[v['sample_data_token']]
                if sample['timestamp'] != v['timestamp_us']:
                    raise ValueError(f'Exposure timestamp mismatch: {g["group_id"]}')
                calibration = cal[sample['calibrated_sensor_token']]
                k = calibration['camera_intrinsic']
                if len(k) != 3 or any(len(row) != 3 for row in k) or k[0][0] <= 0 or k[1][1] <= 0:
                    raise ValueError(f'Invalid intrinsic calibration: {g["group_id"]}')
                pose = matmul(matmul(ref_inverse, poses[sample['ego_pose_token']]),
                              transform(calibration))
                views.append({
                    **v, 'K_native': k, 'native_hw': [sample['height'], sample['width']],
                    'T_reference_from_cam': pose,
                    'source_roles': {'source': 'input', 'target': 'gt'},
                })
            frame = {
                'group_id': g['group_id'], 'frame_index': g['frame_index'],
                'timestamp_us': g['timestamp_us'], 'sample_token': g['sample_token'],
                'context_id': g['context_id'], 'asset_identity': g['asset_identity'],
                'exposure_delta_us': g['exposure_delta_us'], 'views': views,
            }
            output[split][g['group_id']] = (g['scene'], g['asset_identity'], frame)
    return output


def make_sequences(prepared: dict) -> tuple[list[dict], dict]:
    sequences = []
    stats = {}
    for split in ('train', 'val'):
        by_identity = defaultdict(list)
        for _, (scene, identity, frame) in prepared[split].items():
            by_identity[(scene, identity)].append(frame)
        count = 0
        multi = 0
        for (scene, identity), frames in sorted(by_identity.items(), key=lambda x: (x[0][0], x[0][1])):
            frames.sort(key=lambda x: (x['timestamp_us'], x['frame_index'], x['group_id']))
            if any(b['timestamp_us'] <= a['timestamp_us'] for a, b in zip(frames, frames[1:])):
                raise ValueError(f'Nonmonotonic frame timestamps: {scene} {identity}')
            # The native trainer consumes at most the first five frames of each
            # train sequence. Nonoverlapping chunks cover every frame once.
            chunks = [frames[i:i+5] for i in range(0, len(frames), 5)] if split == 'train' else [frames]
            for chunk in chunks:
                ids = [f['group_id'] for f in chunk]
                sid = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()[:24]
                sequences.append({'sequence_id': sid, 'split': split, 'scene': scene,
                                  'asset_identity': identity, 'frames': chunk})
                count += 1
                multi += len(chunk) >= 2
        stats[split] = {'sequences': count, 'frames': len(prepared[split]),
                        'multi_frame_sequences': multi, 'asset_identities': len(by_identity)}
    return sequences, stats


def make_evaluation(prepared: dict, data_root: Path):
    selection = {}
    old_data = '/mnt/ojc/workplace2/dataset/nusc_pair_6cam'
    old_masks = '/mnt/ojc/workplace2/asset_mask_inventory/nusc_pair_insert_masks'
    for line in (data_root / 'manifests/eval_5713.jsonl').open():
        row = json.loads(line)
        key = (row['group_id'], row['camera_id'])
        if key in selection:
            raise ValueError(f'Duplicate evaluation view: {key}')
        if row['group_id'] not in prepared['val']:
            raise ValueError(f'Evaluation group absent from published images: {key}')
        expected = prepared['val'][row['group_id']][2]['views'][ORDER.index(row['camera_id'])]
        if (row['input_path'].replace(old_data, str(data_root / 'dataset/published'), 1)
                != expected['source']):
            raise ValueError(f'Evaluation source mismatch: {key}')
        if (row['gt_path'].replace(old_data, str(data_root / 'dataset/published'), 1)
                != expected['target']):
            raise ValueError(f'Evaluation GT mismatch: {key}')
        mask = row.get('mask_path')
        if mask:
            mask = mask.replace(old_masks, str(data_root / 'masks/nusc_pair_insert_masks'), 1)
            if not Path(mask).is_file():
                raise FileNotFoundError(mask)
        selection[key] = {
            'group_id': row['group_id'], 'camera': row['camera_id'],
            'source': expected['source'], 'gt': expected['target'],
            'mask': mask, 'mask_status': row['mask_status'],
            'source_sha256': expected['source_sha256'],
            'gt_sha256': expected['target_sha256'],
        }
    by_group = defaultdict(set)
    for gid, camera in selection:
        by_group[gid].add(camera)
    if len(by_group) != 5713 or any(c != set(ORDER) for c in by_group.values()):
        raise ValueError('Official evaluation selection is not 5713 complete six-camera groups')
    plan = json.loads((data_root / 'manifests/autoregressive_sequences.json').read_text())
    sequences = []
    covered = set()
    for item in plan['sequences']:
        gids = [gid for gid in item['groups'] if gid in by_group]
        if not gids:
            continue
        frames = [prepared['val'][gid][2] for gid in gids]
        if any(b['timestamp_us'] <= a['timestamp_us'] for a, b in zip(frames, frames[1:])):
            raise ValueError('Evaluation sequence is not chronological')
        if any(gid in covered for gid in gids):
            raise ValueError('Duplicate evaluation group in sequence plan')
        covered.update(gids)
        sid = hashlib.sha256(json.dumps(gids, separators=(',', ':')).encode()).hexdigest()[:24]
        sequences.append({'sequence_id': sid, 'split': 'val', 'scene': item['scene'],
                          'asset_identity': item['asset_identity'], 'frames': frames})
    if covered != set(by_group):
        raise ValueError(f'Evaluation plan misses {len(set(by_group) - covered)} groups')
    return sequences, list(selection.values())


def make_configs(out: Path, manifest_sha: str, eval_sha: str, ref_sha: str):
    cfg_dir = HERE / 'configs'
    cfg_dir.mkdir(exist_ok=True)
    for mode in ('one_way', 'joint', 'modulation'):
        base = PROJECT / f'runs/tanhe_{mode}/sixcam_v4_ddp4/config.json'
        cfg = json.loads(base.read_text())
        if cfg['exchange']['mode'] != mode:
            raise ValueError(f'Base configuration mode mismatch: {mode}')
        cfg['status'] = 'prepared_sixcam_train_eval'
        cfg['training_enabled'] = True
        cfg['inference_enabled'] = True
        cfg['data'].update({
            'manifest': str(out / 'sequences.jsonl'), 'manifest_sha256': manifest_sha,
            'max_views': 6, 'allow_variable_views': False,
            'source_dataset': str(DATA), 'source_role': 'input', 'supervision_role': 'gt',
        })
        cfg['evaluation'].update({
            'manifest': str(out / 'eval_5713_sequences.jsonl'),
            'manifest_sha256': eval_sha,
            'reference_manifest': str(out / 'eval_5713_reference.jsonl'),
            'reference_manifest_sha256': ref_sha,
            'autoregressive_generated_history': True,
        })
        cfg.pop('validation', None)
        cfg['runtime'].update({'gpus': [0, 1, 2, 3], 'data_bundle': str(DATA)})
        for relative in list(cfg['implementation_sha256']):
            path = PROJECT / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            cfg['implementation_sha256'][relative] = sha(path)
        cfg['implementation_sha256']['double_dit/sixcam_contract.py'] = sha(PROJECT / 'double_dit/sixcam_contract.py')
        cfg['whole_data_interface_sha256'] = {p.name: sha(p) for p in HERE.glob('*.py')}
        write_json(cfg_dir / f'{mode}.json', cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DATA)
    parser.add_argument('--nuscenes-root', type=Path, default=NUSCENES)
    parser.add_argument('--output', type=Path, default=HERE / 'processed')
    parser.add_argument('--replace', action='store_true', help='replace an existing prepared manifest')
    args = parser.parse_args()
    if os.getuid() == 0 or os.getuid() != HERE.stat().st_uid:
        raise PermissionError('Run as the owner of whole_data, never as root')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / 'sequences.jsonl'
    if manifest.exists() and not args.replace:
        raise FileExistsError(f'{manifest} exists; pass --replace for an intentional rebuild')
    groups, tokens, source_hashes, active, slow = read_groups(args.data_root)
    print('GROUPS', json.dumps({k: len(v) for k, v in groups.items()}), flush=True)
    sd, poses, cal = load_calibration(args.nuscenes_root, tokens)
    print('CALIBRATION', json.dumps({'sample_data': len(sd), 'ego_poses': len(poses),
                                     'calibrated_sensors': len(cal)}), flush=True)
    prepared = enrich(groups, sd, poses, cal)
    del groups, sd, poses, cal
    gc.collect()
    sequences, stats = make_sequences(prepared)
    eval_sequences, references = make_evaluation(prepared, args.data_root)
    main_sha = write_jsonl(manifest, sequences)
    eval_sha = write_jsonl(out / 'eval_5713_sequences.jsonl', eval_sequences)
    ref_sha = write_jsonl(out / 'eval_5713_reference.jsonl', references)
    report = {
        'status': 'PREPARED', 'training_started': False, 'source_data': str(args.data_root),
        'nuscenes_metadata': str(args.nuscenes_root / 'v1.0-trainval'),
        'source_group_metadata_sha256': source_hashes,
        'manifest': str(manifest), 'manifest_sha256': main_sha,
        'eval_manifest_sha256': eval_sha, 'eval_reference_sha256': ref_sha,
        'training_and_validation': stats,
        'eval_sequences': len(eval_sequences), 'eval_frames': sum(len(s['frames']) for s in eval_sequences),
        'eval_reference_views': len(references),
        'active_camera_counts': {f'{split}_{views}': n for (split, views), n in active.items()},
        'exposure_over_50ms_groups': dict(slow),
        'camera_order': ORDER, 'source_role': 'input', 'supervision_role': 'gt',
        'geometry': 'all cameras transformed to the CAM_FRONT exposure ego frame',
        'history_protocol': 'generated model outputs only; no original RGB history',
    }
    write_json(out / 'DATA_REPORT.json', report)
    make_configs(out, main_sha, eval_sha, ref_sha)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
