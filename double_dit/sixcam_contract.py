"""Fail closed before GPU initialization when a final run lacks six views."""
import hashlib
import json
from pathlib import Path


EXPECTED_CAMERA_IDS = set(range(6))


def validate_sixcam_manifest(cfg, *, require_validation_split):
    data = cfg['data']
    if data.get('max_views') != 6 or data.get('allow_variable_views') is not False:
        raise ValueError('Six-camera runs require max_views=6 and allow_variable_views=false')
    path = Path(data['manifest'])
    if not path.is_file():
        raise FileNotFoundError(path)
    recorded_hash = data.get('manifest_sha256')
    if recorded_hash:
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != recorded_hash:
            raise ValueError('Six-camera manifest SHA256 does not match the resolved config')
    splits = {'train': {'sequences': 0, 'frames': 0}, 'val': {'sequences': 0, 'frames': 0}}
    scenes = {'train': set(), 'val': set()}
    train_with_history = 0
    with path.open() as manifest:
        for line_number, line in enumerate(manifest, 1):
            if not line.strip():
                continue
            sequence = json.loads(line)
            split = sequence.get('split')
            if split not in splits:
                raise ValueError(f'Line {line_number}: unexpected split {split!r}')
            frames = sequence.get('frames', [])
            if not frames:
                raise ValueError(f'Line {line_number}: sequence has no frames')
            times = [frame['timestamp_us'] for frame in frames]
            if any(b <= a for a, b in zip(times, times[1:])):
                raise ValueError(f'Line {line_number}: frames are not in increasing time order')
            for frame in frames:
                views = frame.get('views', [])
                ids = [view.get('camera_id') for view in views]
                names = [view.get('camera') for view in views]
                if len(views) != 6 or set(ids) != EXPECTED_CAMERA_IDS or len(set(names)) != 6:
                    raise ValueError(
                        f'Line {line_number}, frame {frame.get("frame_index")}: '
                        f'expected six distinct cameras 0..5, found {ids}'
                    )
                for view in views:
                    for role in ('source', 'target'):
                        file = Path(view[role])
                        if not file.is_file():
                            raise FileNotFoundError(f'Line {line_number}: missing {role} image {file}')
                    if len(view['K_native']) != 3 or any(len(row) != 3 for row in view['K_native']):
                        raise ValueError(f'Line {line_number}: invalid camera intrinsics')
                    pose = view['T_reference_from_cam']
                    if len(pose) != 4 or any(len(row) != 4 for row in pose):
                        raise ValueError(f'Line {line_number}: invalid camera pose')
            splits[split]['sequences'] += 1
            splits[split]['frames'] += len(frames)
            scenes[split].add(sequence['scene'])
            if split == 'train' and len(frames) >= 2:
                train_with_history += 1
    if not splits['train']['sequences'] or not train_with_history:
        raise ValueError('Six-camera training needs train sequences with at least two frames')
    if require_validation_split and not splits['val']['sequences']:
        raise ValueError('Formal six-camera training requires a nonempty val split')
    if scenes['train'] & scenes['val']:
        raise ValueError('Train and val share scene IDs')
    return {
        'manifest': str(path), 'splits': splits,
        'train_sequences_with_history': train_with_history,
        'cameras_per_frame': 6,
        'validation_split_required': require_validation_split,
    }
