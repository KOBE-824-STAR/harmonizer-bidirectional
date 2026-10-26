"""Six-camera autoregressive Double DiT inference and paired diagnostics.

The default action is a preflight. --execute runs model inference. Official
FID/FVD/DINO metrics are intentionally not claimed by this interface.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
PROJECT = Path('/data/juziwei/hlk/double_DiT')
PYTHON = PROJECT / 'runtime/tanhe/venv/bin/python'
MODES = ('one_way', 'joint', 'modulation')
ORDER = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
         'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def preflight(args):
    if os.getuid() == 0 or os.getuid() != PROJECT.stat().st_uid:
        raise PermissionError('Run as the Double DiT owner, never as root')
    cfg_path = HERE / 'configs' / f'{args.mode}.json'
    cfg = json.loads(cfg_path.read_text())
    if cfg['exchange']['mode'] != args.mode or cfg.get('inference_enabled') is not True:
        raise ValueError('Mode or inference flag mismatch')
    for rel, expected in cfg['implementation_sha256'].items():
        if sha(PROJECT / rel) != expected:
            raise ValueError(f'Implementation changed since preparation: {rel}')
    eval_path = Path(cfg['evaluation']['manifest'])
    ref_path = Path(cfg['evaluation']['reference_manifest'])
    if sha(eval_path) != cfg['evaluation']['manifest_sha256']:
        raise ValueError('Evaluation sequence manifest changed')
    if sha(ref_path) != cfg['evaluation']['reference_manifest_sha256']:
        raise ValueError('Evaluation reference manifest changed')
    count = 0
    frames = 0
    seen = set()
    for line in eval_path.open():
        seq = json.loads(line)
        if seq['split'] != 'val' or not seq['frames']:
            raise ValueError('Invalid evaluation sequence')
        times = []
        for frame in seq['frames']:
            ids = [v['camera_id'] for v in frame['views']]
            names = [v['camera'] for v in frame['views']]
            if ids != list(range(6)) or names != list(ORDER):
                raise ValueError(f'Not a canonical six-camera frame: {frame["group_id"]}')
            if frame['group_id'] in seen:
                raise ValueError('Duplicate evaluation frame')
            seen.add(frame['group_id'])
            times.append(frame['timestamp_us'])
            frames += 1
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError('Nonchronological evaluation sequence')
        count += 1
    if count != 770 or frames != 5713:
        raise ValueError(f'Unexpected evaluation coverage: {count} sequences, {frames} frames')
    if args.max_sequences is not None and not (1 <= args.max_sequences <= count):
        raise ValueError('Invalid --max-sequences')
    if not (0 <= args.start_sequence < count):
        raise ValueError('Invalid --start-sequence')
    if args.max_sequences is not None and args.start_sequence + args.max_sequences > count:
        raise ValueError('Requested sequence range exceeds evaluation manifest')
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError('Invalid --max-frames')
    if args.initialize_only:
        if args.checkpoint:
            raise ValueError('--initialize-only cannot be combined with --checkpoint')
        if args.max_frames is None or args.max_frames > 2:
            raise ValueError('--initialize-only requires --max-frames 1 or 2')
    elif args.checkpoint is None:
        raise ValueError('Evaluation requires --checkpoint; use --initialize-only for a diagnostic')
    if args.checkpoint and not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.execute and not args.initialize_only:
        integrity_path = HERE / 'processed/DATA_INTEGRITY.json'
        if not integrity_path.is_file():
            raise RuntimeError('Full image SHA256 verification must finish before evaluation')
        integrity = json.loads(integrity_path.read_text())
        if integrity.get('status') != 'PASS' or integrity.get('counts', {}).get('val_groups') != 5880:
            raise RuntimeError('Validation image verification report is incomplete')
        masks_path = HERE / 'processed/MASK_INTEGRITY.json'
        if not masks_path.is_file():
            raise RuntimeError('Evaluation mask SHA256 verification must finish before evaluation')
        masks = json.loads(masks_path.read_text())
        if masks.get('status') != 'PASS' or masks.get('ready_mask_rows') != 5863:
            raise RuntimeError('Evaluation mask verification report is incomplete')
    return cfg, {'sequences': count, 'six_camera_frames': frames,
                 'camera_views': frames * 6,
                 'autoregressive_history': 'own generated RGB only'}


def claim_gpu(gpu: int):
    if gpu < 0:
        raise ValueError('GPU index must be nonnegative')
    (HERE / 'runtime').mkdir(exist_ok=True)
    lock = (HERE / 'runtime' / f'gpu{gpu}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    listed = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True)
    uuids = {int(row.split(',')[0]): row.split(',')[1].strip() for row in listed.splitlines()}
    if gpu not in uuids:
        raise ValueError(f'GPU {gpu} is not present')
    apps = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    if any(row.split(',')[0].strip() == uuids[gpu] for row in apps.splitlines()):
        raise RuntimeError(f'GPU {gpu} has a compute process')
    return lock


def worker(args):
    import numpy as np
    from PIL import Image
    import torch
    from double_dit.data import load_frame, pack_history, quantized_history
    from double_dit.model_injection_tanhe import DoubleDiT

    cfg = json.loads((HERE / 'configs' / f'{args.mode}.json').read_text())
    output = args.output.resolve()
    (output / 'model_audit').mkdir(exist_ok=True)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    model = DoubleDiT(cfg, output / 'model_audit', str(device))
    checkpoint_info = None
    if args.checkpoint:
        # The native trainer saves NumPy RNG state in its checkpoint. Only
        # allow the minimal NumPy types needed by the restricted loader.
        with torch.serialization.safe_globals([
                np.core.multiarray._reconstruct, np.ndarray, np.dtype,
                type(np.dtype('uint32'))]):
            saved = torch.load(args.checkpoint, map_location='cpu',
                               weights_only=True, mmap=True)
        if saved['config']['exchange']['mode'] != args.mode:
            raise ValueError('Checkpoint belongs to a different injection mode')
        if saved['config']['implementation_sha256'] != cfg['implementation_sha256']:
            raise ValueError('Checkpoint implementation differs from current config')
        if saved['data_sha256'] != cfg['data']['manifest_sha256']:
            raise ValueError('Checkpoint was trained on another data manifest')
        model.load_state_dict(saved['model'], strict=True)
        checkpoint_info = {'path': str(args.checkpoint), 'sha256': sha(args.checkpoint),
                           'optimizer_step': saved['step']}
        del saved
    model.eval().requires_grad_(False)
    write_json(output / 'CHECKPOINT.json', checkpoint_info or {
        'status': 'official_initialized_untrained_diagnostic'})
    reference = {}
    for line in Path(cfg['evaluation']['reference_manifest']).open():
        row = json.loads(line)
        reference[(row['group_id'], row['camera'])] = row
    total = Counter()
    by_camera = defaultdict(Counter)
    completed_sequences = 0
    completed_frames = 0
    image_dir = output / 'predictions'
    image_dir.mkdir(exist_ok=True)
    started = time.time()
    with (output / 'predictions_manifest.jsonl').open('w', buffering=1) as journal, \
            torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for sequence_index, line in enumerate(Path(cfg['evaluation']['manifest']).open()):
            if sequence_index < args.start_sequence:
                continue
            if args.max_sequences is not None and completed_sequences >= args.max_sequences:
                break
            seq = json.loads(line)
            history = []
            history_ids = []
            for frame in seq['frames']:
                if args.max_frames is not None and completed_frames >= args.max_frames:
                    break
                src, _, k, poses, ids = load_frame(frame, cfg['data']['evaluation_hw'], device)
                packed = pack_history(src, history)
                pred = model(src, packed, k, poses, ids)
                if pred.shape != src.shape or not torch.isfinite(pred).all():
                    raise RuntimeError(f'Invalid prediction: {frame["group_id"]}')
                quantized = quantized_history(pred)
                for camera, view in enumerate(frame['views']):
                    key = (frame['group_id'], view['camera'])
                    ref = reference[key]
                    pixels = ((pred[camera].float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
                    array = pixels.permute(1, 2, 0).cpu().numpy()
                    target = np.asarray(Image.open(ref['gt']).convert('RGB'))
                    if target.shape != array.shape:
                        raise ValueError(f'Prediction/GT shape mismatch: {key}')
                    delta = array.astype(np.int16) - target.astype(np.int16)
                    sum_abs = int(np.abs(delta).sum())
                    sum_sq = int(np.square(delta.astype(np.int32)).sum())
                    n = int(delta.size)
                    total['absolute_pixel_error'] += sum_abs
                    total['squared_pixel_error'] += sum_sq
                    total['rgb_channel_values'] += n
                    total['views'] += 1
                    by_camera[view['camera']]['absolute_pixel_error'] += sum_abs
                    by_camera[view['camera']]['squared_pixel_error'] += sum_sq
                    by_camera[view['camera']]['rgb_channel_values'] += n
                    by_camera[view['camera']]['views'] += 1
                    path = image_dir / f'{frame["group_id"]}__{view["camera"]}.png'
                    Image.fromarray(array).save(path)
                    journal.write(json.dumps({
                        'sequence_id': seq['sequence_id'], 'group_id': frame['group_id'],
                        'frame_index': frame['frame_index'], 'camera': view['camera'],
                        'T': int(packed.shape[2]), 'history_groups_newest_first': list(reversed(history_ids[-4:])),
                        'prediction': str(path), 'prediction_sha256': sha(path),
                        'source': ref['source'], 'gt': ref['gt'],
                        'mask': ref['mask'], 'mask_status': ref['mask_status'],
                        'diagnostic_mae_8bit': sum_abs / n,
                        'diagnostic_mse_8bit': sum_sq / n,
                    }, ensure_ascii=False) + '\n')
                history.append(quantized)
                history = history[-4:]
                history_ids.append(frame['group_id'])
                history_ids = history_ids[-4:]
                completed_frames += 1
                write_json(output / 'STATUS.json', {
                    'status': 'GENERATING', 'mode': args.mode,
                    'sequences_completed': completed_sequences, 'frames_completed': completed_frames,
                    'views_completed': total['views'], 'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
                    'elapsed_seconds': time.time() - started,
                })
                del pred, packed, src, k, poses, ids
            completed_sequences += 1
            if args.max_frames is not None and completed_frames >= args.max_frames:
                break
    if total['views'] != completed_frames * 6:
        raise RuntimeError('Some camera predictions are missing')
    def aggregate(c):
        return {'views': c['views'], 'diagnostic_mae_8bit':
                c['absolute_pixel_error'] / c['rgb_channel_values'],
                'diagnostic_mse_8bit': c['squared_pixel_error'] / c['rgb_channel_values']}
    result = {
        'status': 'COMPLETE', 'mode': args.mode,
        'scope': 'untrained_initialization_diagnostic' if args.initialize_only else 'checkpoint_inference',
        'official_fid_fvd_dino_computed': False,
        'metric_note': 'MAE/MSE are paired RGB diagnostics, not the official quality protocol',
        'checkpoint': checkpoint_info, 'sequences': completed_sequences,
        'six_camera_frames': completed_frames, 'predicted_views': total['views'],
        'generated_history': True, 'overall': aggregate(total),
        'by_camera': {k: aggregate(v) for k, v in by_camera.items()},
        'predictions_manifest': str(output / 'predictions_manifest.jsonl'),
        'elapsed_seconds': time.time() - started,
        'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
    }
    write_json(output / 'RESULT.json', result)
    write_json(output / 'STATUS.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=MODES)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max-sequences', type=int)
    parser.add_argument('--start-sequence', type=int, default=0,
                        help='zero-based sequence index for a bounded diagnostic')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--initialize-only', action='store_true',
                        help='diagnose prepared inputs with initial official weights')
    parser.add_argument('--execute', action='store_true', help='run GPU inference')
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker:
        worker(args)
        return
    cfg, coverage = preflight(args)
    cmd = [str(PYTHON), str(Path(__file__).resolve()), '--_worker',
           '--mode', args.mode, '--output', str(args.output.resolve()),
           '--gpu', str(args.gpu), '--start-sequence', str(args.start_sequence)]
    if args.checkpoint:
        cmd += ['--checkpoint', str(args.checkpoint.resolve())]
    if args.max_sequences:
        cmd += ['--max-sequences', str(args.max_sequences)]
    if args.max_frames:
        cmd += ['--max-frames', str(args.max_frames)]
    if args.initialize_only:
        cmd += ['--initialize-only']
    plan = {'status': 'READY' if not args.execute else 'STARTING',
            'mode': args.mode, 'coverage': coverage, 'command': cmd,
            'checkpoint': str(args.checkpoint) if args.checkpoint else None,
            'official_quality_metrics_included': False,
            'output': str(args.output.resolve())}
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    lock = claim_gpu(args.gpu)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite evaluation output: {output}')
    output.mkdir(parents=True)
    write_json(output / 'LAUNCH.json', plan)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
               PYTHONPATH=os.pathsep.join((str(PROJECT), str(PROJECT / 'runtime/tanhe/vendor'))),
               PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TORCH_HOME=str(PROJECT / 'runtime/tanhe/torch_cache'),
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    started = time.time()
    with (output / 'evaluate.log').open('w') as log:
        result = subprocess.run(cmd, env=env, cwd=PROJECT,
                                stdout=log, stderr=subprocess.STDOUT)
    plan.update(status='FINISHED' if result.returncode == 0 else 'FAILED',
                started_at=started, finished_at=time.time(), exit_code=result.returncode)
    write_json(output / 'LAUNCH.json', plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    del lock
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
