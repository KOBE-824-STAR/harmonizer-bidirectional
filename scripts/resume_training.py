"""Resume the current run with an explicit total optimizer-step budget."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

def main():
    assert os.getuid() == os.getgid() == 1005
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error('--steps must be positive')
    status = {'stage': 'train', 'status': 'RESUMING', 'launcher_uid': os.getuid(),
              'time_unix': time.time(), 'gpus': [0, 1, 2, 3],
              'tmux_socket': 'double_dit', 'tmux_session': 'train_0_3',
              'total_steps': args.steps, 'resume': str(ROOT / 'runs/train/latest.pt')}
    path = ROOT / 'runs/PIPELINE_STATUS.json'
    path.write_text(json.dumps(status, indent=2))
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/run_experiment.py'),
                             'train', '--steps', str(args.steps), '--resume', status['resume']])
    status.update(status='COMPLETE' if result.returncode == 0 else 'FAILED',
                  exit_code=result.returncode)
    path.write_text(json.dumps(status, indent=2))
    raise SystemExit(result.returncode)

if __name__ == '__main__':
    main()
