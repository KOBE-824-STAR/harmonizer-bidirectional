"""Proceed to training only after a successful native four-GPU short run."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
assert os.getuid()==os.getgid()==1005
native=json.loads((ROOT/'runs/native/RESULT.json').read_text())
assert native['status']=='PASS'
for phase in ['ddp_smoke','train']:
    status={'stage':phase,'status':'RUNNING','launcher_uid':os.getuid(),'time_unix':time.time(),
      'gpus':[0,1,2,3],'tmux_socket':'double_dit','tmux_session':'train_0_3'}
    (ROOT/'runs/PIPELINE_STATUS.json').write_text(json.dumps(status,indent=2))
    result=subprocess.run([sys.executable,str(ROOT/'scripts/run_experiment.py'),phase])
    if result.returncode:
        status.update(status='FAILED',exit_code=result.returncode)
        (ROOT/'runs/PIPELINE_STATUS.json').write_text(json.dumps(status,indent=2))
        raise SystemExit(result.returncode)
status.update(status='COMPLETE')
(ROOT/'runs/PIPELINE_STATUS.json').write_text(json.dumps(status,indent=2))
