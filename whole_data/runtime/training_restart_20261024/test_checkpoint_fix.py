import ast, collections, json, os
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist
ROOT=Path('/data/juziwei/hlk/whole_data/runtime/training_restart_20261024')
source=(ROOT/'train_tanhe_checkpoint_fix.py').read_text()
tree=ast.parse(source)
helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='checkpoint_to_cpu')
exec(compile(ast.Module(body=[helper],type_ignores=[]),str(ROOT/'train_tanhe_checkpoint_fix.py'),'exec'))
dist.init_process_group('gloo',timeout=timedelta(seconds=60))
rank=dist.get_rank();world=dist.get_world_size()
model=torch.nn.Linear(3,2);optimizer=torch.optim.AdamW(model.parameters())
model(torch.ones(1,3)).sum().backward();optimizer.step()
rng=[None]*world;dist.all_gather_object(rng,{'rank':rank,'torch':torch.get_rng_state()})
if rank==0:
 payload=checkpoint_to_cpu({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng_by_rank':rng,'step':5})
 assert hasattr(payload['model'],'_metadata')
 path=ROOT/'checkpoint_roundtrip_test.pt'
 with path.open('wb') as stream:
  torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
 loaded=torch.load(path,map_location='cpu',weights_only=False)
 assert loaded['step']==5 and [r['rank'] for r in loaded['rng_by_rank']]==[0,1]
 fresh=torch.nn.Linear(3,2);fresh.load_state_dict(loaded['model'],strict=True)
 opt=torch.optim.AdamW(fresh.parameters());opt.load_state_dict(loaded['optimizer'])
 for a,b in zip(model.parameters(),fresh.parameters()):assert torch.equal(a,b)
 path.unlink()
dist.barrier()
if rank==0:
 (ROOT/'CHECKPOINT_FIX_TEST.json').write_text(json.dumps({'status':'PASS','world_size':world,'tested':['Gloo RNG gather and barrier','CPU checkpoint preserves state dict metadata','model and optimizer checkpoint roundtrip','atomic file flush']},indent=2)+'\n')
 print('CHECKPOINT_FIX_TEST_PASS',flush=True)
dist.destroy_process_group()
