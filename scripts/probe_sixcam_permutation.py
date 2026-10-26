import json,os
from pathlib import Path
import torch
from double_dit.model_injection_tanhe import DoubleDiT
from double_dit.data import load_sequences,load_frame,pack_history
root=Path(__file__).resolve().parents[1]
cfg=json.loads((root/'runs/tanhe_one_way/sixcam_v1_single/config.json').read_text())
out=root/'reports/sixcam_permutation_probe';out.mkdir(exist_ok=True)
torch.manual_seed(42);model=DoubleDiT(cfg,out,'cuda');model.eval()
s=load_sequences(cfg['data']['manifest'],'train')[0]
x,t,K,P,ids=load_frame(s['frames'][0],cfg['data']['image_hw'],'cuda');p=torch.arange(5,-1,-1,device='cuda')
traces=[];hooks=[]
for stream in ['main','condition']:
 for i,b in enumerate(getattr(model,stream).blocks):
  def hook(mod,args,result,key=(stream,i)):
   traces.append((key,result.detach().float().cpu()))
  hooks.append(b.register_forward_hook(hook))
with torch.no_grad():
 ref=model(x,pack_history(x,[]),K,P,ids,official_only=True)
 refp=model(x[p],pack_history(x[p],[]),K[p],P[p],ids[p],official_only=True)[p]
 for n,v in model.named_parameters():
  if n.endswith('.gate'):v.fill_(.2)
 traces.clear();y=model(x,pack_history(x,[]),K,P,ids);a=traces[:];edges=model.last_edges
 traces.clear();yp=model(x[p],pack_history(x[p],[]),K[p],P[p],ids[p])[p];b=traces[:];edgesp=model.last_edges
 traces.clear();repeat=model(x,pack_history(x,[]),K,P,ids)
 report={'official_permutation_error':float((ref-refp).abs().max()),'active_permutation_error':float((y-yp).abs().max()),'repeat_error':float((y-repeat).abs().max()),'edges_equivalent':torch.equal(torch.tensor(edges),torch.tensor(edgesp)[:,p.cpu()][:,:,p.cpu()]),'block_errors':[{'key':key,'error':float((aa-bb[p.cpu()]).abs().max())} for (key,aa),(_,bb) in zip(a,b)]}
 (out/'RESULT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
