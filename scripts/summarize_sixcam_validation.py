"""Aggregate successful bounded runs; fail closed on missing evidence."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--single-tag',default='sixcam_v2');p.add_argument('--ddp-tag',default='sixcam_v4');a=p.parse_args()
    modes={}
    for mode in ['one_way','joint','modulation']:
        modes[mode]={}
        for kind,tag,world in [('single',a.single_tag,1),('ddp4',a.ddp_tag,4)]:
            out=ROOT/f'runs/tanhe_{mode}/{tag}_{kind}'
            result=read(out/'RESULT.json');launch=read(out/'LAUNCH.json');cfg=read(out/'config.json');mem=read(out/'device_memory.json')
            assert result['status']==launch['state']=='PASS' and launch['exit_code']==0
            assert result['cameras_per_frame']==6 and result['world_size']==world
            assert result['generated_history_verified'] and result['diagnostic_optimizer_steps']==4
            assert cfg['training_enabled'] is False
            assert cfg['data']['manifest_sha256']==sha(Path(cfg['data']['manifest']))
            # Model mathematics must be exactly the implementation actually tested.
            for rel,digest in cfg['implementation_sha256'].items():
                if rel.startswith('double_dit/') or rel.endswith('validate_sixcam.py'):
                    assert sha(ROOT/rel)==digest,rel
            official=[]
            for rank in range(world):
                loaded=read(out/f'rank{rank}/official_load.json')
                assert loaded['numel_coverage']==1.0
                assert not any(loaded[k] for k in ['missing_core','shape_mismatch','load_missing','load_unexpected'])
                official.append({k:loaded[k] for k in ['matched_parameter_tensors','matched_parameter_numel','numel_coverage']})
            if world==1:assert result['permutation_max_error']==0
            else:
                assert result['all_step_gradients_identical_across_ranks'] and result['final_parameters_identical_across_ranks']
            modes[mode][kind]={'directory':str(out),'result':result,'official_load_by_rank':official,
                'peak_device_used_mib':mem['peak_device_used_mib'],'device_memory_sample_errors':mem['errors'],
                'launch':launch,'model_code_sha256':{k:v for k,v in cfg['implementation_sha256'].items() if k.startswith('double_dit/')}}
    original=read(ROOT/'reports/injection_install_20261021.json')['original_source_preserved']
    changed=[p for p,h in original.items() if sha(ROOT/p)!=h];assert not changed
    checkpoint=ROOT/'runs/train/latest.pt';assert checkpoint.stat().st_size==8057053230
    report={'status':'PASS','scope':'512x288_real_six_camera_single_gpu_and_4_gpu_ddp_bounded_diagnostics',
        'data':read(ROOT/'reports/tanhe_sixcam_data.json'),'modes':modes,'original_sources_changed':changed,
        'original_checkpoint':{'path':str(checkpoint),'bytes':checkpoint.stat().st_size,'not_written_by_validation':True},
        'formal_training_started':False,'geometry_run':False,'strictly_synchronized_data':False,
        'failed_attempts_retained':['sixcam_v1_single (camera ordering)','sixcam_v2_ddp4 (NCCL host allocation)','sixcam_v3_ddp4 (P2P disabled, same NCCL failure)'],
        'cpu_tests_log':'reports/tanhe_sixcam_cpu_tests.log'}
    path=ROOT/'reports/tanhe_sixcam_validation_summary.json';path.write_text(json.dumps(report,indent=2))
    for mode,kind in modes.items():
        print(mode,{k:{'allocated_GiB':max(x['peak_allocated_bytes'] for x in v['result']['memory_by_rank'])/2**30,
                       'reserved_GiB':max(x['peak_reserved_bytes'] for x in v['result']['memory_by_rank'])/2**30,
                       'driver_GiB':max(v['peak_device_used_mib'].values())/1024} for k,v in kind.items()})
    print(path)
if __name__=='__main__':main()
