"""Read-only preflight for the migrated native runtime."""
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runtime/tanhe/vendor')]


def main():
    import torch,transformer_engine,importlib.metadata as metadata
    from double_dit.model_injection_tanhe import DoubleDiT
    report={'uid':os.getuid(),'gid':os.getgid(),'root':str(ROOT),
        'python':sys.version,'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
        'transformer_engine':transformer_engine.__version__,
        'lpips':metadata.version('lpips'),'real_model_import':'PASS',
        'data':json.loads((ROOT/'reports/tanhe_data_restored.json').read_text()),
        'weights':json.loads((ROOT/'reports/tanhe_weights_verified.json').read_text())}
    original=json.loads((ROOT/'reports/injection_install_20261021.json').read_text())['original_source_preserved']
    report['original_source_changed']=[p for p,h in original.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    origin=json.loads((ROOT/'runtime/tanhe/vendor_origin_sha256.json').read_text())
    report['vendor_changed']=[p for p,h in origin.items() if hashlib.sha256((ROOT/'runtime/tanhe/vendor'/p).read_bytes()).hexdigest()!=h]
    assert not report['original_source_changed']
    assert report['vendor_changed']==['imaginaire/auxiliary/text_encoder.py'],report['vendor_changed']
    report['status']='PASS';print(json.dumps(report,indent=2))

if __name__=='__main__':main()
