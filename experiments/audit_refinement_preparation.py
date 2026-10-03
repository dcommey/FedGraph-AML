"""Independent NumPy audit of all validation searches and frozen choices."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from experiments.summarize_refinements import audit_selection,digest

FAMILIES={'fedavg':'sage_fl','centralized':'sage_central','mlp':'mlp_central','fedmlp':'mlp_fl','fedgcn':'fedgcn'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.root);results=[];count=0
    for dataset in ['elliptic','ibm']:
        first=next((root/f'{dataset}-tune-v2').glob('*/validation_predictions.npz'))
        with np.load(first,allow_pickle=False) as f:step_map={int(i):int(s) for i,s in zip(f['ids'],f['timestep'])}
        for kind in ['','trees-']:
            base=root/f'{dataset}-{kind}tune-v2';m=json.loads((base/'manifest.json').read_text())
            gate=json.loads((root/f'{dataset}-{kind}{"main-" if not kind else ""}gate-v2.json').read_text())
            if m['status']!='completed' or any(r['test'] is not None for r in m['results']):raise AssertionError('Incomplete or contaminated validation search')
            if gate['source_sha256']!=m['source_sha256'] or gate['dataset']!=m['dataset'] or gate['tuning_manifest_sha256']!=digest(base/'manifest.json'):raise AssertionError('Gate provenance differs')
            expected=60 if not kind else 48
            if len(m['results'])!=expected:raise AssertionError('Fixed trial count differs')
            for cell in base.glob('*/result.json'):
                r=json.loads(cell.read_text());audit_selection(cell.parent,r,step_map,dataset);count+=1
            for method in sorted({r['method'] for r in m['results']}):
                candidates=[r for r in m['results'] if r['method']==method]
                if len(candidates)!=12:raise AssertionError('Family search differs')
                selected=max(candidates,key=lambda r:r['selection_ap'])
                key=FAMILIES[method] if not kind else method
                if selected['config']!=gate['selected_configs'][key]:raise AssertionError('Gate did not select earliest validation maximum')
                results.append(dict(dataset=dataset,method=method,trials=len(candidates),selected_config=selected['config'],selection_ap=selected['selection_ap']))
    Path(a.output).write_text(json.dumps(dict(passed=True,audited_trials=count,choices=results,
        numerical_tolerances=dict(metric_absolute=1e-12,coefficient_probability_reconstruction_absolute=2e-6),
        scope='all validation predictions, calibration coefficients/thresholds, checkpoint maxima and gate selections; no test data'),indent=2,allow_nan=False))
    print('PASSED independent preparation audit:',count,'trials')


if __name__=='__main__':main()
