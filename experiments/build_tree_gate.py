"""Freeze the complete twelve-trial tree searches before final test access."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data.refinement_data import sha256
from experiments.refinement_study import write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tuning',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();path=Path(a.tuning)/'manifest.json';m=json.loads(path.read_text())
    if m['status']!='completed' or m['phase']!='tune' or any(r['test'] is not None for r in m['results']):
        raise RuntimeError('Tree search incomplete or test already inspected')
    if {p:sha256(p) for p in m['source_sha256']}!=m['source_sha256']:
        raise RuntimeError('Tree search source changed')
    configs={}
    for method in ['rf_pooled','rf_local','xgb_pooled','xgb_local']:
        rows=[r for r in m['results'] if r['method']==method]
        if len(rows)!=12 or len({json.dumps(r['config'],sort_keys=True) for r in rows})!=12:
            raise RuntimeError('Incomplete family search')
        configs[method]=max(rows,key=lambda r:r['selection_ap'])['config']
    write_json(a.output,dict(passed=True,source_sha256=m['source_sha256'],dataset=m['dataset'],
        selected_configs=configs,tuning_manifest_sha256=sha256(path),
        main_inventory='pooled3training seeds; local2strategies×3owners×3training seeds; GTX only'))
    print('FROZEN TREE GATE',a.output,flush=True)


if __name__=='__main__':main()
