"""Audit one completed inventory before the full crossed study is available."""
import argparse
import itertools
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from experiments.summarize_refinements import (
    audit_predictions,audit_selection,audit_frozen_recipe,digest,METHODS,TREES,OWNERS,SEEDS)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--dataset',choices=['elliptic','ibm'],required=True)
    parser.add_argument('--kind',choices=['main','trees-main'],default='main')
    parser.add_argument('--neural-root')
    parser.add_argument('--output',required=True)
    a=parser.parse_args();root=Path(a.root)
    m=json.loads((root/'manifest.json').read_text())
    if m['status']!='completed' or m.get('failures'):raise AssertionError('Incomplete inventory')
    gate_path=root.parent/f'{a.dataset}-{"main" if a.kind=="main" else "trees"}-gate-v2.json'
    gate=json.loads(gate_path.read_text())
    if not gate['passed'] or gate['source_sha256']!=m['source_sha256'] or gate['dataset']!=m['dataset']:
        raise AssertionError('Inventory differs from frozen gate')
    expected=set()
    for method in METHODS if a.kind=='main' else TREES:
        pooled=method in ['centralized','mlp','rf_pooled','xgb_pooled']
        for strategy,owner,seed in itertools.product(['random'] if pooled else ['random','louvain'],
            [OWNERS[0]] if pooled else OWNERS,SEEDS):
            expected.add((method,strategy,owner,seed))
    records={(r['method'],r['strategy'],r['ownership_seed'],r['seed']):r for r in m['results']}
    files=sorted(root.glob('*/result.json'))
    if set(records)!=expected or len(records)!=len(m['results']) or len(files)!=len(expected):
        raise AssertionError('Registered inventory differs')
    step_root=Path(a.neural_root) if a.neural_root else root
    with np.load(next(step_root.glob('*/validation_predictions.npz')),allow_pickle=False) as v:
        step_map={int(i):int(s) for i,s in zip(v['ids'],v['timestep'])}
    reference=None;ledger={};rows=[]
    for file in files:
        r=json.loads(file.read_text())
        key=(r['method'],r['strategy'],r['ownership_seed'],r['seed'])
        if r!=records[key]:raise AssertionError('Cell receipt differs from manifest')
        audit_frozen_recipe(r,gate,a.kind)
        data,metrics=audit_predictions(file.parent/'test_predictions.npz',r)
        audit_selection(file.parent,r,step_map,a.dataset)
        support=(data['ids'],data['y'])
        if reference is not None and not all(np.array_equal(x,y) for x,y in zip(reference,support)):
            raise AssertionError('Within-inventory test support differs')
        reference=support
        rows.append(dict(method=r['method'],strategy=r['strategy'],ownership_seed=r['ownership_seed'],seed=r['seed'],**metrics))
        for artifact in file.parent.iterdir():
            if artifact.is_file():ledger[str(artifact.relative_to(root))]=digest(artifact)
    result=dict(passed=True,registered_runs=len(rows),dataset=a.dataset,kind=a.kind,
        scope='This completed inventory only; final crossed study and paired hardware audit remain separate',
        manifest_sha256=digest(root/'manifest.json'),metrics=rows,evidence_sha256=ledger)
    Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(passed=True,registered_runs=len(rows),dataset=a.dataset,kind=a.kind)))


if __name__=='__main__':main()
