"""Freeze all new test choices after a passed, validation-only search."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from data.refinement_data import prepare,sha256
from experiments.refinement_study import (PROTOCOL,METHODS,OWNERSHIP_SEEDS,TRAINING_SEEDS,
    source_hashes,family,write_json)
from experiments.corrected_evaluation import make_ownership,digest_tensor,environment


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',required=True)
    p.add_argument('--data-root',required=True)
    p.add_argument('--pilot',required=True)
    p.add_argument('--tuning',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    pilot=json.loads((Path(a.pilot)/'manifest.json').read_text())
    tuning=json.loads((Path(a.tuning)/'manifest.json').read_text())
    if pilot['status']!='completed' or tuning['status']!='completed' or tuning['failures']:
        raise RuntimeError('Incomplete/failed pilot or tuning')
    for m in [pilot,tuning]:
        if m['source_sha256']!=source_hashes():
            raise RuntimeError('Numerical sources changed after the recorded pilot/search')
        if any(r.get('test') is not None for r in m['results']):
            raise RuntimeError('New study test already accessed in preparatory runs')
        if m['environment']['dependencies']!=environment()['dependencies']:
            raise RuntimeError('Numerical dependencies changed')
    if max(r['peak_cuda_bytes'] for r in pilot['results'])>environment()['gpu_memory_bytes']*.85:
        raise RuntimeError('Insufficient pilot memory margin')
    selected={}
    for method in ['fedavg','centralized','mlp','fedmlp','fedgcn']:
        rows=[r for r in tuning['results'] if r['method']==method]
        if len(rows)!=12 or len({json.dumps(r['config'],sort_keys=True) for r in rows})!=12:
            raise RuntimeError('Fixed twelve-trial family search incomplete')
        # Python max keeps the earliest trial on an exact tie.
        selected[family(method)]=max(rows,key=lambda r:r['selection_ap'])['config']
    root=Path(a.output).parent;root.mkdir(parents=True,exist_ok=True)
    data,meta=prepare(a.dataset,a.data_root,root)
    if meta!=pilot['dataset'] or meta!=tuning['dataset']:
        raise RuntimeError('Raw/preprocessing/task adapter changed')
    hashes={}
    support={}
    for strategy in ['random','louvain']:
        for seed in OWNERSHIP_SEEDS:
            owner=make_ownership(data,strategy,seed,3)
            name=f'{strategy}-{seed}'
            hashes[name]=digest_tensor(owner)
            support[name]={}
            for split in ['train','val']:
                mask=getattr(data,f'{split}_mask')
                counts=[dict(positive=int((mask&(owner==k)&(data.y==1)).sum()),
                             negative=int((mask&(owner==k)&(data.y==0)).sum())) for k in range(3)]
                support[name][split]=counts
                if split=='train' and any(c['positive']==0 or c['negative']==0 for c in counts):
                    raise RuntimeError('Predeclared infeasible ownership; preserve failure, do not substitute')
    gate=dict(protocol=PROTOCOL,passed=True,source_sha256=source_hashes(),dataset=meta,
        selected_configs=selected,ownership_sha256=hashes,pilot_support=support,
        environment=environment(),pilot_manifest_sha256=sha256(Path(a.pilot)/'manifest.json'),
        tuning_manifest_sha256=sha256(Path(a.tuning)/'manifest.json'),
        test_status_at_freeze='not evaluated in this follow-up; Elliptic previously exposed',
        primary='IBM random fresh minus FedAvg raw score average precision',
        inference='average training seeds within each of three ownership draws; mean/range, no population t-test',
        main_config=dict(rounds=50,local_epochs=2,evaluate_every=5,clients=3,methods=METHODS,
                         strategies=['random','louvain'],ownership_seeds=OWNERSHIP_SEEDS,seeds=TRAINING_SEEDS,fraction=1.))
    write_json(a.output,gate)
    print('FROZEN',a.output,selected,flush=True)


if __name__=='__main__':
    main()
