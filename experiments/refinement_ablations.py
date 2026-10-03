"""Fixed exploratory Elliptic availability, cache-age and dropout interventions."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data.refinement_data import prepare,sha256
from experiments.corrected_evaluation import make_ownership
from experiments.refinement_study import run_neural,write_json,source_hashes
import experiments.refinement_study as runner
from experiments.ablations_routing import routed_exchange


def main():
    p=argparse.ArgumentParser();p.add_argument('--main-root',required=True);p.add_argument('--gate',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    gate=json.loads(Path(a.gate).read_text())
    if gate['source_sha256']!=source_hashes():raise RuntimeError('Ablation source changed')
    # Explicit numerical adapter in this ablation process only; all fraction1
    # controls retain the frozen main behavior. Record its own source hashes.
    runner.routed_exchange=routed_exchange
    data,meta=prepare('elliptic','data/elliptic',root)
    args=argparse.Namespace(clients=3,device='cuda',rounds=50,local_epochs=2,evaluate_every=5,dataset='elliptic',fraction=1.)
    config=gate['selected_configs']['sage_fl']
    manifest=dict(dataset=meta,scope='exploratory; previously exposed Elliptic test',results=[],source_sha256=source_hashes(),
        intervention_source_sha256={p:sha256(p) for p in ['experiments/refinement_ablations.py','experiments/ablations_routing.py']},
        availability_rule='sourceID multiplicative integer hash modulo2147483647, seed20261006; independent of snapshot size/GPU/model RNG')
    for strategy in ['random','louvain']:
        for oseed in [20261003,20261004,20261005]:
            owner=make_ownership(data,strategy,oseed,3)
            for fraction in [0.,.25,.5]:
                args.fraction=fraction
                r=run_neural(data,owner,oseed,42,strategy,'current',config,args,root/f'{strategy}-o{oseed}-coverage{fraction}','main')
                r['intervention']=dict(type='source_availability',fraction=fraction,availability_seed=20261006)
                if fraction==0:
                    reference=json.loads((Path(a.main_root)/f'{strategy}-o{oseed}-s42-fedavg/result.json').read_text())
                    if abs(r['primary_test_ap']-reference['primary_test_ap'])>1e-8:
                        raise AssertionError('Zero exchange differs from paired FedAvg')
                manifest['results'].append(r);write_json(root/'manifest.json',manifest)
            args.fraction=1.
            r=run_neural(data,owner,oseed,42,strategy,'stale5',config,args,root/f'{strategy}-o{oseed}-stale5','main')
            r['intervention']=dict(type='cache_refresh',round_interval=5,inference='fresh selected-checkpoint snapshot')
            for h in r['history']:
                h['cache_created_round']=1+5*((h['round']-1)//5)
                h['cache_age_rounds']=h['round']-h['cache_created_round']
                h['source_optimizer_step']=2*(h['cache_created_round']-1)
            manifest['results'].append(r);write_json(root/'manifest.json',manifest)
            for method in ['current','fresh']:
                c=dict(config,dropout=0.)
                r=run_neural(data,owner,oseed,42,strategy,method,c,args,root/f'{strategy}-o{oseed}-{method}-dropout0','main')
                r['intervention']=dict(type='dropout',dropout=0.)
                manifest['results'].append(r);write_json(root/'manifest.json',manifest)
    manifest['status']='completed';write_json(root/'manifest.json',manifest)


if __name__=='__main__':main()
