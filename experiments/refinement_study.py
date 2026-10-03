"""Versioned follow-up: fixed searches, sealed new test, explicit simulation scope."""
import argparse
import copy
import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from torch_geometric.utils import k_hop_subgraph
from data.refinement_data import prepare, sha256
from experiments.corrected_evaluation import seed_all, make_ownership, build_views, put_view, cpu_states, metric_dict, calibration, digest_tensor, environment
from experiments.refinement_diagnostic import routed_exchange
from models.refined_sage import RefinedSAGE, TabularMLP
from models.fedgcn_reference import FedGCNReference

PROTOCOL = 'fedgraph-refinements-v1-gtx-only-20261002'
METHODS = ['local','fedavg','current','fresh','shuffled','oracle','centralized','mlp','fedmlp','fedgcn']
OWNERSHIP_SEEDS = [20261003,20261004,20261005]
TRAINING_SEEDS = [42,123,456]
SOURCE_PATHS = ['data/refinement_data.py','data/elliptic_loader.py','data/partitioner.py',
    'models/cross_silo_sage.py','models/refined_sage.py','models/fedgcn_reference.py',
    'experiments/corrected_evaluation.py','experiments/refinement_diagnostic.py',
    'experiments/refinement_study.py','tests/test_refinement_mechanism.py',
    'external/FedGCN/src/gnn_models.py']


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False))


def source_hashes():
    return {p:sha256(p) for p in SOURCE_PATHS}


def family(method):
    return {'local':'sage_fl','fedavg':'sage_fl','current':'sage_fl','fresh':'sage_fl',
        'shuffled':'sage_fl','oracle':'sage_fl','centralized':'sage_central',
        'mlp':'mlp_central','fedmlp':'mlp_fl','fedgcn':'fedgcn'}[method]


def neural_grid(method):
    rates = [.05,.2,.5] if method == 'fedgcn' else [.001,.003,.01]
    return [dict(lr=lr,dropout=dropout,reset_adam=reset,hidden=64,weight_decay=.0005)
            for lr,dropout,reset in itertools.product(rates,[0.,.3],[False,True])]


def views_for(data, owner, clients, cutoff, step=None, halo=False):
    views = build_views(data,owner,clients,cutoff,target_step=step)
    if halo:
        # The author code supplies exact two-hop feature halos to each owner.
        # Preserve directed incoming neighborhoods, duplicating context not labels.
        active = data.timestep <= cutoff
        if step is not None and data.no_cross_time_edges:
            active = data.timestep == step
        edge = data.edge_index[:,active[data.edge_index].all(0)]
        for k,v in enumerate(views):
            own_ids = v['ids']
            ids, subedge, _, _ = k_hop_subgraph(own_ids,2,edge,relabel_nodes=True,num_nodes=data.num_nodes)
            own = owner[ids] == k
            v.update(ids=ids,x=data.x[ids],edge_index=subedge,y=data.y[ids],
                     train_mask=data.train_mask[ids]&own, owned_mask=own,
                     remote_src=torch.empty(0,dtype=torch.long),remote_dst=torch.empty(0,dtype=torch.long))
    for v in views:
        v['timestep'] = data.timestep[v['ids']]
    return views


def build_evaluation(data, owner, clients, steps, halo=False):
    return {step:views_for(data,owner,clients,step,step,halo) for step in steps}


def predict(models,data,owner,method,step_views,device,fraction=1.):
    ys,ps,ids_all = [],[],[]
    use_exchange = method in ['current','fresh','shuffled','oracle','stale5']
    mode = 'oracle' if method=='oracle' else ('shuffled' if method=='shuffled' else 'current')
    for step,views in step_views.items():
        remote,raw = [None]*len(models),[None]*len(models)
        if use_exchange:
            remote,raw,_ = routed_exchange(models,views,device,mode,seed=991+step,fraction=fraction)
        for model,v,foreign,first in zip(models,views,remote,raw):
            ids = v['ids']
            mask = (data.timestep[ids]==step)&(data.y[ids]>=0)
            if 'owned_mask' in v:
                mask &= v['owned_mask']
            if not mask.any():
                continue
            model.eval()
            with torch.no_grad():
                p = model(v['x'].to(device),v['edge_index'].to(device),foreign,first).sigmoid().cpu()
            ys.append(data.y[ids[mask]].numpy())
            ps.append(p[mask].numpy())
            ids_all.append(ids[mask].numpy())
    order = np.argsort(np.concatenate(ids_all))
    ids = np.concatenate(ids_all)[order]
    expected = torch.where(torch.isin(data.timestep,torch.tensor(list(step_views)))&(data.y>=0))[0].numpy()
    if not np.array_equal(ids,expected):
        raise AssertionError('Prediction support differs or duplicated halo supervision')
    return np.concatenate(ys)[order],np.concatenate(ps)[order],ids


def operating(y,p,threshold):
    out = metric_dict(y,p,threshold)
    out['brier'] = float(brier_score_loss(y,p))
    out['log_loss'] = float(log_loss(y,np.clip(p,1e-7,1-1e-7),labels=[0,1]))
    out['alert_budgets'] = {}
    # Exact top-k operating points are a declared score ranking, not a test-tuned
    # classification threshold. All denominators include labeled nodes only.
    order = np.argsort(-p,kind='stable')
    for budget in [.01,.05]:
        k = max(1,int(np.ceil(len(y)*budget)))
        positive = int(y[order[:k]].sum())
        out['alert_budgets'][str(budget)] = dict(k=k,precision=positive/k,
            recall=positive/max(int(y.sum()),1),denominator='labeled benchmark nodes')
    edges = np.linspace(0,1,11)
    bins = []
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask = (p>=lo)&((p<=hi) if hi==1 else (p<hi))
        bins.append(dict(lo=float(lo),hi=float(hi),n=int(mask.sum()),
            confidence=float(p[mask].mean()) if mask.any() else None,
            observed=float(y[mask].mean()) if mask.any() else None))
    out['reliability'] = bins
    return out


def calibrated_outputs(data,y,p,ids,selection_steps):
    # Disjoint windows: choose model on earlier validation; calibrate on final
    # validation window. No test outcomes enter fitting or threshold selection.
    select = np.isin(data.timestep[torch.from_numpy(ids)].numpy(),selection_steps)
    c = ~select
    if len(np.unique(y[c])) != 2:
        raise ValueError('Calibration window requires both classes; no reselection')
    z = np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))[:,None]
    fit = LogisticRegression(C=1e6,max_iter=1000).fit(z[c],y[c])
    prob = fit.predict_proba(z)[:,1]
    threshold = calibration(y[c],prob[c])
    return fit,threshold,prob


def run_neural(data,owner,ownership_seed,seed,strategy,method,config,args,out,phase):
    out = Path(out)
    if (out/'result.json').exists():
        return json.loads((out/'result.json').read_text())
    out.mkdir(parents=True,exist_ok=True)
    seed_all(seed)
    central = method in ['centralized','mlp']
    nclients = 1 if central else args.clients
    actual_owner = torch.zeros_like(owner) if central else owner
    halo = method=='fedgcn'
    train_views = views_for(data,actual_owner,nclients,data.train_end,halo=halo)
    val_views = build_evaluation(data,actual_owner,nclients,data.val_steps,halo)
    counts = [int(v['train_mask'].sum()) for v in train_views]
    for v in train_views:
        if torch.unique(v['y'][v['train_mask']]).numel()!=2:
            raise ValueError('Predeclared infeasible ownership: training client lacks a class')
    cls = FedGCNReference if halo else (TabularMLP if method in ['mlp','fedmlp'] else RefinedSAGE)
    base = cls(data.num_features,config['hidden'],config['dropout'])
    initial_hash = digest_tensor(torch.cat([v.flatten() for v in base.state_dict().values()]))
    models = [copy.deepcopy(base).to(args.device) for _ in train_views]
    gpu_views = [put_view(v,args.device) for v in train_views]
    make_optimizer = lambda m: (torch.optim.SGD(m.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
        if halo else torch.optim.Adam(m.parameters(),lr=config['lr'],weight_decay=config['weight_decay']))
    opts = [make_optimizer(m) for m in models]
    use_exchange = method in ['current','fresh','shuffled','oracle','stale5']
    mode = 'oracle' if method=='oracle' else ('shuffled' if method=='shuffled' else 'current')
    history,best,chosen = [],-1,None
    start=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    remote,raw=[None]*nclients,[None]*nclients
    for rnd in range(args.rounds):
        if config['reset_adam']:
            opts=[make_optimizer(m) for m in models]
        comm=dict(unique_sender_bytes=0,receiver_fanout_bytes=0,received_edges=0,oracle_raw_fanout_bytes=0)
        for epoch in range(args.local_epochs):
            refresh = use_exchange and (epoch==0 or method in ['fresh','oracle'])
            if method=='stale5':
                refresh=epoch==0 and rnd%5==0
            if refresh:
                remote,raw,c = routed_exchange(models,gpu_views,args.device,mode,seed=seed+100003*rnd,fraction=args.fraction)
                for key in comm:
                    comm[key]+=c[key]
            for m,v,o,foreign,first in zip(models,gpu_views,opts,remote,raw):
                m.train();o.zero_grad()
                y=v['y'][v['train_mask']].float()
                weight=(y==0).sum()/(y==1).sum()
                loss=F.binary_cross_entropy_with_logits(m(v['x'],v['edge_index'],foreign,first)[v['train_mask']],y,pos_weight=weight)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss')
                loss.backward();o.step()
        if not central and method!='local':
            states=cpu_states(models)
            weights = [1/nclients]*nclients if halo else [n/sum(counts) for n in counts]
            aggregate={key:sum(s[key]*w for s,w in zip(states,weights)) for key in states[0]}
            for m in models:
                m.load_state_dict(aggregate)
        entry=dict(round=rnd+1,**comm)
        if (rnd+1)%args.evaluate_every==0 or rnd+1==args.rounds:
            y,p,ids=predict(models,data,actual_owner,method,val_views,args.device,args.fraction)
            selection = np.isin(data.timestep[torch.from_numpy(ids)].numpy(),data.val_steps[:-1])
            ap=float(average_precision_score(y[selection],p[selection]))
            entry['selection_ap']=ap
            if ap>best:
                best,chosen,best_states,best_round=ap,(y.copy(),p.copy(),ids.copy()),cpu_states(models),rnd+1
            print(f'{phase} {args.dataset} {strategy} owner={ownership_seed} s={seed} {method} r={rnd+1} select_AP={ap:.5f}',flush=True)
        history.append(entry)
    for m,state in zip(models,best_states):
        m.load_state_dict(state)
    y,p,ids=chosen
    fit,threshold,cal_p=calibrated_outputs(data,y,p,ids,data.val_steps[:-1])
    np.savez_compressed(out/'validation_predictions.npz',y=y,probability=p,calibrated_probability=cal_p,ids=ids,timestep=data.timestep[torch.from_numpy(ids)].numpy())
    torch.save(dict(states=best_states,selected_round=best_round,config=config,method=method,
        calibration_coef=fit.coef_,calibration_intercept=fit.intercept_,threshold=threshold),out/'checkpoint.pt')
    result=dict(protocol=PROTOCOL,method=method,dataset=args.dataset,strategy=strategy,ownership_seed=ownership_seed,
        seed=seed,config=config,selected_round=best_round,selection_ap=best,initial_state_hash=initial_hash,
        ownership_hash=digest_tensor(owner),validation=operating(y,cal_p,threshold),
        calibration=dict(coef=fit.coef_.tolist(),intercept=fit.intercept_.tolist(),fit_steps=[data.val_steps[-1]]),
        history=history,test=None,seconds=time.perf_counter()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        parameters=sum(p.numel() for p in base.parameters()),
        boundary_bytes=sum(h['receiver_fanout_bytes']+h['oracle_raw_fanout_bytes'] for h in history),
        model_update_bytes=0 if central or method=='local' else args.rounds*nclients*2*sum(p.numel()*p.element_size() for p in base.parameters()),
        halo_feature_bytes=sum(int((~v['owned_mask']).sum())*data.num_features*4 for v in train_views) if halo else 0,
        timing_scope='train and validation; excludes ownership, preprocessing and final test')
    if phase=='main':
        y,p,ids=predict(models,data,actual_owner,method,build_evaluation(data,actual_owner,nclients,data.test_steps,halo),args.device,args.fraction)
        z=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))[:,None]
        cal_p=fit.predict_proba(z)[:,1]
        result['test']=operating(y,cal_p,threshold)
        c=np.isin(data.timestep[torch.from_numpy(chosen[2])].numpy(),[data.val_steps[-1]])
        result['test_uncalibrated']=operating(y,p,calibration(chosen[0][c],chosen[1][c]))
        result['primary_test_ap']=float(average_precision_score(y,p))
        np.savez_compressed(out/'test_predictions.npz',y=y,probability=p,calibrated_probability=cal_p,ids=ids,timestep=data.timestep[torch.from_numpy(ids)].numpy(),owner=owner[torch.from_numpy(ids)].numpy())
    write_json(out/'result.json',result)
    del models,gpu_views,train_views,val_views,base,opts
    torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['pilot','tune','main'],default='pilot')
    p.add_argument('--dataset',choices=['elliptic','ibm'],required=True)
    p.add_argument('--data-root',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--gate')
    p.add_argument('--methods',nargs='+',default=METHODS)
    p.add_argument('--strategies',nargs='+',default=['random','louvain'])
    p.add_argument('--ownership-seeds',nargs='+',type=int,default=OWNERSHIP_SEEDS)
    p.add_argument('--seeds',nargs='+',type=int,default=TRAINING_SEEDS)
    p.add_argument('--rounds',type=int,default=50)
    p.add_argument('--local-epochs',type=int,default=2)
    p.add_argument('--evaluate-every',type=int,default=5)
    p.add_argument('--clients',type=int,default=3)
    p.add_argument('--fraction',type=float,default=1.)
    a=p.parse_args();a.device='cuda'
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    sources=source_hashes()
    data,meta=prepare(a.dataset,a.data_root,root)
    manifest=dict(protocol=PROTOCOL,config=vars(a),source_sha256=sources,dataset=meta,environment=environment(),results=[],failures=[],status='running')
    write_json(root/'manifest.json',manifest)
    gate=None
    if a.phase=='main':
        if not a.gate:
            raise RuntimeError('New independent test requires a frozen passed pilot+tuning gate')
        gate=json.loads(Path(a.gate).read_text())
        if not gate.get('passed') or gate['source_sha256']!=sources or gate['dataset']!=meta:
            raise RuntimeError('Gate source/dataset/preprocessing differs')
        for key in ['rounds','local_epochs','evaluate_every','clients','methods','strategies','ownership_seeds','seeds','fraction']:
            if getattr(a,key)!=gate['main_config'][key]:
                raise RuntimeError(f'Gate configuration differs: {key}')
    for strategy in a.strategies:
        for ownership_seed in a.ownership_seeds:
            owner=make_ownership(data,strategy,ownership_seed,a.clients)
            np.save(root/f'ownership-{strategy}-{ownership_seed}.npy',owner.numpy())
            if gate is not None and digest_tensor(owner)!=gate['ownership_sha256'][f'{strategy}-{ownership_seed}']:
                raise RuntimeError('Ownership differs from frozen gate')
            for seed in a.seeds:
                for method in a.methods:
                    if method in ['centralized','mlp'] and (strategy!=a.strategies[0] or ownership_seed!=a.ownership_seeds[0]):
                        continue
                    configs=neural_grid(method) if a.phase=='tune' else [gate['selected_configs'][family(method)] if gate else neural_grid(method)[4]]
                    for i,config in enumerate(configs):
                        label=f'{strategy}-o{ownership_seed}-s{seed}-{method}'+(f'-c{i}' if a.phase=='tune' else '')
                        try:
                            result=run_neural(data,owner,ownership_seed,seed,strategy,method,config,a,root/label,a.phase)
                            manifest['results'].append(result)
                        except Exception as exc:
                            manifest['failures'].append(dict(cell=label,error=repr(exc)))
                            write_json(root/'manifest.json',manifest)
                            raise
                        write_json(root/'manifest.json',manifest)
    manifest['status']='completed';write_json(root/'manifest.json',manifest)


if __name__=='__main__':
    main()
