"""Pooled/local RF and XGBoost with a fixed validation-only search."""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from data.refinement_data import prepare,sha256
from experiments.corrected_evaluation import make_ownership,digest_tensor
from experiments.refinement_study import write_json,operating,calibrated_outputs


def grid(kind):
    if kind=='rf':
        return [dict(n_estimators=200,max_depth=d,min_samples_leaf=l,class_weight=w)
            for d,l,w in itertools.product([6,12,None],[1,10],[None,'balanced'])]
    return [dict(n_estimators=200,max_depth=d,learning_rate=lr,weighted=w)
        for d,lr,w in itertools.product([3,6,10],[.03,.1],[False,True])]


def fit_predict(data,owner,kind,local,seed,config):
    begin=time.perf_counter()
    val_ids=torch.where(data.val_mask)[0].numpy()
    p=np.zeros(len(val_ids))
    models=[]
    for k in range(3 if local else 1):
        train=data.train_mask & ((owner==k) if local else torch.ones_like(data.train_mask))
        ids=torch.where(train)[0].numpy()
        y=data.y[ids].numpy()
        if len(np.unique(y))!=2:
            raise ValueError('Infeasible training client; no tree resampling')
        if kind=='rf':
            model=RandomForestClassifier(**config,n_jobs=4,random_state=seed)
        else:
            from xgboost import XGBClassifier
            c=config.copy();weighted=c.pop('weighted')
            model=XGBClassifier(**c,scale_pos_weight=float((y==0).sum()/(y==1).sum()) if weighted else 1.,
                device='cuda',tree_method='hist',n_jobs=4,random_state=seed,eval_metric='logloss')
        model.fit(data.x[ids].numpy(),y)
        select=(owner[torch.from_numpy(val_ids)].numpy()==k) if local else np.ones(len(val_ids),dtype=bool)
        p[select]=model.predict_proba(data.x[val_ids[select]].numpy())[:,1]
        models.append(model)
    return models,p,val_ids,time.perf_counter()-begin


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--dataset',required=True)
    a.add_argument('--data-root',required=True)
    a.add_argument('--output',required=True)
    a.add_argument('--phase',choices=['tune','main'],default='tune')
    a.add_argument('--gate')
    args=a.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    data,meta=prepare(args.dataset,args.data_root,root)
    source={p:sha256(p) for p in ['experiments/refinement_trees.py','data/refinement_data.py','experiments/refinement_study.py']}
    manifest=dict(dataset=meta,source_sha256=source,phase=args.phase,results=[],status='running')
    gate=None
    if args.phase=='main':
        if not args.gate:
            raise RuntimeError('Tree test requires a frozen selection gate')
        gate=json.loads(Path(args.gate).read_text())
        if not gate['passed'] or gate['source_sha256']!=source or gate['dataset']!=meta:
            raise RuntimeError('Tree gate mismatch')
    for kind,local in itertools.product(['rf','xgb'],[False,True]):
        method=kind+('_local' if local else '_pooled')
        seeds=[711] if args.phase=='tune' else [42,123,456]
        ownerships=[20261003] if args.phase=='tune' or not local else [20261003,20261004,20261005]
        strategies=['random'] if args.phase=='tune' or not local else ['random','louvain']
        for strategy in strategies:
            for ownership_seed in ownerships:
                owner=make_ownership(data,strategy,ownership_seed,3)
                for seed in seeds:
                    configs=grid(kind) if gate is None else [gate['selected_configs'][method]]
                    for i,c in enumerate(configs):
                        name=f'{method}-{strategy}-o{ownership_seed}-s{seed}-c{i}'
                        out=root/name
                        if (out/'result.json').exists():
                            manifest['results'].append(json.loads((out/'result.json').read_text()));continue
                        out.mkdir(exist_ok=True)
                        models,p,ids,seconds=fit_predict(data,owner,kind,local,seed,c)
                        y=data.y[ids].numpy()
                        select=np.isin(data.timestep[torch.from_numpy(ids)].numpy(),data.val_steps[:-1])
                        fit,threshold,cp=calibrated_outputs(data,y,p,ids,data.val_steps[:-1])
                        result=dict(method=method,strategy=strategy,ownership_seed=ownership_seed,seed=seed,
                            ownership_hash=digest_tensor(owner),config=c,selection_ap=float(average_precision_score(y[select],p[select])),
                            validation=operating(y,cp,threshold),seconds=seconds,test=None,
                            calibration=dict(coef=fit.coef_.tolist(),intercept=fit.intercept_.tolist(),threshold=threshold))
                        np.savez_compressed(out/'validation_predictions.npz',ids=ids,y=y,probability=p,calibrated_probability=cp)
                        if args.phase=='main':
                            ids=torch.where(data.test_mask)[0].numpy();p=np.zeros(len(ids))
                            for k,model in enumerate(models):
                                mask=owner[torch.from_numpy(ids)].numpy()==k if local else np.ones(len(ids),dtype=bool)
                                p[mask]=model.predict_proba(data.x[ids[mask]].numpy())[:,1]
                            y=data.y[ids].numpy()
                            z=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))[:,None]
                            cp=fit.predict_proba(z)[:,1]
                            result['test']=operating(y,cp,threshold)
                            result['primary_test_ap']=float(average_precision_score(y,p))
                            np.savez_compressed(out/'test_predictions.npz',ids=ids,y=y,probability=p,calibrated_probability=cp,
                                timestep=data.timestep[torch.from_numpy(ids)].numpy(),owner=owner[torch.from_numpy(ids)].numpy())
                        write_json(out/'result.json',result)
                        manifest['results'].append(result);write_json(root/'manifest.json',manifest)
                        print(args.phase,args.dataset,name,result['selection_ap'],flush=True)
                        del models
    manifest['status']='completed';write_json(root/'manifest.json',manifest)


if __name__=='__main__':
    main()
