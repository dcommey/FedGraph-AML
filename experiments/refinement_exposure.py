"""Recipient exposure experiments, never a privacy guarantee.

Public/scaler/model-aware auxiliary reconstruction and controlled supervised
membership. Attack splits are node-disjoint; unknown labels are not benign.
"""
import argparse
import copy
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score,roc_curve
from data.refinement_data import prepare,sha256
from models.refined_sage import RefinedSAGE
from experiments.corrected_evaluation import make_ownership,build_views,seed_all
from experiments.refinement_study import write_json,run_neural


def states_for(models,data,owner,steps,allowed_owner=None,recipient=None):
    hs,xs,all_ids=[],[],[]
    for step in steps:
        views=build_views(data,owner,3,step,target_step=step)
        visible=torch.unique(views[recipient]['remote_src']) if recipient is not None else None
        for k,(model,v) in enumerate(zip(models,views)):
            if allowed_owner is not None and k!=allowed_owner:
                continue
            mask=(data.timestep[v['ids']]==step)
            if visible is not None:
                mask &= torch.isin(v['ids'],visible)
            if not mask.any():
                continue
            model.eval()
            with torch.no_grad():
                h=model.first_layer(v['x'].cuda(),v['edge_index'].cuda()).cpu()[mask]
            ids=v['ids'][mask]
            hs.append(h.numpy());xs.append(data.x[ids].numpy());all_ids.append(ids.numpy())
    return np.concatenate(hs),np.concatenate(xs),np.concatenate(all_ids)


def errors(truth,pred):
    mse=((truth-pred)**2).mean(0)
    return dict(mean_normalized_feature_mse=float(mse.mean()),per_feature_mse=mse.tolist(),
        rmse=float(np.sqrt(mse.mean())),n=len(truth),
        mean_feature_r2=float(np.mean(1-mse/np.maximum(truth.var(0),1e-8))))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root',default='data/elliptic')
    ap.add_argument('--main-root',required=True)
    ap.add_argument('--gate',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    data,meta=prepare('elliptic',a.data_root,root)
    owner=make_ownership(data,'random',20261003,3)
    checkpoint=torch.load(Path(a.main_root)/'random-o20261003-s42-current/checkpoint.pt',map_location='cpu',weights_only=False)
    config=checkpoint['config']
    models=[]
    for state in checkpoint['states']:
        model=RefinedSAGE(data.num_features,config['hidden'],config['dropout']).cuda()
        model.load_state_dict(state);models.append(model)
    # Recipient1 knows its own features and can compute auxiliary source states.
    ztrain,xtrain,itrain=states_for(models,data,owner,range(0,30),allowed_owner=1)
    zval,xval,ival=states_for(models,data,owner,range(30,34),allowed_owner=1)
    zeval,xeval,ieval=states_for(models,data,owner,range(39,49),recipient=1)
    if set(itrain)&set(ival) or set(itrain)&set(ieval) or set(ival)&set(ieval):
        raise AssertionError('Repeated node leaked across attack splits')
    results=[]
    reconstruction_predictions={}
    mean=np.repeat(xtrain.mean(0)[None,:],len(ieval),axis=0)
    results.append(dict(attack='mean',evaluation=errors(xeval,mean)))
    reconstruction_predictions['mean']=mean
    trials=[]
    for alpha in [.1,1.,10.,100.]:
        model=Ridge(alpha=alpha).fit(ztrain,xtrain)
        trial=dict(alpha=alpha,validation=errors(xval,model.predict(zval)))
        trials.append((trial,model))
    trial,model=min(trials,key=lambda item:item[0]['validation']['mean_normalized_feature_mse'])
    ridge_prediction=model.predict(zeval)
    results.append(dict(attack='ridge',selected_alpha=trial['alpha'],validation_trials=[t for t,_ in trials],evaluation=errors(xeval,ridge_prediction)))
    reconstruction_predictions['ridge']=ridge_prediction
    model=MLPRegressor(hidden_layer_sizes=(128,),random_state=20261002,max_iter=100,batch_size=256,
                       learning_rate_init=.001,early_stopping=False).fit(ztrain,xtrain)
    nonlinear_prediction=model.predict(zeval)
    results.append(dict(attack='nonlinear_mlp',validation=errors(xval,model.predict(zval)),evaluation=errors(xeval,nonlinear_prediction)))
    reconstruction_predictions['nonlinear_mlp']=nonlinear_prediction
    np.savez_compressed(root/'reconstruction_exposure.npz',auxiliary_ids=itrain,validation_ids=ival,evaluation_ids=ieval,
                        exposed_embeddings=zeval,truth_normalized_features=xeval,**reconstruction_predictions)
    for m in models:del m
    torch.cuda.empty_cache()
    # Controlled within-period supervised-label membership, graph presence held
    # constant. Its validation checkpoint is not an additional utility main run.
    gate=json.loads(Path(a.gate).read_text())
    original_mask=data.train_mask.clone()
    gen=torch.Generator().manual_seed(20261007)
    membership=torch.rand(data.num_nodes,generator=gen)<.5
    data.train_mask=original_mask&membership
    args=argparse.Namespace(clients=3,device='cuda',rounds=50,local_epochs=2,evaluate_every=5,fraction=1.,dataset='elliptic')
    run_neural(data,owner,20261003,701,'random','current',gate['selected_configs']['sage_fl'],args,root/'membership-victim','pilot')
    victim=torch.load(root/'membership-victim/checkpoint.pt',map_location='cpu',weights_only=False)
    models=[]
    for state in victim['states']:
        m=RefinedSAGE(data.num_features,config['hidden'],gate['selected_configs']['sage_fl']['dropout']).cuda()
        m.load_state_dict(state);models.append(m)
    z,x,ids=states_for(models,data,owner,range(0,34),recipient=1)
    known=original_mask[torch.from_numpy(ids)].numpy()
    z,ids=z[known],ids[known]
    degree=torch.bincount(data.edge_index[1],minlength=data.num_nodes)
    buckets=np.minimum(np.floor(np.log2(degree[torch.from_numpy(ids)].numpy()+1)).astype(int),5)
    label=data.y[torch.from_numpy(ids)].numpy()
    times=data.timestep[torch.from_numpy(ids)].numpy()
    owners=owner[torch.from_numpy(ids)].numpy()
    target=membership[torch.from_numpy(ids)].numpy().astype(int)
    balanced=[]
    rng=np.random.default_rng(20261008)
    groups=sorted(set(zip(times,owners,label,buckets)))
    for step,k,y,b in groups:
        group=(times==step)&(owners==k)&(label==y)&(buckets==b)
        yes=np.where(group&(target==1))[0];no=np.where(group&(target==0))[0]
        n=min(len(yes),len(no))
        balanced.extend(rng.choice(yes,n,replace=False));balanced.extend(rng.choice(no,n,replace=False))
    balanced=np.asarray(balanced,dtype=int)
    rng.shuffle(balanced);cut=int(len(balanced)*.7)
    train_idx,eval_idx=balanced[:cut],balanced[cut:]
    attacks=[]
    membership_predictions={}
    for name,model in [('logistic',LogisticRegression(max_iter=2000)),
                       ('random_forest',RandomForestClassifier(n_estimators=200,max_depth=8,n_jobs=4,random_state=20261008))]:
        model.fit(z[train_idx],target[train_idx]);score=model.predict_proba(z[eval_idx])[:,1]
        fpr,tpr,_=roc_curve(target[eval_idx],score)
        attacks.append(dict(attack=name,roc_auc=float(roc_auc_score(target[eval_idx],score)),
            tpr_at_fpr_001=float(tpr[fpr<=.01].max()),n_auxiliary=len(train_idx),n_evaluation=len(eval_idx),
            auxiliary_member_count=int(target[train_idx].sum()),evaluation_member_count=int(target[eval_idx].sum())))
        membership_predictions[name]=score
    np.savez_compressed(root/'membership_exposure.npz',auxiliary_ids=ids[train_idx],evaluation_ids=ids[eval_idx],
                        auxiliary_membership=target[train_idx],embedding=z[eval_idx],membership=target[eval_idx],
                        auxiliary_timestep=times[train_idx],evaluation_timestep=times[eval_idx],
                        auxiliary_owner=owners[train_idx],evaluation_owner=owners[eval_idx],
                        auxiliary_label=label[train_idx],evaluation_label=label[eval_idx],
                        auxiliary_degree_bucket=buckets[train_idx],evaluation_degree_bucket=buckets[eval_idx],
                        **membership_predictions)
    record=dict(dataset=meta,reconstruction=results,supervised_membership=attacks,
        source_sha256={p:sha256(p) for p in ['experiments/refinement_exposure.py','experiments/refinement_study.py','experiments/refinement_diagnostic.py','models/refined_sage.py','data/refinement_data.py']},
        gate_sha256=sha256(a.gate),utility_checkpoint_sha256=sha256(Path(a.main_root)/'random-o20261003-s42-current/checkpoint.pt'),
        threat_model='recipient1; shared model/scaler, own train-period shadow features; known-membership auxiliary examples for membership',
        exposure='actual routed first-layer states and source IDs; model state known; no remote raw graph/features supplied to attack predictor',
        membership_definition='label used in supervised loss; all graph nodes remain visible',
        matching='same train period; class,owner,timestep,in-degree bucket balanced before auxiliary/evaluation split',
        limitations='one victim/draw; frozen inference representations only; repeated training-round/adaptive/update attacks untested; temporal shift remains in reconstruction; finite auxiliary attacks are not comprehensive; failure is not privacy',
        privacy_guarantee=False)
    write_json(root/'exposure_results.json',record)


if __name__=='__main__':main()
