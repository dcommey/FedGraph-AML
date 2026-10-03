"""Describe the fixed task, degree distribution and registered ownership support.

After the gate, labels are descriptive support, not a selection or resampling rule.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from data.refinement_data import prepare,sha256
from experiments.refinement_study import source_hashes,write_json
from experiments.corrected_evaluation import digest_tensor


def raw_identity_audit(dataset,data,data_root,out):
    """Read IDs/time/outcomes directly with csv, independently of pandas/cache."""
    if dataset=='ibm':
        records=[]
        with (data_root/'HI-Small_Trans.csv').open(newline='') as stream:
            reader=csv.reader(stream);header=next(reader)
            if len(header)!=11:raise AssertionError('IBM column count changed')
            for raw_id,row in enumerate(reader):
                if len(row)!=11:raise AssertionError('Malformed source row')
                stamp=row[0]
                if ('2022/09/01 00:00'<=stamp<'2022/09/11 00:00' and
                    ((raw_id*1103515245+20261002)%2147483647)%5==0):
                    records.append((stamp,raw_id,int(row[10])))
        records.sort(key=lambda r:(r[0],r[1]))
        raw_ids=np.array([r[1] for r in records],dtype=np.int64)
        times=np.array([int(r[0][8:10])-1 for r in records],dtype=np.int64)
        labels=np.array([r[2] for r in records],dtype=np.int64)
    else:
        raw=data_root if (data_root/'elliptic_txs_classes.csv').exists() else data_root/'raw'
        outcomes={}
        with (raw/'elliptic_txs_classes.csv').open(newline='') as stream:
            reader=csv.reader(stream);next(reader)
            for identifier,label in reader:
                key=int(identifier)
                if key in outcomes:raise AssertionError('Duplicate original class ID')
                outcomes[key]={'1':1,'2':0,'unknown':-1}[label]
        raw_ids=[];times=[]
        with (raw/'elliptic_txs_features.csv').open(newline='') as stream:
            for row in csv.reader(stream):
                raw_ids.append(int(row[0]));times.append(int(row[1])-1)
        raw_ids=np.asarray(raw_ids,dtype=np.int64);times=np.asarray(times,dtype=np.int64)
        labels=np.array([outcomes[int(i)] for i in raw_ids],dtype=np.int64)
    if (not np.array_equal(raw_ids,data.transaction_ids.numpy()) or
        not np.array_equal(times,data.timestep.numpy()) or not np.array_equal(labels,data.y.numpy())):
        raise AssertionError('Original CSV ID/time/outcome differs from prepared nodes')
    target=out/'node_identity_map.npz'
    np.savez_compressed(target,node_index=np.arange(len(raw_ids)),original_transaction_id=raw_ids,timestep=times,y=labels)
    return dict(passed=True,nodes=len(raw_ids),method='Independent standard-library CSV ID/time/label read and label-blind sample/order verification',
        map_sha256=sha256(target),source_sha256=sha256(__file__))


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--main-root',required=True);p.add_argument('--gate',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    gate=json.loads(Path(a.gate).read_text())
    if not gate['passed'] or gate['source_sha256']!=source_hashes():raise RuntimeError('Data audit gate changed')
    torch.set_num_threads(4)
    data,meta=prepare(a.dataset,Path('data')/a.dataset,out)
    if meta!=gate['dataset']:raise RuntimeError('Data audit preparation differs')
    indegree=torch.bincount(data.edge_index[1],minlength=data.num_nodes).numpy()
    outdegree=torch.bincount(data.edge_index[0],minlength=data.num_nodes).numpy()
    observed_masks=dict(train=data.timestep<=data.train_end,
        val=torch.isin(data.timestep,torch.tensor(data.val_steps)),
        test=torch.isin(data.timestep,torch.tensor(data.test_steps)))
    result=dict(dataset=meta,indegree=dict(mean=float(indegree.mean()),maximum=int(indegree.max()),isolated_incoming_fraction=float((indegree==0).mean()),quantiles=np.quantile(indegree,[0,.25,.5,.75,.95,.99,1]).tolist()),
        outdegree=dict(mean=float(outdegree.mean()),maximum=int(outdegree.max())),
        cross_time_edges=int((data.timestep[data.edge_index[0]]!=data.timestep[data.edge_index[1]]).sum()),support={},ownership={})
    result['raw_identity_audit']=raw_identity_audit(a.dataset,data,Path('data')/a.dataset,out)
    result['standardized_feature_audit']=dict(maximum_absolute_value=float(data.x.abs().max()),
        constant_training_columns=torch.where(data.x[observed_masks['train']].std(0)==0)[0].tolist())
    if a.dataset=='ibm':
        offset=2;unknown={}
        for name,vocab in meta['feature_vocabularies'].items():
            index=offset+len(vocab);offset=index+1
            unknown[name]={split:int((data.x[observed_masks[split],index]!=0).sum()) for split in ['train','val','test']}
        result['standardized_feature_audit']['unknown_category_counts']=unknown
    for split in ['train','val','test']:
        mask=observed_masks[split];known=mask&(data.y>=0)
        result['support'][split]=dict(observed=int(mask.sum()),labeled=int(known.sum()),positive=int((known&(data.y==1)).sum()),negative=int((known&(data.y==0)).sum()),unknown=int((mask&(data.y<0)).sum()))
    for strategy in ['random','louvain']:
        for seed in [20261003,20261004,20261005]:
            owner=torch.from_numpy(np.load(Path(a.main_root)/f'ownership-{strategy}-{seed}.npy',allow_pickle=False))
            if digest_tensor(owner)!=gate['ownership_sha256'][f'{strategy}-{seed}']:raise RuntimeError('Audited ownership changed')
            cuts={}
            for split in ['train','val','test']:
                mask=observed_masks[split];edges=data.edge_index[:,mask[data.edge_index[1]]]
                cuts[split]=dict(edge_count=edges.shape[1],cut_fraction=float((owner[edges[0]]!=owner[edges[1]]).float().mean()),
                    clients=[dict(observed=int((mask&(owner==k)).sum()),positive=int((mask&(owner==k)&(data.y==1)).sum()),negative=int((mask&(owner==k)&(data.y==0)).sum()),unknown=int((mask&(owner==k)&(data.y<0)).sum())) for k in range(3)])
            result['ownership'][f'{strategy}-{seed}']=cuts
    write_json(out/'data_audit.json',result)


if __name__=='__main__':main()
