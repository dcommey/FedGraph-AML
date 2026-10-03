"""Secondary synthetic-bank grouping; fixed training-volume assignment, no outcomes.

This is not another independent dataset or a three-draw ownership experiment.
Transaction record belongs to its sender bank; the feature/right assumption is
declared explicitly rather than presenting random node ownership as real silos.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from data.refinement_data import prepare
from experiments.refinement_study import run_neural,write_json,source_hashes,family
from experiments.corrected_evaluation import digest_tensor,environment


def main():
    p=argparse.ArgumentParser();p.add_argument('--gate',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    gate=json.loads(Path(a.gate).read_text())
    if not gate['passed'] or gate['source_sha256']!=source_hashes():raise RuntimeError('Bank sensitivity gate differs')
    data,meta=prepare('ibm','data/ibm',root)
    if meta!=gate['dataset']:raise RuntimeError('Bank sensitivity dataset differs')
    banks,counts=torch.unique(data.bank[data.train_mask],return_counts=True)
    ordered=sorted(zip(banks.tolist(),counts.tolist()),key=lambda v:(-v[1],v[0]))
    volumes=[0,0,0];mapping={}
    for bank,n in ordered:
        client=min(range(3),key=lambda c:(volumes[c],c));mapping[bank]=client;volumes[client]+=n
    # An unseen sender bank has no outcome-informed fallback: deterministic hash.
    for bank in data.bank.unique().tolist():
        if bank not in mapping:
            mapping[bank]=int(hashlib.sha256(str(bank).encode()).hexdigest()[:16],16)%3
    owner=torch.tensor([mapping[b] for b in data.bank.tolist()],dtype=torch.long)
    np.save(root/'ownership.npy',owner.numpy())
    support={}
    for split in ['train','val','test']:
        mask=getattr(data,split+'_mask');edge=data.edge_index[:,mask[data.edge_index[1]]]
        support[split]=dict(nodes=int(mask.sum()),positive=int(data.y[mask].sum()),
            cross_edge_fraction=float((owner[edge[0]]!=owner[edge[1]]).float().mean()),
            clients=[dict(n=int((mask&(owner==k)).sum()),positive=int(data.y[mask&(owner==k)].sum())) for k in range(3)])
    args=argparse.Namespace(clients=3,device='cuda',rounds=50,local_epochs=2,evaluate_every=5,fraction=1.,dataset='ibm')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest=dict(status='running',source_sha256=source_hashes(),dataset=meta,environment=environment(),
        scope='secondary synthetic bank-group sensitivity; one fixed ownership; not institutional confirmation',
        ownership_protocol='sender-bank record ownership; banks greedily grouped by train-period volume, ties by bank ID; unseen bank SHA256 modulo3',
        information_assumption='sender owns the complete payment record features; trusted coordinator knows account continuity edges; public train scaler',
        bank_to_client={str(b):c for b,c in sorted(mapping.items())},train_volumes=volumes,
        ownership_hash=digest_tensor(owner),support=support,results=[])
    write_json(root/'manifest.json',manifest)
    for seed in [42,123,456]:
        for method in ['local','fedavg','current','fresh','fedmlp','fedgcn']:
            r=run_neural(data,owner,0,seed,'bank_groups',method,gate['selected_configs'][family(method)],args,root/f's{seed}-{method}','main')
            manifest['results'].append(r);write_json(root/'manifest.json',manifest)
    manifest['status']='completed';write_json(root/'manifest.json',manifest)


if __name__=='__main__':main()
