"""Explicit ablation-only stable source-ID thinning of the frozen full router.

Main runs have fraction1 and use the untouched frozen router. CUDA RNG prefix
layout need not be stable across tensor lengths or GPU architectures; an integer
hash defines availability independently of snapshot size, model RNG and device.
"""
import torch
from experiments.refinement_diagnostic import routed_exchange as full_router


def source_available(ids,fraction):
    return ((ids*1103515245+20261006)%2147483647).to(torch.float64)/2147483647<fraction


def routed_exchange(models,views,device,mode='current',seed=0,fraction=1.):
    if fraction==1.:
        return full_router(models,views,device,mode,seed,fraction=1.)
    if mode=='oracle':raise ValueError('The full-neighborhood oracle requires fraction=1')
    remote,raw,_=full_router(models,views,device,mode,seed,fraction=1.)
    selected=[];outputs=[];fanout=0;edges=0
    for v,r in zip(views,remote):
        src=v['remote_src'].to(device);keep=source_available(src,fraction)
        kept=src[keep];selected.append(kept);fanout+=int(kept.unique().numel());edges+=len(kept)
        outputs.append((r[0][keep],r[1][keep]) if r is not None and keep.any() else None)
    unique=int(torch.cat(selected).unique().numel());dim=models[0].norm.normalized_shape[0]
    return outputs,raw,dict(unique_sender_bytes=unique*dim*4,receiver_fanout_bytes=fanout*dim*4,received_edges=edges,oracle_raw_fanout_bytes=0)
