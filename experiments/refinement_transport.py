"""Actual mutual-TLS snapshot routing probe, with frozen-model parity.

One GPU host holds two logical senders; the receiver uses NumPy or a second
GPU's frozen model. No training, network FL convergence or PQ claim.
"""
import argparse
import hashlib
import json
import socket
import ssl
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from federated.measured_transport import encode,receive,context,validate_context,first_numpy,second_numpy


def prepare(a):
    import torch
    from data.refinement_data import prepare as load
    from experiments.corrected_evaluation import make_ownership,build_views
    from experiments.refinement_diagnostic import routed_exchange
    from models.refined_sage import RefinedSAGE
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    data,_=load('elliptic',a.data_root,root)
    owner=make_ownership(data,'random',20261003,3)
    c=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    models=[]
    for state in c['states']:
        m=RefinedSAGE(data.num_features,c['config']['hidden'],c['config']['dropout']).cuda().eval()
        m.load_state_dict(state);models.append(m)
    views=build_views(data,owner,3,38,target_step=38)
    remote,_,_=routed_exchange(models,views,'cuda')
    state={k:v.numpy().astype(np.float32) for k,v in c['states'][2].items()}
    with torch.no_grad():
        hs=[m.first_layer(v['x'].cuda(),v['edge_index'].cuda()).cpu().numpy() for m,v in zip(models,views)]
        truth=models[2](views[2]['x'].cuda(),views[2]['edge_index'].cuda(),remote[2]).cpu().numpy()
    v=views[2]
    needed=np.unique(np.concatenate([w['remote_src'].numpy() for w in views[:2]]))
    send_ids=needed[owner[torch.from_numpy(needed)].numpy()==2]
    np.savez(root/'client-view.npz',ids=v['ids'].numpy(),x=v['x'].numpy(),edge=v['edge_index'].numpy(),
        remote_dst=v['remote_dst'].numpy(),remote_src=v['remote_src'].numpy(),send_ids=send_ids,**state)
    np.savez(root/'server-view.npz',ids=v['ids'].numpy(),send_ids=send_ids,reference_first=hs[2],reference_logits=truth,
        remote=remote[2][1].cpu().numpy(),remote_dst=v['remote_dst'].numpy(),**state)
    print('PREPARED',len(v['ids']),len(send_ids),flush=True)


def server(a):
    root=Path(a.output);arrays=dict(np.load(root/'server-view.npz',allow_pickle=False))
    tls=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);tls.minimum_version=ssl.TLSVersion.TLSv1_3
    tls.load_cert_chain(a.cert,a.key);tls.load_verify_locations(a.ca);tls.verify_mode=ssl.CERT_REQUIRED
    listener=socket.socket();listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);listener.bind((a.host,a.port));listener.listen(8);listener.settimeout(180)
    print('LISTENING',a.host,a.port,a.run,flush=True)
    failures=[];records=[];complete=False
    while not complete:
        connection,_=listener.accept()
        try:
            with tls.wrap_socket(connection,server_side=True) as s:
                s.settimeout(30)
                name=dict(x[0] for x in s.getpeercert()['subject'])['commonName']
                if name!='owner-2':raise ValueError('Unauthorized certificate identity')
                for r in range(1,36):
                    start=time.perf_counter();h,x,n=receive(s)
                    validate_context(h,a.run,r,'owner-2','coordinator','embedding')
                    if set(x)!={'ids','h'} or x['ids'].ndim!=1 or not np.array_equal(x['ids'],arrays['send_ids']) or x['h'].shape!=(len(x['ids']),64):
                        raise ValueError('Wrong owner routing or malformed tensor shape')
                    local=np.searchsorted(arrays['ids'],x['ids'])
                    first_error=float(np.max(np.abs(x['h']-arrays['reference_first'][local])))
                    if first_error>1e-4:raise ValueError('Client first-layer parity failed')
                    payload=encode(context(a.run,r,'coordinator','owner-2','remote'),
                        {'remote':arrays['remote'],'remote_dst':arrays['remote_dst']})
                    s.sendall(payload)
                    h,x,n2=receive(s);validate_context(h,a.run,r,'owner-2','coordinator','logits')
                    if set(x)!={'logits'} or x['logits'].shape!=arrays['reference_logits'].shape:
                        raise ValueError('Malformed inference tensor shape')
                    error=float(np.max(np.abs(x['logits']-arrays['reference_logits'])))
                    if error>1e-4:raise ValueError('Transport/in-process prediction parity failed')
                    ack=encode(context(a.run,r,'coordinator','owner-2','ack'),{'error':np.array([error],dtype=np.float32)})
                    s.sendall(ack)
                    records.append(dict(round=r,seconds=time.perf_counter()-start,
                        application_bytes_received=n+n2,application_bytes_sent=len(payload)+len(ack),max_abs_error=error,first_max_abs_error=first_error))
                # Replay last valid frame after completed observations; no state
                # may be accepted for another round or another connection/run.
                h,_,_=receive(s)
                try:validate_context(h,a.run,36,'owner-2','coordinator','logits')
                except ValueError:failures.append('replay_rejected');complete=True
                if not complete:raise ValueError('Expected replay rejection')
        except (ValueError,ssl.SSLError,EOFError,KeyError) as exc:
            failures.append(type(exc).__name__+': '+str(exc))
            connection.close()
    listener.close()
    warm=records[5:]
    volumes=[r['application_bytes_received']+r['application_bytes_sent'] for r in warm]
    result=dict(run=a.run,placement=a.placement,
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['experiments/refinement_transport.py','federated/measured_transport.py']},
        snapshot_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['client-view.npz','server-view.npz']},
        trust_anchor_file_sha256=hashlib.sha256(Path(a.ca).read_bytes()).hexdigest(),
        protocol='mutual TLS1.3 over Tailscale, safe numeric framing and authenticated owner/context',
        operation='snapshot inference routing; source states prepared on GTX before measurement; no transported training rounds',
        warmup=5,steady_state_observations=len(warm),median_seconds=float(np.median([r['seconds'] for r in warm])),
        p95_seconds=float(np.quantile([r['seconds'] for r in warm],.95)),
        application_bytes_per_round=int(np.median(volumes)),
        application_bytes_statistic='median of30steady observations; JSON round/sequence digit length can vary',
        application_bytes_minimum=min(volumes),application_bytes_maximum=max(volumes),application_bytes_total=sum(volumes),
        byte_scope='numeric arrays, routing IDs and envelopes; excludes TLS/TCP/IP/Tailscale framing and initial staging',
        max_abs_error=max(r['max_abs_error'] for r in records),records=records,rejection_checks=failures,
        post_quantum=False,privacy_guarantee=False)
    (root/'transport_results.json').write_text(json.dumps(result,indent=2,allow_nan=False))


def client(a):
    data=dict(np.load(a.view,allow_pickle=False))
    state={k:v for k,v in data.items() if '.' in k}
    tls=ssl.create_default_context(cafile=a.ca);tls.minimum_version=ssl.TLSVersion.TLSv1_3;tls.load_cert_chain(a.cert,a.key)
    def connect():
        s=tls.wrap_socket(socket.create_connection((a.host,a.port),timeout=30),server_hostname='gtx-coordinator');s.settimeout(30);return s
    model=None
    if a.device=='cuda':
        import torch
        from models.refined_sage import RefinedSAGE
        model=RefinedSAGE(data['x'].shape[1],64,0).cuda().eval()
        model.load_state_dict({k:torch.from_numpy(v) for k,v in state.items()})
        own_x=torch.from_numpy(data['x']).cuda()
        own_edge=torch.from_numpy(data['edge']).cuda()
        with torch.no_grad():first=model.first_layer(own_x,own_edge).cpu().numpy()
    else:
        first=first_numpy(data['x'],data['edge'],state).astype(np.float32)
    ids=data['send_ids'];local=np.searchsorted(data['ids'],ids)
    arrays={'ids':ids,'h':first[local]}
    # A peer without a client certificate must fail TLS authentication before
    # any application frame can be accepted.
    anonymous=ssl.create_default_context(cafile=a.ca);anonymous.minimum_version=ssl.TLSVersion.TLSv1_3
    try:
        with anonymous.wrap_socket(socket.create_connection((a.host,a.port),timeout=30),server_hostname='gtx-coordinator') as s:
            s.settimeout(30);s.sendall(encode(context(a.run,1,'owner-2','coordinator','embedding'),arrays))
            receive(s)
    except (ssl.SSLError,EOFError,ConnectionResetError,BrokenPipeError):
        pass
    else:
        raise RuntimeError('Unauthenticated client was accepted')
    # Context/shape/digest negatives, each independent of benchmark observations.
    for case in ['wrong_recipient','stale_round','malformed_shape','tampered']:
        with connect() as s:
            h=context(a.run,1,'owner-2','coordinator','embedding');x=arrays
            if case=='wrong_recipient':h['destination']='owner-0'
            if case=='stale_round':h['round']=0
            if case=='malformed_shape':x=dict(arrays,h=arrays['h'][:1])
            packet=encode(h,x)
            if case=='tampered':packet=packet[:-1]+bytes([packet[-1]^1])
            try:
                s.sendall(packet);receive(s);raise RuntimeError('Invalid message was accepted')
            except (EOFError,ssl.SSLError,ConnectionResetError,BrokenPipeError):pass
    with connect() as s:
        for r in range(1,36):
            s.sendall(encode(context(a.run,r,'owner-2','coordinator','embedding'),arrays))
            h,x,_=receive(s);validate_context(h,a.run,r,'coordinator','owner-2','remote')
            if model is None:
                logits=second_numpy(first,data['edge'],(x['remote_dst'],x['remote']),state).astype(np.float32)
            else:
                with torch.no_grad():
                    logits=model(own_x,own_edge,(torch.from_numpy(x['remote_dst']).cuda(),torch.from_numpy(x['remote']).cuda())).cpu().numpy()
            last=encode(context(a.run,r,'owner-2','coordinator','logits'),{'logits':logits})
            s.sendall(last);h,x,_=receive(s);validate_context(h,a.run,r,'coordinator','owner-2','ack')
            print(r,float(x['error'][0]),flush=True)
        s.sendall(last)


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','server','client'],required=True)
    p.add_argument('--output',default='results/transport-v1');p.add_argument('--data-root',default='data/elliptic');p.add_argument('--checkpoint')
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=38697);p.add_argument('--run',default='fedgraph-transport-20261002-v1')
    p.add_argument('--ca');p.add_argument('--cert');p.add_argument('--key');p.add_argument('--view')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--placement',default='GTX CUDA source-state preparation; Mac NumPy receiver; three logical owners/two hosts')
    a=p.parse_args();{'prepare':prepare,'server':server,'client':client}[a.mode](a)


if __name__=='__main__':main()
