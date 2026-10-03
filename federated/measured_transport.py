"""Bounded numeric framing and authenticated context for the snapshot probe.

TLS supplies authenticated encryption. This is not a post-quantum protocol.
"""
import hashlib
import io
import json
import struct
import zipfile
import numpy as np

MAX_BYTES=16*1024*1024


def encode(header,arrays):
    f=io.BytesIO()
    for a in arrays.values():
        if a.dtype not in [np.dtype('float32'),np.dtype('int64')] or a.size>2000000:
            raise ValueError('Unsupported numeric array')
    np.savez(f,**arrays)
    body=f.getvalue()
    meta=dict(header,payload_sha256=hashlib.sha256(body).hexdigest())
    h=json.dumps(meta,sort_keys=True,separators=(',',':')).encode()
    packet=struct.pack('!I',len(h))+h+body
    if len(packet)>MAX_BYTES:raise ValueError('Frame too large')
    return struct.pack('!I',len(packet))+packet


def exact(sock,n):
    out=bytearray()
    while len(out)<n:
        part=sock.recv(n-len(out))
        if not part:raise EOFError('Truncated frame')
        out.extend(part)
    return bytes(out)


def receive(sock):
    n=struct.unpack('!I',exact(sock,4))[0]
    if n>MAX_BYTES or n<8:raise ValueError('Invalid frame length')
    packet=exact(sock,n)
    hlen=struct.unpack('!I',packet[:4])[0]
    if hlen>16384 or hlen>n-4:raise ValueError('Invalid context length')
    h=json.loads(packet[4:4+hlen]);body=packet[4+hlen:]
    if hashlib.sha256(body).hexdigest()!=h['payload_sha256']:raise ValueError('Tampered payload')
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        if len(z.infolist())>100 or sum(x.file_size for x in z.infolist())>MAX_BYTES:
            raise ValueError('Expanded numeric frame too large')
    with np.load(io.BytesIO(body),allow_pickle=False) as a:
        arrays={name:a[name] for name in a.files}
    for a in arrays.values():
        if a.dtype not in [np.dtype('float32'),np.dtype('int64')] or a.size>2000000 or a.ndim>2:
            raise ValueError('Malformed numeric shape/dtype')
        if not np.isfinite(a).all():raise ValueError('Nonfinite numeric payload')
    return h,arrays,n+4


def validate_context(h,run,round_index,source,destination,kind):
    expected=dict(run=run,round=round_index,sequence=round_index,source=source,destination=destination,kind=kind,layer=1)
    if any(h.get(k)!=v for k,v in expected.items()):
        raise ValueError('Wrong identity, recipient, round, replay or layer')


def context(run,r,source,destination,kind):
    return dict(run=run,round=r,sequence=r,source=source,destination=destination,kind=kind,layer=1)


def mean(h,edge,remote=None):
    sums=np.zeros_like(h);degree=np.zeros(len(h),dtype=np.float32)
    np.add.at(sums,edge[1],h[edge[0]]);np.add.at(degree,edge[1],1)
    if remote is not None:
        dst,state=remote;np.add.at(sums,dst,state);np.add.at(degree,dst,1)
    return sums/np.maximum(degree,1)[:,None]


def first_numpy(x,edge,state):
    h=mean(x,edge)@state['first.lin_l.weight'].T+state['first.lin_l.bias']+x@state['first.lin_r.weight'].T
    h=(h-h.mean(1,keepdims=True))/np.sqrt(h.var(1,keepdims=True)+1e-5)
    return np.maximum(h*state['norm.weight']+state['norm.bias'],0)


def second_numpy(h,edge,remote,state):
    z=mean(h,edge,remote)@state['second.lin_l.weight'].T+state['second.lin_l.bias']+h@state['second.lin_r.weight'].T
    return (np.maximum(z,0)@state['classifier.weight'].T+state['classifier.bias']).reshape(-1)
