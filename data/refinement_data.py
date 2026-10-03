"""Audited temporal datasets for the prospective refinement study.

IBM transaction continuity is an explicit constructed proxy: the three nearest
prior incoming payments into the sender's bank/account within 24 hours. It does
not claim to trace actual funds. No outcomes enter sampling, features or edges.
"""
import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from data.elliptic_loader import EllipticDataset


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ibm_sample_keep(indices, modulus=5):
    # Multiplicative permutation modulo a prime, followed by prespecified thinning.
    indices = np.asarray(indices, dtype=np.int64)
    return ((indices * 1103515245 + 20261002) % 2147483647) % modulus == 0


def continuity_edges(seconds, senders, receivers):
    """Nearest three strictly earlier payments; simultaneous rows update as a batch.

    Raw-row order only chooses among otherwise tied *earlier* candidate payments.
    No row at the current timestamp can alter another row's available history.
    """
    recent = defaultdict(lambda: deque(maxlen=3))
    src, dst = [], []
    begin = 0
    while begin < len(seconds):
        end = begin + 1
        now = seconds[begin]
        while end < len(seconds) and seconds[end] == now:
            end += 1
        for i in range(begin, end):
            for previous in recent[senders[i]]:
                if now - seconds[previous] <= 86400:
                    src.append(previous)
                    dst.append(i)
        for i in range(begin, end):
            recent[receivers[i]].append(i)
        begin = end
    return np.array([src, dst], dtype=np.int64)


def load_ibm(raw_path, cache_dir):
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    cache = root / 'ibm-proxy-v2.npz'
    raw_hash = sha256(raw_path)
    if cache.exists():
        a = np.load(cache, allow_pickle=False)
        meta = json.loads(str(a['metadata']))
        if meta['raw_sha256'] != raw_hash:
            raise ValueError('IBM raw data changed relative to cached graph')
        data = Data(x=torch.from_numpy(a['x']), y=torch.from_numpy(a['y']),
            edge_index=torch.from_numpy(a['edges']), timestep=torch.from_numpy(a['step']),
            transaction_ids=torch.from_numpy(a['ids']), bank=torch.from_numpy(a['bank']))
    else:
        pieces, offset, discarded_tail, full_rows = [], 0, 0, 0
        names = ['timestamp','from_bank','from_account','to_bank','to_account',
                 'received','received_currency','paid','paid_currency','format','label']
        for frame in pd.read_csv(raw_path, names=names, header=0, chunksize=250000,
                                 dtype={'from_account':str,'to_account':str}):
            row_ids = np.arange(offset, offset+len(frame), dtype=np.int64)
            offset += len(frame)
            full_rows += len(frame)
            dates = pd.to_datetime(frame['timestamp'], format='%Y/%m/%d %H:%M')
            primary = (dates >= '2022-09-01') & (dates < '2022-09-11')
            discarded_tail += int((~primary).sum())
            keep = primary.to_numpy() & ibm_sample_keep(row_ids)
            part = frame.loc[keep].copy()
            part['raw_row_id'] = row_ids[keep]
            part['seconds'] = (dates[keep].astype('int64') // 10**9).to_numpy()
            pieces.append(part)
        df = pd.concat(pieces, ignore_index=True).sort_values(['seconds','raw_row_id'], kind='stable').reset_index(drop=True)
        step = ((df['seconds'].to_numpy() - pd.Timestamp('2022-09-01').value//10**9)//86400).astype(np.int64)
        if not (np.isfinite(df[['received','paid']].to_numpy()).all() and
                (df[['received','paid']].to_numpy() >= 0).all()):
            raise ValueError('Invalid IBM payment amounts')
        features = [np.log1p(df['received'].to_numpy(dtype=np.float32)),
                    np.log1p(df['paid'].to_numpy(dtype=np.float32))]
        vocabularies = {}
        for col in ['received_currency','paid_currency','format']:
            vocab = sorted(df.loc[step < 6, col].unique().tolist())
            vocabularies[col] = vocab
            # Unknown category has its own column; fitting uses train only.
            for category in vocab + ['__UNKNOWN__']:
                features.append((~df[col].isin(vocab) if category == '__UNKNOWN__' else df[col] == category).to_numpy(dtype=np.float32))
        x = np.stack(features, axis=1).astype(np.float32)
        seconds = df['seconds'].to_numpy()
        senders = list(zip(df['from_bank'].tolist(), df['from_account'].tolist()))
        receivers = list(zip(df['to_bank'].tolist(), df['to_account'].tolist()))
        edges = continuity_edges(seconds, senders, receivers)
        y = df['label'].to_numpy(dtype=np.int64)
        if not np.isin(y,[0,1]).all():
            raise ValueError('Invalid IBM outcome codes')
        meta = dict(name='IBM HI-Small fixed 20% temporal transaction-proxy subset',
            source='https://github.com/IBM/AML-Data', license='CDLA-Sharing-1.0',
            raw_sha256=raw_hash, raw_rows=full_rows, excluded_outside_primary_period=discarded_tail,
            sampling='label-blind raw-row multiplicative permutation modulo5, seed20261002',
            primary_period='2022-09-01..10', feature_vocabularies=vocabularies,
            edge_rule='nearest3 strictly earlier incoming transfers into source bank/account within24h; simultaneous updates batched; earlier candidate ties resolved by raw row ID',
            adapter_version='ibm-proxy-v2-batched-timestamps',
            synthetic=True, known_test_outcomes=False, overlap_with_elliptic='different synthetic bank-transfer generator; no Bitcoin transaction IDs')
        ids = df['raw_row_id'].to_numpy(dtype=np.int64)
        bank = df['from_bank'].to_numpy(dtype=np.int64)
        np.savez_compressed(cache,x=x,y=y,edges=edges,step=step,ids=ids,bank=bank,metadata=json.dumps(meta))
        data = Data(x=torch.from_numpy(x),y=torch.from_numpy(y),edge_index=torch.from_numpy(edges),
                    timestep=torch.from_numpy(step),transaction_ids=torch.from_numpy(ids),bank=torch.from_numpy(bank))
    data.train_end = 5
    data.val_steps = [6,7]
    data.test_steps = [8,9]
    data.train_mask = data.timestep <= 5
    data.val_mask = (data.timestep >= 6) & (data.timestep <= 7)
    data.test_mask = data.timestep >= 8
    data.no_cross_time_edges = bool((data.timestep[data.edge_index[0]] == data.timestep[data.edge_index[1]]).all())
    return data, meta


def prepare(dataset, data_root, output):
    if dataset == 'elliptic':
        loader = EllipticDataset(data_root)
        data = loader.load()
        data.x = data.x[:,:93].clone()
        data.train_end, data.val_steps, data.test_steps = 33, list(range(34,39)), list(range(39,49))
        data.no_cross_time_edges = bool((data.timestep[data.edge_index[0]] == data.timestep[data.edge_index[1]]).all())
        meta = dict(name='Elliptic local93',raw_sha256={p.name:sha256(p) for p in sorted(loader.raw_dir.glob('*.csv'))},
                    synthetic=False,known_test_outcomes=True,scope='exploratory refinement on previously exposed test')
    else:
        data, meta = load_ibm(Path(data_root)/'HI-Small_Trans.csv', Path(data_root)/'cache')
    observed_train = data.timestep <= data.train_end
    mean = data.x[observed_train].mean(0)
    scale = data.x[observed_train].std(0).clamp_min(1e-6)
    data.x = (data.x-mean)/scale
    np.savez_compressed(Path(output)/'preprocessing.npz',mean=mean.numpy(),scale=scale.numpy())
    meta.update(nodes=data.num_nodes,edges=data.num_edges,features=data.num_features,
                preprocessing_numeric_sha256=hashlib.sha256(mean.numpy().tobytes()+scale.numpy().tobytes()).hexdigest(),
                no_cross_time_edges=data.no_cross_time_edges)
    return data, meta
