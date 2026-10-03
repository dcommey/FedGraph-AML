"""Canonical v2 experiment: causal Elliptic snapshots, true cut-edge exchange.

Historical drivers/results are NOT inputs. Pilot never evaluates held-out test
outcomes. Main runs select checkpoints using pooled validation AP, calibrate
threshold using validation alone, and evaluate each held-out node once.
"""
import argparse
import copy
import hashlib
import json
import os
import platform
import random
import importlib.metadata
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from torch_geometric.data import Data
from data.elliptic_loader import EllipticDataset
from data.partitioner import GraphPartitioner
from models.cross_silo_sage import CrossSiloSAGE

PROTOCOL = 'fedgraph-vasp-v2-causal-cutedge-20261002'


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Deterministic scatter on CUDA is not supported in every PyTorch version.
    # Do not silently promise bitwise determinism: repeated hardware pilot measures it.
    torch.use_deterministic_algorithms(True)


def digest_tensor(t):
    return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()


def source_manifest():
    root = Path(__file__).resolve().parents[1]
    paths = ['experiments/corrected_evaluation.py', 'models/cross_silo_sage.py',
             'data/elliptic_loader.py', 'data/partitioner.py', 'federated/security.py',
             'tests/test_corrected_protocol.py']
    return {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}


def environment():
    return dict(host=platform.node(), python=platform.python_version(),
                torch=torch.__version__, pyg=torch_geometric.__version__,
                numpy=np.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0),
                gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                nvidia_smi=subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total',
                                                   '--format=csv,noheader'], text=True).strip(),
                dependencies={name: importlib.metadata.version(name) for name in
                              ['torch', 'torch-geometric', 'numpy', 'pandas', 'scipy',
                               'scikit-learn', 'networkx', 'pymetis', 'pycryptodome']},
                deterministic_algorithms=True, tf32=False)


def make_ownership(data, strategy, seed, clients):
    """Synthetic ownership fitted separately to each observed timestep snapshot.

    Community packing assumes a trusted coordinator knows the current snapshot
    topology. It uses no labels/features or later snapshot structure. This is
    retrospective screening within each observed window, not transaction arrival
    prediction or empirically observed VASP ownership.
    """
    if strategy == 'random':
        owner = torch.randint(clients, (data.num_nodes,), generator=torch.Generator().manual_seed(seed))
    else:
        if strategy == 'metis':
            import pymetis  # Fail rather than silently substituting spectral clustering.
        owner = torch.empty(data.num_nodes, dtype=torch.long)
        for step in sorted(data.timestep.unique().tolist()):
            ids = torch.where(data.timestep == step)[0]
            mapping = torch.full((data.num_nodes,), -1, dtype=torch.long)
            mapping[ids] = torch.arange(ids.numel())
            edge = data.edge_index[:, (data.timestep[data.edge_index] == step).all(0)]
            graph = Data(edge_index=mapping[edge], num_nodes=ids.numel())
            silos, _ = GraphPartitioner(clients, strategy, seed=seed + step).partition(graph)
            for silo in silos:
                owner[ids[silo.node_mask]] = silo.silo_id
    return owner


def build_views(data, owner, clients, cutoff, target_step=None):
    """Only observed nodes/edges enter a forward pass.

    For a graph whose edges never cross timesteps, a single observed timestep
    is an exact disconnected-component shortcut at evaluation (LayerNorm is
    per node). Otherwise retain the whole causal history through cutoff.
    """
    active = data.timestep <= cutoff
    if target_step is not None and data.no_cross_time_edges:
        active = data.timestep == target_step
    src, dst = data.edge_index
    observed = active[src] & active[dst]
    views = []
    for k in range(clients):
        ids = torch.where(active & (owner == k))[0]
        mapping = torch.full((data.num_nodes,), -1, dtype=torch.long)
        mapping[ids] = torch.arange(ids.numel())
        local = observed & (owner[src] == k) & (owner[dst] == k)
        cross = observed & (owner[src] != k) & (owner[dst] == k)
        view = dict(ids=ids, x=data.x[ids], edge_index=mapping[data.edge_index[:, local]],
                    train_mask=data.train_mask[ids], y=data.y[ids],
                    remote_src=src[cross], remote_dst=mapping[dst[cross]],
                    cutoff=cutoff)
        if ids.numel() == 0:
            raise ValueError(f'Empty client {k} at observed cutoff {cutoff}')
        views.append(view)
    return views


def put_view(view, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in view.items()}


def exchange(models, views, device, shuffled=False, secure=False, shuffle_seed=0):
    """Owner-tagged first-layer snapshot; only requested source states transmitted.

    Round/layer scope is the lifetime of this function's fresh result. No raw
    foreign features or labels are supplied to receivers. Secure mode is a
    local serialization/cryptography simulation, not authenticated networking.
    """
    needed = torch.unique(torch.cat([v['remote_src'] for v in views]))
    if needed.numel() == 0:
        return [None] * len(views), dict(transmitted_nodes=0, transmitted_bytes=0, received_edges=0)
    lookup = {}
    transmitted_bytes = 0
    tunnel = None
    if secure:
        from federated.security import PostQuantumTunnel
        tunnel = PostQuantumTunnel()
    for owner_id, (model, view) in enumerate(zip(models, views)):
        wanted_mask = torch.isin(view['ids'], needed)
        if not wanted_mask.any():
            continue
        model.eval()
        with torch.no_grad():
            h = model.first_layer(view['x'].to(device), view['edge_index'].to(device))[wanted_mask.to(device)].cpu()
        if tunnel:
            payload = tunnel.encrypt_embedding(h)
            if not payload.get('encrypted'):
                raise RuntimeError('Secure exchange produced plaintext')
            decrypted = tunnel.decrypt_embedding(payload)
            if decrypted is None or not torch.equal(decrypted, h):
                raise RuntimeError('Secure exchange parity failed')
            transmitted_bytes += sum(len(payload[k]) for k in ['kem_ct', 'aes_nonce', 'aes_tag', 'ciphertext'])
            h = decrypted
        else:
            transmitted_bytes += h.numel() * h.element_size()
        ids = view['ids'][wanted_mask].tolist()
        for node, embedding in zip(ids, h):
            lookup[node] = (owner_id, embedding)
    remote = []
    for receiver, view in enumerate(views):
        records = [lookup[int(i)] for i in view['remote_src']]
        if any(sender == receiver for sender, _ in records):
            raise AssertionError('Self-owned embedding routed as foreign')
        if records:
            h = torch.stack([e for _, e in records])
            if shuffled:
                h = h[torch.randperm(h.shape[0], generator=torch.Generator().manual_seed(shuffle_seed + receiver))]
            remote.append((view['remote_dst'].to(device), h.to(device).detach()))
        else:
            remote.append(None)
    return remote, dict(transmitted_nodes=len(lookup), transmitted_bytes=transmitted_bytes,
                        received_edges=sum(v['remote_src'].numel() for v in views))


def metric_dict(y, p, threshold=0.5):
    y = np.asarray(y)
    p = np.asarray(p)
    if not len(y) or len(np.unique(y)) < 2:
        raise ValueError('Pooled evaluation requires both classes')
    pred = p >= threshold
    tp = int(((y == 1) & pred).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    tn = int(((y == 0) & ~pred).sum())
    return dict(n=len(y), prevalence=float(y.mean()), tp=tp, fp=fp, fn=fn, tn=tn,
                f1=2 * tp / max(2 * tp + fp + fn, 1),
                precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                average_precision=float(average_precision_score(y, p)),
                roc_auc=float(roc_auc_score(y, p)), threshold=float(threshold))


def calibration(y, p):
    prec, rec, thresholds = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-15)
    # Among equal F1 use the highest threshold to avoid arbitrary false positives.
    return float(thresholds[np.where(f1 == f1.max())[0][-1]])


def predict(models, data, owner, args, split, use_exchange, shuffled=False):
    steps = range(34, 39) if split == 'val' else range(39, 49)
    ys, ps, node_ids = [], [], []
    for step in steps:
        views = build_views(data, owner, len(models), step, target_step=step)
        remote = exchange(models, views, args.device, shuffled=shuffled, secure=args.secure, shuffle_seed=step)[0] if use_exchange else [None] * len(models)
        for model, view, foreign in zip(models, views, remote):
            ids = view['ids']
            mask = (data.timestep[ids] == step) & getattr(data, f'{split}_mask')[ids]
            if not mask.any():
                continue
            model.eval()
            with torch.no_grad():
                prob = model(view['x'].to(args.device), view['edge_index'].to(args.device), foreign).sigmoid().cpu()
            ys.append(data.y[ids[mask]].numpy())
            ps.append(prob[mask].numpy())
            node_ids.append(ids[mask].numpy())
    order = np.argsort(np.concatenate(node_ids))
    ids = np.concatenate(node_ids)[order]
    if len(ids) != len(np.unique(ids)):
        raise AssertionError('A held-out node was evaluated more than once')
    if not np.array_equal(ids, torch.where(getattr(data, f'{split}_mask'))[0].numpy()):
        raise AssertionError('Evaluation support differs from the declared split')
    return np.concatenate(ys)[order], np.concatenate(ps)[order], ids


def cpu_states(models):
    return [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for m in models]


def run_method(data, owner, seed, strategy, method, args, root):
    seed_all(seed)
    nclients = 1 if method == 'centralized' else args.clients
    actual_owner = torch.zeros_like(owner) if method == 'centralized' else owner
    views = build_views(data, actual_owner, nclients, 33)
    base = CrossSiloSAGE(data.num_features, args.hidden, args.dropout)
    models = [copy.deepcopy(base).to(args.device) for _ in range(nclients)]
    initial_hash = digest_tensor(torch.cat([t.flatten() for t in base.state_dict().values()]))
    use_exchange = method in ['fedgraph', 'shuffled']
    counts = [int(v['train_mask'].sum()) for v in views]
    if min(counts) == 0:
        raise ValueError('Client has no supervised training data')
    for view in views:
        if torch.unique(view['y'][view['train_mask']]).numel() != 2:
            raise ValueError('Every training client must contain both classes')
    history, best, best_states, best_val = [], -1.0, None, None
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for round_index in range(args.rounds):
        remote, comm = exchange(models, views, args.device, method == 'shuffled', args.secure, seed + round_index) if use_exchange else (
            [None] * nclients, dict(transmitted_nodes=0, transmitted_bytes=0, received_edges=0))
        losses = []
        for client, (model, view, foreign) in enumerate(zip(models, views, remote)):
            v = put_view(view, args.device)
            model.train()
            # Reset local Adam moments at each synchronization for all methods.
            opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            labels = v['y'][v['train_mask']].float()
            positive_weight = (labels == 0).sum() / (labels == 1).sum().clamp_min(1)
            for _ in range(args.local_epochs):
                opt.zero_grad()
                logits = model(v['x'], v['edge_index'], foreign)
                loss = F.binary_cross_entropy_with_logits(logits[v['train_mask']], labels, pos_weight=positive_weight)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))
        if method not in ['local', 'centralized']:
            states = cpu_states(models)
            aggregate = {key: sum(state[key] * (n / sum(counts)) for state, n in zip(states, counts))
                         for key in states[0]}
            for model in models:
                model.load_state_dict(aggregate)
        y_val, p_val, ids_val = predict(models, data, actual_owner, args, 'val', use_exchange, method == 'shuffled')
        val_metrics = metric_dict(y_val, p_val)
        entry = dict(round=round_index + 1, loss=float(np.mean(losses)), validation=val_metrics, **comm)
        history.append(entry)
        if val_metrics['average_precision'] > best:
            best = val_metrics['average_precision']
            best_states = cpu_states(models)
            best_val = (y_val.copy(), p_val.copy(), ids_val.copy())
            best_round = round_index + 1
        print(f'{strategy} seed={seed} {method} round={round_index+1}/{args.rounds} '
              f'loss={entry["loss"]:.4f} val_AP={val_metrics["average_precision"]:.4f} '
              f'foreign_edges={comm["received_edges"]}', flush=True)
    for model, state in zip(models, best_states):
        model.load_state_dict(state)
    threshold = calibration(best_val[0], best_val[1])
    outdir = root / f'{strategy}-s{seed}-{method}'
    outdir.mkdir(exist_ok=True)
    torch.save(dict(protocol=PROTOCOL, states=best_states, selected_round=best_round,
                    threshold=threshold, initial_state_hash=initial_hash), outdir / 'checkpoint.pt')
    np.savez_compressed(outdir / 'validation_predictions.npz', y=best_val[0], probability=best_val[1], ids=best_val[2])
    result = dict(seed=seed, strategy=strategy, method=method, selected_round=best_round,
                  selection_metric='pooled_validation_average_precision',
                  initial_state_hash=initial_hash, ownership_hash=digest_tensor(owner),
                  validation=metric_dict(best_val[0], best_val[1], threshold),
                  history=history, test=None,
                  train_cross_edge_ratio=float(((owner[data.edge_index[0]] != owner[data.edge_index[1]]) &
                        (data.timestep[data.edge_index] < 34).all(0)).sum() /
                        max(int((data.timestep[data.edge_index] < 34).all(0).sum()), 1)),
                  training_validation_seconds=time.perf_counter() - start,
                  peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                  communication_scope='unique sender embeddings; excludes model updates, IDs, and network framing',
                  communication_bytes=sum(h['transmitted_bytes'] for h in history))
    result['support'] = {}
    for split in ['train', 'val', 'test']:
        mask = getattr(data, f'{split}_mask')
        observed = ((data.timestep[data.edge_index] < 34).all(0) if split == 'train' else
                    ((data.timestep[data.edge_index] >= (34 if split == 'val' else 39)) &
                     (data.timestep[data.edge_index] < (39 if split == 'val' else 49))).all(0))
        result['support'][split] = dict(
            cut_edge_ratio=float((owner[data.edge_index[0]][observed] != owner[data.edge_index[1]][observed]).float().mean()),
            clients=[dict(negative=int((mask & (owner == k) & (data.y == 0)).sum()),
                          positive=int((mask & (owner == k) & (data.y == 1)).sum())) for k in range(args.clients)])
    if args.phase == 'main':
        y_test, p_test, ids_test = predict(models, data, actual_owner, args, 'test', use_exchange, method == 'shuffled')
        result['test'] = metric_dict(y_test, p_test, threshold)
        np.savez_compressed(outdir / 'test_predictions.npz', y=y_test, probability=p_test, ids=ids_test)
    (outdir / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    del models
    torch.cuda.empty_cache()
    return result


def paired_summary(results, split):
    comparisons = {}
    for strategy in sorted({r['strategy'] for r in results}):
        by_method = {m: {r['seed']: r for r in results if r['strategy'] == strategy and r['method'] == m}
                     for m in ['fedavg', 'fedgraph']}
        seeds = sorted(set(by_method['fedavg']) & set(by_method['fedgraph']))
        if not seeds:
            continue
        comparisons[strategy] = {}
        for metric in ['average_precision', 'f1']:
            diff = np.array([by_method['fedgraph'][s][split][metric] - by_method['fedavg'][s][split][metric] for s in seeds])
            n = len(diff)
            mean = float(diff.mean())
            half = float(stats.t.ppf(.975, n - 1) * diff.std(ddof=1) / np.sqrt(n)) if n > 1 else None
            p = float(stats.ttest_1samp(diff, 0).pvalue) if n > 1 and diff.std() > 0 else None
            comparisons[strategy][metric] = dict(seeds=seeds, paired_differences=diff.tolist(), mean_difference=mean,
                                                 ci95=[mean-half, mean+half] if half is not None else None,
                                                 unadjusted_p=p, confirmatory=False)
    return comparisons


def verify_main_gate(args, gate, record):
    """Bind the reviewed pilot to data, protocol and the exact main recipe."""
    if gate.get('protocol') != PROTOCOL or gate.get('raw_data_sha256') != record['raw_data_sha256']:
        raise RuntimeError('Main gate protocol or raw dataset mismatch')
    locked = gate['main_config']
    for key, value in locked.items():
        if key != 'seeds' and getattr(args, key) != value:
            raise RuntimeError(f'Main configuration differs from preregistration: {key}')
    if not set(args.seeds).issubset(set(locked['seeds'])) or len(args.seeds) != len(set(args.seeds)):
        raise RuntimeError('Main seed list differs from preregistration')
    preprocessing_hash = hashlib.sha256((Path(args.output) / 'preprocessing.npz').read_bytes()).hexdigest()
    # Compare numerical transform bytes, not ZIP timestamps in the NPZ container.
    prep = np.load(Path(args.output) / 'preprocessing.npz')
    numeric_hash = hashlib.sha256(prep['mean'].tobytes() + prep['scale'].tobytes()).hexdigest()
    if numeric_hash != gate['preprocessing_numeric_sha256']:
        raise RuntimeError('Main preprocessing differs from the reviewed pilot')
    return preprocessing_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['pilot', 'main'], default='pilot')
    p.add_argument('--data-root', default='data/elliptic')
    p.add_argument('--output', required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 123])
    p.add_argument('--strategies', nargs='+', choices=['random', 'louvain', 'metis'], default=['louvain', 'random'])
    p.add_argument('--methods', nargs='+', choices=['local', 'fedavg', 'fedgraph', 'shuffled', 'centralized'], default=['local', 'fedavg', 'fedgraph', 'centralized'])
    p.add_argument('--rounds', type=int, default=8)
    p.add_argument('--local-epochs', type=int, default=2)
    p.add_argument('--clients', type=int, default=3)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--feature-set', choices=['local93', 'all165'], default='local93')
    p.add_argument('--partition-seed', type=int, default=20261002)
    p.add_argument('--dropout', type=float, default=.3)
    p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--weight-decay', type=float, default=.0005)
    p.add_argument('--secure', action='store_true')
    p.add_argument('--gate', help='Verified pilot gate JSON required before main phase')
    args = p.parse_args()
    args.device = 'cuda'
    if not torch.cuda.is_available():
        raise RuntimeError('GPU experiment requires CUDA; no CPU fallback')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    manifest = source_manifest()
    if args.phase == 'main':
        if not args.gate:
            raise RuntimeError('Main phase requires a passed reviewed two-GPU pilot gate')
        gate = json.loads(Path(args.gate).read_text())
        if not gate.get('passed') or gate.get('source_sha256') != manifest:
            raise RuntimeError('Pilot gate failed or source has changed since gate')
    loader = EllipticDataset(args.data_root)
    data = loader.load()
    data.no_cross_time_edges = bool((data.timestep[data.edge_index[0]] == data.timestep[data.edge_index[1]]).all())
    if args.feature_set == 'local93':
        data.x = data.x[:, :93].clone()
    train_nodes = data.timestep < 34
    # Fit only on observed training-period transactions, including unlabeled nodes.
    mean = data.x[train_nodes].mean(0)
    scale = data.x[train_nodes].std(0).clamp_min(1e-6)
    data.x = (data.x - mean) / scale
    np.savez_compressed(root / 'preprocessing.npz', mean=mean.numpy(), scale=scale.numpy())
    raw_hashes = {}
    for path in sorted(loader.raw_dir.glob('*.csv')):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(chunk)
        raw_hashes[path.name] = h.hexdigest()
    config = vars(args).copy()
    record = dict(protocol=PROTOCOL, config=config, environment=environment(), source_sha256=manifest,
                  raw_data_sha256=raw_hashes, dataset=loader.get_statistics(),
                  split=dict(train='1..34', validation='35..39', test='40..49'),
                  no_cross_time_edges=data.no_cross_time_edges,
                  ownership_protocol='per-observed-timestep community packing with trusted topology coordinator; label-free snapshot simulation',
                  exchange_protocol='directed source first-layer detached state; fresh synchronous round snapshot; second-layer degree-correct mean',
                  normalization='shared public train-period mean/std oracle; per-node LayerNorm; no preprocessing privacy claim',
                  secure_mode='ML-KEM-512 + AES-GCM local crypto simulation' if args.secure else 'plaintext local exchange simulation',
                  results=[])
    (root / 'manifest.json').write_text(json.dumps(record, indent=2, allow_nan=False))
    if args.phase == 'main':
        verify_main_gate(args, gate, record)
        record['pilot_gate_sha256'] = hashlib.sha256(Path(args.gate).read_bytes()).hexdigest()
    for strategy in args.strategies:
        for seed in args.seeds:
            owner = make_ownership(data, strategy, args.partition_seed, args.clients)
            if args.phase == 'main' and digest_tensor(owner) != gate['ownership_sha256'][strategy]:
                raise RuntimeError('Main ownership differs from the reviewed pilot')
            np.save(root / f'ownership-{strategy}-s{seed}.npy', owner.numpy())
            for method in args.methods:
                result = run_method(data, owner, seed, strategy, method, args, root)
                record['results'].append(result)
                (root / 'manifest.json').write_text(json.dumps(record, indent=2, allow_nan=False))
    record['paired_comparisons'] = paired_summary(record['results'], 'test' if args.phase == 'main' else 'validation')
    record['status'] = 'completed'
    (root / 'manifest.json').write_text(json.dumps(record, indent=2, allow_nan=False))
    print('COMPLETED', root, flush=True)


if __name__ == '__main__':
    main()
