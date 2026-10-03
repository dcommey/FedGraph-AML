"""Validation-only mechanism diagnostic. No held-out test outcomes are read."""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from data.elliptic_loader import EllipticDataset
from experiments.corrected_evaluation import (seed_all, make_ownership, build_views,
    put_view, cpu_states, metric_dict, calibration, environment, digest_tensor)
from models.refined_sage import RefinedSAGE

PROTOCOL = 'fedgraph-refinement-validation-diagnostic-v1'


def routed_exchange(models, views, device, mode='current', seed=0, fraction=1.0):
    """Vectorized exact routing, with unique-sender and receiver-fanout byte counts."""
    if mode == 'oracle' and fraction != 1.0:
        raise ValueError('The full-neighborhood oracle requires fraction=1')
    n = max(int(v['ids'].max()) for v in views) + 1
    lookup = torch.zeros((n, models[0].norm.normalized_shape[0]), device=device)
    owner = torch.full((n,), -1, dtype=torch.long, device=device)
    raw = torch.zeros((n, views[0]['x'].shape[1]), device=device)
    for k, view in enumerate(views):
        ids = view['ids'].to(device)
        raw[ids] = view['x'].to(device)
        owner[ids] = k
    oracle_raw = []
    for k, view in enumerate(views):
        src, dst = view['remote_src'].to(device), view['remote_dst'].to(device)
        assert bool((owner[src] >= 0).all()) and bool((owner[src] != k).all())
        oracle_raw.append((dst, raw[src].detach()) if mode == 'oracle' else None)
    for model, view, raw_foreign in zip(models, views, oracle_raw):
        model.eval()
        with torch.no_grad():
            lookup[view['ids'].to(device)] = model.first_layer(
                view['x'].to(device), view['edge_index'].to(device), raw_foreign)
    # One replacement per unique source, shared by every receiver/edge. Preserve
    # sender ownership and snapshot when temporal metadata is available.
    replacement = torch.arange(n, device=device)
    if mode == 'shuffled':
        needed = torch.cat([v['remote_src'].to(device) for v in views]).unique()
        gen = torch.Generator(device=device).manual_seed(seed)
        for k, view in enumerate(views):
            ids = view['ids'].to(device)
            times = view.get('timestep', torch.zeros(len(ids), dtype=torch.long)).to(device)
            for step in times.unique():
                candidates = ids[(times == step) & torch.isin(ids, needed)]
                replacement[candidates] = candidates[torch.randperm(len(candidates), generator=gen, device=device)]
    remote, selected, fanout_nodes, edge_count = [], [], 0, 0
    availability = torch.rand(n, generator=torch.Generator(device=device).manual_seed(20261006), device=device)
    for k, view in enumerate(views):
        src, dst = view['remote_src'].to(device), view['remote_dst'].to(device)
        # Nested, stable source-ID thinning: same graph and ownership at all fractions.
        keep = availability[src] < fraction
        src, dst = src[keep], dst[keep]
        states = lookup[replacement[src]].detach()
        remote.append((dst, states) if len(src) else None)
        selected.append(src)
        fanout_nodes += int(src.unique().numel())
        edge_count += len(src)
    unique = int(torch.cat(selected).unique().numel())
    dim = lookup.shape[1]
    return remote, oracle_raw, dict(unique_sender_bytes=unique * dim * 4,
        receiver_fanout_bytes=fanout_nodes * dim * 4, received_edges=edge_count,
        oracle_raw_fanout_bytes=fanout_nodes * raw.shape[1] * 4 if mode == 'oracle' else 0)


def validation_predict(models, data, owner, device, method, fraction=1.0):
    ys, ps, ids_all = [], [], []
    for step in range(34, 39):
        views = build_views(data, owner, len(models), step, target_step=step)
        for view in views:
            view['timestep'] = data.timestep[view['ids']]
        mode = 'oracle' if method == 'oracle' else ('shuffled' if method == 'shuffled' else 'current')
        if method == 'fedavg':
            remote, raw = [None] * len(models), [None] * len(models)
        else:
            remote, raw, _ = routed_exchange(models, views, device, mode, step, fraction)
        for model, view, foreign, first in zip(models, views, remote, raw):
            ids = view['ids']
            mask = data.val_mask[ids] & (data.timestep[ids] == step)
            if mask.any():
                model.eval()
                with torch.no_grad():
                    p = model(view['x'].to(device), view['edge_index'].to(device), foreign, first).sigmoid().cpu()
                ys.append(data.y[ids[mask]].numpy())
                ps.append(p[mask].numpy())
                ids_all.append(ids[mask].numpy())
    order = np.argsort(np.concatenate(ids_all))
    ids = np.concatenate(ids_all)[order]
    assert np.array_equal(ids, torch.where(data.val_mask)[0].numpy())
    return np.concatenate(ys)[order], np.concatenate(ps)[order], ids


def run(data, owner, seed, strategy, method, args, root):
    seed_all(seed)
    base = RefinedSAGE(data.num_features, args.hidden, args.dropout)
    models = [copy.deepcopy(base).to(args.device) for _ in range(args.clients)]
    views = [put_view(v, args.device) for v in build_views(data, owner, args.clients, 33)]
    for view in views:
        view['timestep'] = data.timestep[view['ids'].cpu()].to(args.device)
    counts = [int(v['train_mask'].sum()) for v in views]
    history, best, chosen = [], -1, None
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    mode = 'oracle' if method == 'oracle' else ('shuffled' if method == 'shuffled' else 'current')
    for rnd in range(args.rounds):
        opt = [torch.optim.Adam(m.parameters(), lr=args.lr, weight_decay=args.weight_decay) for m in models]
        remote, raw = [None] * args.clients, [None] * args.clients
        comm = dict(unique_sender_bytes=0, receiver_fanout_bytes=0, received_edges=0, oracle_raw_fanout_bytes=0)
        for epoch in range(args.local_epochs):
            refresh = method != 'fedavg' and (epoch == 0 or method in ['fresh', 'oracle'])
            if refresh:
                remote, raw, c = routed_exchange(models, views, args.device, mode, seed + rnd, args.fraction)
                for key in comm:
                    comm[key] += c[key]
            for m, v, o, foreign, first in zip(models, views, opt, remote, raw):
                m.train()
                o.zero_grad()
                y = v['y'][v['train_mask']].float()
                weight = (y == 0).sum() / (y == 1).sum().clamp_min(1)
                loss = F.binary_cross_entropy_with_logits(m(v['x'], v['edge_index'], foreign, first)[v['train_mask']], y, pos_weight=weight)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss')
                loss.backward()
                o.step()
        states = cpu_states(models)
        agg = {key: sum(s[key] * (n / sum(counts)) for s, n in zip(states, counts)) for key in states[0]}
        for m in models:
            m.load_state_dict(agg)
        y, p, ids = validation_predict(models, data, owner, args.device, method, args.fraction)
        val = metric_dict(y, p)
        history.append(dict(round=rnd + 1, validation=val, **comm))
        if val['average_precision'] > best:
            best, chosen, best_states = val['average_precision'], (y.copy(), p.copy(), ids.copy()), cpu_states(models)
            best_round = rnd + 1
        print(f'{strategy} seed={seed} {method} round={rnd+1} val_AP={val["average_precision"]:.4f}', flush=True)
    y, p, ids = chosen
    threshold = calibration(y, p)
    out = root / f'{strategy}-s{seed}-{method}'
    out.mkdir()
    np.savez_compressed(out / 'validation_predictions.npz', y=y, probability=p, ids=ids)
    torch.save(dict(states=best_states, selected_round=best_round, threshold=threshold), out / 'checkpoint.pt')
    record = dict(method=method, seed=seed, strategy=strategy, ownership_hash=digest_tensor(owner),
        selected_round=best_round, validation=metric_dict(y, p, threshold), history=history,
        test=None, seconds=time.perf_counter()-start, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        initial_state_hash=digest_tensor(torch.cat([v.flatten() for v in base.state_dict().values()])))
    (out / 'result.json').write_text(json.dumps(record, indent=2, allow_nan=False))
    del models
    torch.cuda.empty_cache()
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--data-root', default='data/elliptic')
    ap.add_argument('--methods', nargs='+', default=['fedavg', 'current', 'fresh', 'oracle', 'shuffled'])
    ap.add_argument('--strategies', nargs='+', default=['random', 'louvain'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 123])
    ap.add_argument('--partition-seed', type=int, default=20261003)
    ap.add_argument('--rounds', type=int, default=10)
    ap.add_argument('--local-epochs', type=int, default=2)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--clients', type=int, default=3)
    ap.add_argument('--lr', type=float, default=.003)
    ap.add_argument('--weight-decay', type=float, default=.0005)
    ap.add_argument('--dropout', type=float, default=.3)
    ap.add_argument('--fraction', type=float, default=1.0)
    args = ap.parse_args()
    args.device = 'cuda'
    assert torch.cuda.is_available()
    assert set(args.methods) <= {'fedavg', 'current', 'fresh', 'oracle', 'shuffled'}
    assert 0 <= args.fraction <= 1
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    loader = EllipticDataset(args.data_root)
    data = loader.load()
    data.no_cross_time_edges = bool((data.timestep[data.edge_index[0]] == data.timestep[data.edge_index[1]]).all())
    data.x = data.x[:, :93].clone()
    train_nodes = data.timestep < 34
    mean, scale = data.x[train_nodes].mean(0), data.x[train_nodes].std(0).clamp_min(1e-6)
    data.x = (data.x-mean)/scale
    np.savez_compressed(root / 'preprocessing.npz', mean=mean.numpy(), scale=scale.numpy())
    paths = ['models/refined_sage.py', 'experiments/refinement_diagnostic.py', 'experiments/corrected_evaluation.py', 'data/elliptic_loader.py', 'data/partitioner.py']
    manifest = dict(protocol=PROTOCOL, config=vars(args), environment=environment(),
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        dataset=loader.get_statistics(), test_evaluated=False, results=[])
    for strategy in args.strategies:
        owner = make_ownership(data, strategy, args.partition_seed, args.clients)
        np.save(root/f'ownership-{strategy}.npy', owner.numpy())
        for seed in args.seeds:
            for method in args.methods:
                manifest['results'].append(run(data, owner, seed, strategy, method, args, root))
                (root/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
    manifest['status'] = 'completed'
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
