"""Elliptic loader: transaction-ID joins and genuine CSV timestamps.

The raw layout is ID, timestep (1..49), 165 features. Labels are
1=illicit, 2=licit, unknown. PyG is used only to obtain raw files;
its processed cache is never used to infer timestamps or label ordering.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import EllipticBitcoinDataset as PyGElliptic


class EllipticDataset:
    NUM_FEATURES = 165
    NUM_CLASSES = 2
    NUM_TIMESTEPS = 49

    def __init__(self, root='./data/elliptic', use_pyg=True):
        self.root = Path(root)
        self.use_pyg = use_pyg
        self._data = None

    def load(self):
        if self._data is not None:
            return self._data
        names = ['elliptic_txs_features.csv', 'elliptic_txs_edgelist.csv',
                 'elliptic_txs_classes.csv']
        raw = self.root if all((self.root / n).exists() for n in names) else self.root / 'raw'
        if not all((raw / n).exists() for n in names):
            if not self.use_pyg:
                raise FileNotFoundError(f'Elliptic raw CSV files absent from {self.root}')
            PyGElliptic(root=str(self.root))
            raw = self.root / 'raw'
        feat = pd.read_csv(raw / names[0], header=None)
        if feat.shape[1] != self.NUM_FEATURES + 2:
            raise ValueError(f'Expected ID, timestep, 165 features; got {feat.shape[1]} columns')
        ids = pd.Index(feat.iloc[:, 0])
        if ids.has_duplicates:
            raise ValueError('Duplicate transaction IDs')
        steps = feat.iloc[:, 1].to_numpy()
        if not np.all(np.isfinite(steps) & (steps == np.floor(steps)) & (steps >= 1) & (steps <= 49)):
            raise ValueError('Invalid genuine Elliptic timestamps')
        features = feat.iloc[:, 2:].to_numpy(dtype=np.float32)
        if not np.isfinite(features).all():
            raise ValueError('Nonfinite input features')
        edges = pd.read_csv(raw / names[1])
        src = ids.get_indexer(edges.iloc[:, 0])
        dst = ids.get_indexer(edges.iloc[:, 1])
        if (src < 0).any() or (dst < 0).any():
            raise ValueError('Edge references an absent transaction; refusing mismatched endpoints')
        classes = pd.read_csv(raw / names[2], dtype={'class': str})
        if classes.iloc[:, 0].duplicated().any():
            raise ValueError('Duplicate class transaction IDs')
        labels = classes.set_index(classes.columns[0]).iloc[:, 0].reindex(ids)
        if labels.isna().any() or not labels.isin(['1', '2', 'unknown']).all():
            raise ValueError('Missing or invalid transaction labels')
        self._data = Data(
            x=torch.from_numpy(features),
            edge_index=torch.from_numpy(np.stack([src, dst])).long(),
            y=torch.tensor(labels.map({'1': 1, '2': 0, 'unknown': -1}).to_numpy(), dtype=torch.long),
            timestep=torch.tensor(steps, dtype=torch.long) - 1,
            transaction_ids=torch.tensor(ids.to_numpy(), dtype=torch.long),
        )
        self.raw_dir = raw
        self._create_temporal_masks(self._data)
        return self._data

    def _create_temporal_masks(self, data, train_steps=34, val_steps=5, test_steps=10):
        if train_steps + val_steps + test_steps != self.NUM_TIMESTEPS:
            raise ValueError('Temporal split must cover 49 steps')
        labeled = data.y >= 0
        data.train_mask = (data.timestep < train_steps) & labeled
        data.val_mask = (data.timestep >= train_steps) & (data.timestep < train_steps + val_steps) & labeled
        data.test_mask = (data.timestep >= train_steps + val_steps) & labeled
        data.unlabeled_mask = data.y < 0
        data.train_unlabeled_mask = (data.timestep < train_steps) & data.unlabeled_mask

    def get_class_weights(self):
        data = self.load()
        counts = torch.bincount(data.y[data.train_mask], minlength=2).float()
        if (counts == 0).any():
            raise ValueError('Training set requires both classes')
        return counts.sum() / (2 * counts)

    def get_statistics(self):
        d = self.load()
        return dict(num_nodes=d.num_nodes, num_edges=d.num_edges, num_features=d.num_features,
                    num_licit=int((d.y == 0).sum()), num_illicit=int((d.y == 1).sum()),
                    num_unknown=int((d.y < 0).sum()), num_timesteps=49,
                    train_nodes=int(d.train_mask.sum()), val_nodes=int(d.val_mask.sum()),
                    test_nodes=int(d.test_mask.sum()))


def load_elliptic(root='./data/elliptic'):
    return EllipticDataset(root).load()
