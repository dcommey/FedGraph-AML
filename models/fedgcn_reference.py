"""Pinned-author FedGCN model, with an explicit temporal binary-task adapter.

Upstream model source is loaded unmodified; orchestration ports its two-hop
halo, uniform averaging and SGD. Graph normalization/cache is refreshed for
each distinct observed snapshot. This is not its homomorphic-encryption mode.
"""
import importlib.util
from pathlib import Path
import torch

UPSTREAM_COMMIT = '378438d0a5dbfa0f1cf859a3d6f3aaa4cff232a3'


class FedGCNReference(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64, dropout=.3):
        super().__init__()
        path = Path(__file__).resolve().parents[1]/'external/FedGCN/src/gnn_models.py'
        spec = importlib.util.spec_from_file_location('pinned_fedgcn_models', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.network = module.GCN(in_channels,hidden_channels,2,dropout,2)

    def forward(self, x, edge_index, remote=None, raw_foreign=None):
        # Upstream static-graph cache would leak the preceding snapshot here.
        for conv in self.network.convs:
            conv._cached_edge_index = None
            conv._cached_adj_t = None
        logp = self.network(x, edge_index)
        return logp[:,1]-logp[:,0]
