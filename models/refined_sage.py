"""Versioned mechanism controls; the v2 historical study stays unchanged.

The full-neighborhood control deliberately exposes remote input features.
It is an information oracle, never a privacy-preserving method.
"""
import torch
from torch.nn import functional as F
from models.cross_silo_sage import CrossSiloSAGE


def incoming_mean(h, edge, foreign=None):
    src, dst = edge
    sums = torch.zeros_like(h)
    degree = h.new_zeros(h.shape[0])
    sums.index_add_(0, dst, h[src])
    degree.index_add_(0, dst, h.new_ones(dst.numel()))
    if foreign is not None:
        foreign_dst, states = foreign
        if states.requires_grad:
            raise ValueError('Foreign states must be detached')
        sums.index_add_(0, foreign_dst, states)
        degree.index_add_(0, foreign_dst, h.new_ones(foreign_dst.numel()))
    return sums / degree.clamp_min(1).unsqueeze(1)


class RefinedSAGE(CrossSiloSAGE):
    def first_layer(self, x, edge_index, raw_foreign=None):
        if raw_foreign is None:
            return super().first_layer(x, edge_index)
        z = self.first.lin_l(incoming_mean(x, edge_index, raw_foreign)) + self.first.lin_r(x)
        return F.relu(self.norm(z))

    def forward(self, x, edge_index, remote=None, raw_foreign=None):
        h = F.dropout(self.first_layer(x, edge_index, raw_foreign), self.dropout, self.training)
        z = self.second.lin_l(incoming_mean(h, edge_index, remote)) + self.second.lin_r(h)
        return self.classifier(F.dropout(F.relu(z), self.dropout, self.training)).squeeze(-1)


class TabularMLP(torch.nn.Module):
    """Conventional graph-free MLP; report its actual trainable capacity."""
    def __init__(self, in_channels, hidden_channels=64, dropout=.3):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.LayerNorm(hidden_channels), torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels,hidden_channels), torch.nn.ReLU(),
            torch.nn.Dropout(dropout),torch.nn.Linear(hidden_channels,1))

    def forward(self, x, edge_index=None, remote=None, raw_foreign=None):
        return self.layers(x).squeeze(-1)
