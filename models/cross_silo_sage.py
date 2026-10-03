"""Version 2: directed cross-edge GraphSAGE with detached round snapshots.

Remote inputs are sender first-layer states routed to local destination nodes.
The second-layer mean includes BOTH local and foreign incoming neighbors.
No same-transaction overlap, synthetic neighbor, or cosine alignment is used.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv


class CrossSiloSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, dropout=0.3):
        super().__init__()
        self.first = SAGEConv(in_channels, hidden_channels)
        self.norm = nn.LayerNorm(hidden_channels)
        self.second = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def first_layer(self, x, edge_index):
        return F.relu(self.norm(self.first(x, edge_index)))

    def forward(self, x, edge_index, remote=None):
        h = F.dropout(self.first_layer(x, edge_index), self.dropout, self.training)
        src, dst = edge_index
        sums = torch.zeros_like(h)
        degree = h.new_zeros(h.shape[0])
        sums.index_add_(0, dst, h[src])
        degree.index_add_(0, dst, h.new_ones(dst.numel()))
        if remote is not None:
            remote_dst, remote_h = remote
            if remote_h.requires_grad:
                raise ValueError('Foreign round snapshots must be detached')
            sums.index_add_(0, remote_dst, remote_h)
            degree.index_add_(0, remote_dst, h.new_ones(remote_dst.numel()))
        mean = sums / degree.clamp_min(1).unsqueeze(1)
        z = self.second.lin_l(mean) + self.second.lin_r(h)
        return self.classifier(F.dropout(F.relu(z), self.dropout, self.training)).squeeze(-1)
