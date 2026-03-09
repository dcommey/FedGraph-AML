"""
FedSage+ Baseline Implementation

Reference: "FedSage+: Federated Learning for Graph Neural Networks with Missing Neighbors"
This is the DIRECT COMPETITOR to FedGraph-VASP.

Key Mechanism:
- Instead of exchanging embeddings (FedGraph-VASP), FedSage+ *generates* (hallucinates)
  features for missing cross-silo neighbors using a trained Generator (Imputer).

Components:
1. NeighborGen: Predicts features of missing neighbors.
2. Classifier: Standard GNN classifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv

class NeighborGenerator(nn.Module):
    """
    Generator GNN (Imputer) for FedSage+.
    Takes a node and predicts the aggregate feature vector of its missing neighbors.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        self.convs = nn.ModuleList()
        # Encoder to latent representation
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Decoder to reconstruct neighbor features
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
        
        # Predict neighbor features
        return self.decoder(x)

class FedSagePlus(nn.Module):
    """
    FedSage+ Model: Classifier + Neighbor Generator.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        
        # 1. Main Classifier
        self.classifier = nn.ModuleList([
            SAGEConv(input_dim, hidden_dim),
            SAGEConv(hidden_dim, hidden_dim)
        ])
        self.final_proj = nn.Linear(hidden_dim, num_classes)
        
        # 2. Missing Neighbor Generator (Imputer)
        # Predicts the *mean feature vector* of missing neighbors
        self.generator = NeighborGenerator(input_dim, hidden_dim, input_dim)
        
    def forward_classifier(self, x, edge_index):
        for conv in self.classifier:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
        return self.final_proj(x)
        
    def forward(self, x, edge_index, missing_indices=None):
        """
        Full forward pass. 
        If missing_indices are provided, generates 'hallucinated' neighbors
        and augments the graph before classification.
        """
        if missing_indices is not None and len(missing_indices) > 0:
            # 1. Generate missing neighbors
            # In a real impl, we'd add new nodes. 
            # Simplified: Add generated features to aggregation?
            # Or augment graph structure.
            
            # Strategy:
            # 1. Generate features for missing neighbors of nodes in 'missing_indices'
            # 2. Add these "hallucinated" nodes to the graph
            # 3. Add edges connecting original nodes to these new nodes
            
            # Predict features for ALL nodes (simplified) or just boundary nodes
            gen_feats = self.generator(x, edge_index)
            
            # We specifically care about generating neighbors for the nodes in missing_indices
            # For simplicity in this baseline, we assume 1 missing neighbor per boundary node
            # The generated feature is the EXPECTED VALUE of the missing neighbor.
            
            new_feats = gen_feats[missing_indices]
            
            # Augment features
            x_aug = torch.cat([x, new_feats], dim=0)
            
            # Augment edges
            # Connect original boundary node i to its new generated neighbor j
            # Original nodes: 0 to N-1
            # New nodes: N to N + len(missing_indices) - 1
            
            N = x.shape[0]
            new_node_indices = torch.arange(N, N + len(missing_indices), device=x.device)
            
            # Edges: (missing_indices[k], new_node_indices[k])
            # We make it undirected for GCN/SAGE usually, or directed from neighbor to target
            
            row = torch.cat([missing_indices, new_node_indices])
            col = torch.cat([new_node_indices, missing_indices])
            new_edges = torch.stack([row, col], dim=0)
            
            edge_index_aug = torch.cat([edge_index, new_edges], dim=1)
            
            # Use augmented graph for classification
            return self.forward_classifier(x_aug, edge_index_aug)

        return self.forward_classifier(x, edge_index)

    def gen_loss(self, x, edge_index, target_neighbors_feat):
        """
        Loss for the generator: MSE between predicted neighbor features and actual.
        Used during local training by hiding some existing neighbors.
        """
        pred = self.generator(x, edge_index)
        return F.mse_loss(pred, target_neighbors_feat)

