"""
Graph Neural Network Models for AML Detection

Implements GraphSAGE and GAT models for node classification
on transaction graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool
from typing import Optional, Tuple


class GraphSAGE(nn.Module):
    """
    GraphSAGE model for node classification.
    
    Uses mean aggregation to learn node representations by
    sampling and aggregating features from neighbors.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 1,
        num_layers: int = 2,
        dropout: float = 0.5
    ):
        """
        Initialize GraphSAGE model.
        
        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            out_channels: Output dimension (1 for binary classification)
            num_layers: Number of SAGE layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        # Classification head
        self.classifier = nn.Linear(hidden_channels, out_channels)
        
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        return_embeddings: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge list [2, num_edges]
            return_embeddings: If True, return node embeddings before classifier
            
        Returns:
            Tuple of (logits, embeddings) where embeddings is None if not requested
        """
        # Message passing layers
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final conv layer
        x = self.convs[-1](x, edge_index)
        embeddings = x if return_embeddings else None
        
        # Classification
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.classifier(x)
        
        return logits.squeeze(-1), embeddings
    
    def get_embeddings(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Get node embeddings without classification."""
        _, embeddings = self.forward(x, edge_index, return_embeddings=True)
        return embeddings


class GAT(nn.Module):
    """
    Graph Attention Network for node classification.
    
    Uses attention mechanism to weight neighbor contributions.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 1,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.5
    ):
        """
        Initialize GAT model.
        
        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            out_channels: Output dimension
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(
            in_channels, 
            hidden_channels // num_heads,
            heads=num_heads,
            dropout=dropout
        ))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(
                hidden_channels,
                hidden_channels // num_heads,
                heads=num_heads,
                dropout=dropout
            ))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layer (single head)
        if num_layers > 1:
            self.convs.append(GATConv(
                hidden_channels,
                hidden_channels,
                heads=1,
                concat=False,
                dropout=dropout
            ))
        
        # Classification head
        self.classifier = nn.Linear(hidden_channels, out_channels)
        
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        return_embeddings: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass."""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        embeddings = x if return_embeddings else None
        
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.classifier(x)
        
        return logits.squeeze(-1), embeddings
    
    def get_embeddings(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Get node embeddings without classification."""
        _, embeddings = self.forward(x, edge_index, return_embeddings=True)
        return embeddings


def create_model(
    model_type: str,
    in_channels: int,
    hidden_channels: int = 128,
    num_layers: int = 2,
    **kwargs
) -> nn.Module:
    """
    Factory function to create GNN models.
    
    Args:
        model_type: "graphsage" or "gat"
        in_channels: Input feature dimension
        hidden_channels: Hidden dimension
        num_layers: Number of layers
        **kwargs: Additional model-specific arguments
        
    Returns:
        Initialized model
    """
    if model_type.lower() == "graphsage":
        return GraphSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=kwargs.get("dropout", 0.5)
        )
    elif model_type.lower() == "gat":
        return GAT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_heads=kwargs.get("num_heads", 4),
            dropout=kwargs.get("dropout", 0.5)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test models
    print("=" * 60)
    print("Testing GNN Models")
    print("=" * 60)
    
    # Create dummy data
    num_nodes = 100
    num_edges = 500
    in_channels = 166
    
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Test GraphSAGE
    print("\nGraphSAGE:")
    sage = create_model("graphsage", in_channels, hidden_channels=128)
    logits, emb = sage(x, edge_index, return_embeddings=True)
    print(f"  Logits shape: {logits.shape}")
    print(f"  Embeddings shape: {emb.shape}")
    print(f"  Parameters: {sum(p.numel() for p in sage.parameters()):,}")
    
    # Test GAT
    print("\nGAT:")
    gat = create_model("gat", in_channels, hidden_channels=128, num_heads=4)
    logits, emb = gat(x, edge_index, return_embeddings=True)
    print(f"  Logits shape: {logits.shape}")
    print(f"  Embeddings shape: {emb.shape}")
    print(f"  Parameters: {sum(p.numel() for p in gat.parameters()):,}")
