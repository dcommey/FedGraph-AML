"""
Unit tests for federated learning components.
"""

import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestGNNModels:
    """Tests for GNN models."""
    
    def test_graphsage_forward(self):
        """Test GraphSAGE forward pass."""
        from models.gnn import create_model
        
        model = create_model("graphsage", in_channels=166, hidden_channels=64)
        
        x = torch.randn(100, 166)
        edge_index = torch.randint(0, 100, (2, 300))
        
        logits, embeddings = model(x, edge_index, return_embeddings=True)
        
        assert logits.shape == (100,)
        assert embeddings.shape == (100, 64)
    
    def test_gat_forward(self):
        """Test GAT forward pass."""
        from models.gnn import create_model
        
        model = create_model("gat", in_channels=166, hidden_channels=64)
        
        x = torch.randn(100, 166)
        edge_index = torch.randint(0, 100, (2, 300))
        
        logits, embeddings = model(x, edge_index, return_embeddings=True)
        
        assert logits.shape == (100,)


class TestBoundaryExchange:
    """Tests for boundary embedding exchange."""
    
    def test_psi_intersection(self):
        """Test PSI intersection computation."""
        from federated.boundary_exchange import PrivateSetIntersection
        
        psi = PrivateSetIntersection()
        
        set_a = {1, 2, 3, 4, 5}
        set_b = {4, 5, 6, 7, 8}
        
        intersection = psi.find_intersection(set_a, set_b)
        
        assert intersection == {4, 5}
    
    def test_embedding_buffer(self):
        """Test embedding buffer operations."""
        from federated.boundary_exchange import BoundaryEmbeddingBuffer
        
        buffer = BoundaryEmbeddingBuffer(embedding_dim=64)
        
        # Add embeddings
        indices = torch.tensor([10, 20, 30])
        embeddings = torch.randn(3, 64)
        buffer.update(0, indices, embeddings)
        
        # Retrieve
        query = torch.tensor([10, 20, 40])
        found, embs = buffer.get_embeddings(query)
        
        assert len(found) == 2  # 10 and 20 found, 40 not
        assert embs.shape == (2, 64)
    
    def test_cross_silo_aggregator(self):
        """Test cross-silo embedding aggregation."""
        from federated.boundary_exchange import CrossSiloAggregator
        
        aggregator = CrossSiloAggregator(embedding_dim=64, aggregation="mean")
        
        local_emb = torch.randn(50, 64)
        foreign_emb = torch.randn(5, 64)
        local_indices = torch.tensor([5, 10, 15, 20, 25])
        
        updated = aggregator(local_emb, foreign_emb, local_indices)
        
        assert updated.shape == (50, 64)


class TestFederatedClient:
    """Tests for federated client."""
    
    @pytest.fixture
    def sample_client_data(self):
        """Create sample client data."""
        from torch_geometric.data import Data
        
        num_nodes = 100
        x = torch.randn(num_nodes, 166)
        edge_index = torch.randint(0, num_nodes, (2, 300))
        y = torch.randint(0, 2, (num_nodes,))
        
        # Add masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[:60] = True
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask[60:80] = True
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask[80:] = True
        
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )
        data.global_node_indices = torch.arange(num_nodes)
        data.boundary_local_indices = torch.tensor([0, 5, 10, 15])
        
        return data
    
    def test_client_training(self, sample_client_data):
        """Test client local training."""
        from models.gnn import create_model
        from federated.client import FederatedClient
        
        model = create_model("graphsage", in_channels=166, hidden_channels=64)
        
        client = FederatedClient(
            client_id=0,
            model=model,
            local_data=sample_client_data,
            device="cpu"
        )
        
        metrics = client.train_epoch()
        
        assert 'loss' in metrics
        assert metrics['loss'] >= 0
    
    def test_client_boundary_extraction(self, sample_client_data):
        """Test boundary embedding extraction."""
        from models.gnn import create_model
        from federated.client import FederatedClient
        
        model = create_model("graphsage", in_channels=166, hidden_channels=64)
        
        client = FederatedClient(
            client_id=0,
            model=model,
            local_data=sample_client_data,
            device="cpu"
        )
        
        # Train one epoch to populate embeddings
        client.train_epoch()
        
        # Get boundary embeddings
        result = client.get_boundary_embeddings()
        
        assert result is not None
        indices, embeddings = result
        assert len(indices) == 4  # 4 boundary nodes


class TestFederatedServer:
    """Tests for federated server."""
    
    def test_weight_aggregation(self):
        """Test FedAvg weight aggregation."""
        from models.gnn import create_model
        from federated.server import FederatedServer
        
        model = create_model("graphsage", in_channels=166, hidden_channels=64)
        server = FederatedServer(model)
        
        # Simulate client weights
        client_weights = [
            {k: v + 0.1 * torch.randn_like(v) for k, v in model.state_dict().items()}
            for _ in range(3)
        ]
        
        server.aggregate_weights(client_weights)
        
        assert server.round_num == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
