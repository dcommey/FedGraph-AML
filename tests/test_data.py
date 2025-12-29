"""
Unit tests for data loading and partitioning.
"""

import sys
from pathlib import Path
import pytest
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestEllipticLoader:
    """Tests for Elliptic dataset loading."""
    
    def test_dataset_loads(self):
        """Test that dataset loads without errors."""
        from data.elliptic_loader import EllipticDataset
        
        dataset = EllipticDataset()
        data = dataset.load()
        
        assert data is not None
        assert hasattr(data, 'x')
        assert hasattr(data, 'edge_index')
        assert hasattr(data, 'y')
    
    def test_dataset_shape(self):
        """Test that dataset has expected shape."""
        from data.elliptic_loader import EllipticDataset
        
        dataset = EllipticDataset()
        data = dataset.load()
        
        # Elliptic has ~200k nodes, 166 features
        assert data.num_nodes > 100000
        assert data.x.shape[1] == 166
        assert data.num_edges > 0
    
    def test_temporal_masks(self):
        """Test that temporal masks are created correctly."""
        from data.elliptic_loader import EllipticDataset
        
        dataset = EllipticDataset()
        data = dataset.load()
        
        assert hasattr(data, 'train_mask')
        assert hasattr(data, 'val_mask')
        assert hasattr(data, 'test_mask')
        
        # Masks should not overlap
        assert (data.train_mask & data.val_mask).sum() == 0
        assert (data.train_mask & data.test_mask).sum() == 0
        assert (data.val_mask & data.test_mask).sum() == 0
    
    def test_class_weights(self):
        """Test class weight computation."""
        from data.elliptic_loader import EllipticDataset
        
        dataset = EllipticDataset()
        weights = dataset.get_class_weights()
        
        assert len(weights) == 2
        # Illicit should have higher weight (minority class)
        assert weights[1] > weights[0]


class TestPartitioner:
    """Tests for graph partitioning."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample graph data."""
        from data.elliptic_loader import EllipticDataset
        dataset = EllipticDataset()
        return dataset.load()
    
    def test_temporal_partition(self, sample_data):
        """Test temporal partitioning."""
        from data.partitioner import GraphPartitioner
        
        partitioner = GraphPartitioner(num_clients=3, strategy="temporal")
        silos, stats = partitioner.partition(sample_data)
        
        assert len(silos) == 3
        assert stats['num_clients'] == 3
        assert sum(s.num_nodes for s in silos) == sample_data.num_nodes
    
    def test_random_partition(self, sample_data):
        """Test random partitioning."""
        from data.partitioner import GraphPartitioner
        
        partitioner = GraphPartitioner(num_clients=5, strategy="random")
        silos, stats = partitioner.partition(sample_data)
        
        assert len(silos) == 5
        assert stats['cross_edge_ratio'] > 0  # Should have some cross-silo edges
    
    def test_boundary_nodes_identified(self, sample_data):
        """Test that boundary nodes are correctly identified."""
        from data.partitioner import GraphPartitioner
        
        partitioner = GraphPartitioner(num_clients=3, strategy="temporal")
        silos, stats = partitioner.partition(sample_data)
        
        # Each silo should have some boundary nodes
        for silo in silos:
            assert silo.num_boundary_nodes > 0
    
    def test_silo_subgraph_extraction(self, sample_data):
        """Test subgraph extraction for a silo."""
        from data.partitioner import GraphPartitioner
        
        partitioner = GraphPartitioner(num_clients=3, strategy="temporal")
        silos, _ = partitioner.partition(sample_data)
        
        local_data = partitioner.get_silo_subgraph(sample_data, silos[0])
        
        assert local_data.num_nodes == silos[0].num_nodes
        assert hasattr(local_data, 'global_node_indices')
        assert hasattr(local_data, 'boundary_local_indices')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
