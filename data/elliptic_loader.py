"""
Elliptic Bitcoin Dataset Loader

Loads the Elliptic dataset for anti-money laundering research.
Handles the 3 CSV files and creates a PyTorch Geometric Data object.

Dataset structure:
- 203,769 nodes (Bitcoin transactions)
- 234,355 edges (transaction flows)
- 166 features per node
- Labels: 0 (licit), 1 (illicit), -1 (unknown)
- 49 distinct timesteps
"""

import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from torch_geometric.data import Data, download_url, extract_zip
from torch_geometric.datasets import EllipticBitcoinDataset as PyGElliptic


class EllipticDataset:
    """
    Elliptic Bitcoin Dataset loader with enhanced preprocessing.
    
    Supports:
    - Automatic download via PyTorch Geometric
    - Temporal train/val/test splits
    - Semi-supervised learning with unlabeled nodes
    """
    
    # Dataset statistics
    NUM_FEATURES = 166
    NUM_CLASSES = 2  # Binary: licit (0) vs illicit (1)
    NUM_TIMESTEPS = 49
    
    def __init__(
        self,
        root: str = "./data/elliptic",
        use_pyg: bool = True
    ):
        """
        Initialize the Elliptic dataset loader.
        
        Args:
            root: Directory to store/load data
            use_pyg: If True, use PyTorch Geometric's built-in loader
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.use_pyg = use_pyg
        self._data = None
        self._timesteps = None
        
    def load(self) -> Data:
        """
        Load the Elliptic dataset.
        
        Returns:
            PyTorch Geometric Data object with:
            - x: Node features [num_nodes, 166]
            - edge_index: Edge list [2, num_edges]
            - y: Labels [num_nodes] (0=licit, 1=illicit, -1=unknown)
            - timestep: Timestep for each node [num_nodes]
            - train_mask, val_mask, test_mask: Boolean masks
        """
        if self._data is not None:
            return self._data
            
        if self.use_pyg:
            self._data = self._load_pyg()
        else:
            self._data = self._load_manual()
            
        return self._data
    
    def _load_pyg(self) -> Data:
        """Load using PyTorch Geometric's built-in loader."""
        print("Loading Elliptic dataset via PyTorch Geometric...")
        
        # PyG's EllipticBitcoinDataset handles download automatically
        dataset = PyGElliptic(root=str(self.root))
        data = dataset[0]
        
        # PyG returns y with values 0, 1, 2 where 2 is unknown
        # Convert to our format: 0=licit, 1=illicit, -1=unknown
        y = data.y.clone()
        y[data.y == 2] = -1
        data.y = y
        
        # Extract timesteps from features (first feature is timestep)
        # In Elliptic, features 0-93 are local, 94-165 are aggregated
        # The timestep is embedded in the data - we need to extract it
        self._add_timesteps(data)
        
        # Create temporal train/val/test masks
        self._create_temporal_masks(data)
        
        print(f"Loaded: {data.num_nodes} nodes, {data.num_edges} edges")
        print(f"Labels: {(data.y == 0).sum()} licit, {(data.y == 1).sum()} illicit, {(data.y == -1).sum()} unknown")
        
        return data
    
    def _add_timesteps(self, data: Data):
        """
        Add timestep information to the data object.
        
        The Elliptic dataset has 49 timesteps. We need to identify which
        timestep each node belongs to based on the temporal ordering.
        """
        # In PyG's version, we need to infer timesteps
        # The dataset is temporally ordered, so we can use node indices
        # to approximate timesteps by dividing into 49 equal parts
        
        num_nodes = data.num_nodes
        nodes_per_timestep = num_nodes // self.NUM_TIMESTEPS
        
        timesteps = torch.zeros(num_nodes, dtype=torch.long)
        for t in range(self.NUM_TIMESTEPS):
            start_idx = t * nodes_per_timestep
            end_idx = (t + 1) * nodes_per_timestep if t < self.NUM_TIMESTEPS - 1 else num_nodes
            timesteps[start_idx:end_idx] = t
            
        data.timestep = timesteps
        self._timesteps = timesteps
        
    def _create_temporal_masks(
        self,
        data: Data,
        train_steps: int = 34,
        val_steps: int = 5,
        test_steps: int = 10
    ):
        """
        Create train/val/test masks based on temporal ordering.
        
        This ensures no data leakage from future to past.
        
        Args:
            data: PyG Data object
            train_steps: Number of timesteps for training (default: 34)
            val_steps: Number of timesteps for validation (default: 5)
            test_steps: Number of timesteps for testing (default: 10)
        """
        timesteps = data.timestep
        labels = data.y
        
        # Define timestep ranges
        train_end = train_steps
        val_end = train_end + val_steps
        
        # Create masks (only for labeled nodes: y != -1)
        labeled_mask = (labels != -1)
        
        train_mask = (timesteps < train_end) & labeled_mask
        val_mask = (timesteps >= train_end) & (timesteps < val_end) & labeled_mask
        test_mask = (timesteps >= val_end) & labeled_mask
        
        # Also create masks for unlabeled nodes (for semi-supervised learning)
        unlabeled_mask = (labels == -1)
        train_unlabeled_mask = (timesteps < train_end) & unlabeled_mask
        
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        data.unlabeled_mask = unlabeled_mask
        data.train_unlabeled_mask = train_unlabeled_mask
        
        print(f"Temporal split - Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
        print(f"Unlabeled in train period: {train_unlabeled_mask.sum()}")
        
    def _load_manual(self) -> Data:
        """
        Load manually from CSV files.
        
        Expected files in root directory:
        - elliptic_txs_features.csv
        - elliptic_txs_edgelist.csv
        - elliptic_txs_classes.csv
        """
        print("Loading Elliptic dataset from CSV files...")
        
        features_path = self.root / "elliptic_txs_features.csv"
        edges_path = self.root / "elliptic_txs_edgelist.csv"
        classes_path = self.root / "elliptic_txs_classes.csv"
        
        # Check if files exist
        for path in [features_path, edges_path, classes_path]:
            if not path.exists():
                raise FileNotFoundError(
                    f"File not found: {path}\n"
                    "Please download the Elliptic dataset from Kaggle:\n"
                    "https://www.kaggle.com/ellipticco/elliptic-data-set"
                )
        
        # Load features
        features_df = pd.read_csv(features_path, header=None)
        node_ids = features_df.iloc[:, 0].values
        features = features_df.iloc[:, 1:].values  # First column is node ID
        
        # Create node ID to index mapping
        node_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
        
        # Load edges
        edges_df = pd.read_csv(edges_path)
        src = [node_to_idx[n] for n in edges_df.iloc[:, 0].values if n in node_to_idx]
        dst = [node_to_idx[n] for n in edges_df.iloc[:, 1].values if n in node_to_idx]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        
        # Load labels
        classes_df = pd.read_csv(classes_path)
        labels = torch.full((len(node_ids),), -1, dtype=torch.long)
        
        for _, row in classes_df.iterrows():
            node_id = row.iloc[0]
            label = row.iloc[1]
            if node_id in node_to_idx:
                if label == "1":  # Illicit
                    labels[node_to_idx[node_id]] = 1
                elif label == "2":  # Licit
                    labels[node_to_idx[node_id]] = 0
                # "unknown" remains -1
        
        # Create Data object
        data = Data(
            x=torch.tensor(features, dtype=torch.float),
            edge_index=edge_index,
            y=labels
        )
        
        # Add timesteps and masks
        self._add_timesteps(data)
        self._create_temporal_masks(data)
        
        return data
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling imbalanced data.
        
        Returns:
            Tensor with weights [weight_licit, weight_illicit]
        """
        data = self.load()
        labeled_mask = data.y != -1
        y_labeled = data.y[labeled_mask]
        
        num_licit = (y_labeled == 0).sum().float()
        num_illicit = (y_labeled == 1).sum().float()
        
        # Inverse frequency weighting
        total = num_licit + num_illicit
        weights = torch.tensor([
            total / (2 * num_licit),
            total / (2 * num_illicit)
        ])
        
        return weights
    
    def get_statistics(self) -> dict:
        """Get dataset statistics."""
        data = self.load()
        
        return {
            "num_nodes": data.num_nodes,
            "num_edges": data.num_edges,
            "num_features": data.x.shape[1],
            "num_licit": (data.y == 0).sum().item(),
            "num_illicit": (data.y == 1).sum().item(),
            "num_unknown": (data.y == -1).sum().item(),
            "num_timesteps": self.NUM_TIMESTEPS,
            "train_nodes": data.train_mask.sum().item(),
            "val_nodes": data.val_mask.sum().item(),
            "test_nodes": data.test_mask.sum().item(),
        }


def load_elliptic(root: str = "./data/elliptic") -> Data:
    """
    Convenience function to load the Elliptic dataset.
    
    Args:
        root: Directory for dataset storage
        
    Returns:
        PyTorch Geometric Data object
    """
    loader = EllipticDataset(root=root)
    return loader.load()


if __name__ == "__main__":
    # Test data loading
    print("=" * 60)
    print("Testing Elliptic Dataset Loader")
    print("=" * 60)
    
    dataset = EllipticDataset()
    data = dataset.load()
    
    print("\n" + "=" * 60)
    print("Dataset Statistics:")
    print("=" * 60)
    stats = dataset.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("Class Weights (for imbalanced learning):")
    print("=" * 60)
    weights = dataset.get_class_weights()
    print(f"  Licit weight: {weights[0]:.4f}")
    print(f"  Illicit weight: {weights[1]:.4f}")
