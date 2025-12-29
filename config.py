"""
FedGraph-AML Configuration

Central configuration for all experiments and hyperparameters.
"""

import torch
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class DataConfig:
    """Data loading and partitioning configuration."""
    
    # Dataset paths
    data_root: Path = Path("./data/elliptic")
    
    # Partitioning
    num_clients: int = 3
    partition_strategy: str = "temporal"  # "temporal", "metis", "random"
    
    # Train/val/test split (by timestep for temporal consistency)
    # Elliptic has 49 timesteps; use first 34 for train, 5 for val, 10 for test
    train_timesteps: int = 34
    val_timesteps: int = 5
    test_timesteps: int = 10
    
    # Semi-supervised settings
    use_unlabeled: bool = True
    unlabeled_weight: float = 0.1  # Weight for pseudo-label loss


@dataclass
class ModelConfig:
    """GNN model configuration."""
    
    # Architecture
    model_type: str = "graphsage"  # "graphsage", "gat"
    hidden_channels: int = 128
    num_layers: int = 2
    dropout: float = 0.5
    
    # GAT-specific
    num_heads: int = 4


@dataclass
class FederatedConfig:
    """Federated learning configuration."""
    
    # Training
    num_rounds: int = 20
    local_epochs: int = 3
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    
    # Boundary exchange
    use_boundary_exchange: bool = True
    boundary_embedding_dim: int = 128
    
    # PSI simulation
    psi_hash_salt: str = "fedgraph_aml_2024"
    
    # Aggregation
    aggregation: str = "fedavg"  # "fedavg", "fedprox"
    fedprox_mu: float = 0.01  # Only for fedprox

    # Loss weights
    boundary_loss_weight: float = 0.1
    pseudo_label_weight: float = 0.1


@dataclass
class ExperimentConfig:
    """Experiment and evaluation configuration."""
    
    # Reproducibility
    seed: int = 42
    
    # Metrics
    metrics: List[str] = field(default_factory=lambda: [
        "f1", "precision", "recall", "roc_auc", "pr_auc"
    ])
    
    # Class imbalance handling
    # Elliptic: ~2% illicit, ~21% licit, ~77% unknown
    pos_weight: float = 10.0  # Weight for illicit class
    
    # Thresholds to evaluate
    thresholds: List[float] = field(default_factory=lambda: [
        0.3, 0.4, 0.5, 0.6, 0.7
    ])
    
    # Output
    results_dir: Path = Path("./results")
    save_models: bool = True
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    def __post_init__(self):
        """Ensure directories exist."""
        self.data.data_root.mkdir(parents=True, exist_ok=True)
        self.experiment.results_dir.mkdir(parents=True, exist_ok=True)


# Default configuration instance
def get_config(**overrides) -> Config:
    """
    Get configuration with optional overrides.
    
    Example:
        config = get_config(
            data={"num_clients": 5},
            federated={"num_rounds": 30}
        )
    """
    config = Config()
    
    for section, params in overrides.items():
        if hasattr(config, section):
            section_config = getattr(config, section)
            for key, value in params.items():
                if hasattr(section_config, key):
                    setattr(section_config, key, value)
    
    return config


if __name__ == "__main__":
    # Print default configuration
    config = get_config()
    print("=== FedGraph-AML Configuration ===")
    print(f"\nData Config:")
    print(f"  - Clients: {config.data.num_clients}")
    print(f"  - Partition: {config.data.partition_strategy}")
    print(f"  - Semi-supervised: {config.data.use_unlabeled}")
    print(f"\nModel Config:")
    print(f"  - Type: {config.model.model_type}")
    print(f"  - Hidden: {config.model.hidden_channels}")
    print(f"\nFederated Config:")
    print(f"  - Rounds: {config.federated.num_rounds}")
    print(f"  - Boundary Exchange: {config.federated.use_boundary_exchange}")
    print(f"\nDevice: {config.experiment.device}")
