"""
Privacy Analysis: Embedding Inversion Resistance

This script validates the privacy claims of FedGraph-VASP by attempting
to reconstruct original node features from exchanged embeddings.

If embeddings can be easily inverted to recover features, then the
boundary exchange leaks private information. We show that:
1. Reconstruction error is high (embeddings are not invertible)
2. The GNN's non-linear transformations provide "computational privacy"

Methodology:
- Train a GNN on the graph
- Extract embeddings for all nodes
- Train an MLP "attacker" to reconstruct original features from embeddings
- Measure reconstruction error (MSE, R²)
- Compare to random baseline

Usage:
    python experiments/privacy_analysis.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_squared_error

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from data.elliptic_loader import EllipticDataset
from models.gnn import create_model


class InversionAttacker(nn.Module):
    """
    MLP that attempts to reconstruct original features from embeddings.
    
    This simulates an honest-but-curious adversary who receives embeddings
    and tries to infer the original transaction features.
    """
    
    def __init__(self, embedding_dim: int, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, feature_dim)
        )
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(embeddings)


def train_gnn_and_extract_embeddings(
    data,
    config,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Train a GNN and extract embeddings for all nodes."""
    
    data = data.to(device)
    
    model = create_model(
        config.model.model_type,
        in_channels=data.num_features,
        hidden_channels=config.model.hidden_channels,
        num_layers=config.model.num_layers
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # Train the GNN
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        
        train_mask = data.train_mask & (data.y != -1)
        if train_mask.sum() > 0:
            loss = F.cross_entropy(out[train_mask], data.y[train_mask])
            loss.backward()
            optimizer.step()
    
    # Extract embeddings (use get_embedding method if available)
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'get_embedding'):
            embeddings = model.get_embedding(data.x, data.edge_index)
        else:
            # Get output of last hidden layer
            embeddings = model(data.x, data.edge_index)
    
    return data.x.cpu(), embeddings.cpu()


def train_inversion_attacker(
    features: torch.Tensor,
    embeddings: torch.Tensor,
    train_ratio: float = 0.8,
    epochs: int = 200,
    device: torch.device = torch.device('cpu')
) -> Dict[str, float]:
    """
    Train an MLP to reconstruct features from embeddings.
    
    Returns metrics measuring inversion success/failure.
    """
    
    n_samples = features.shape[0]
    n_train = int(n_samples * train_ratio)
    
    # Random split
    indices = torch.randperm(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    X_train = embeddings[train_idx].to(device)
    X_test = embeddings[test_idx].to(device)
    y_train = features[train_idx].to(device)
    y_test = features[test_idx].to(device)
    
    # Normalize targets for training stability
    y_mean = y_train.mean(dim=0, keepdim=True)
    y_std = y_train.std(dim=0, keepdim=True) + 1e-8
    y_train_norm = (y_train - y_mean) / y_std
    y_test_norm = (y_test - y_mean) / y_std
    
    # Create attacker model
    attacker = InversionAttacker(
        embedding_dim=embeddings.shape[1],
        feature_dim=features.shape[1]
    ).to(device)
    
    optimizer = torch.optim.Adam(attacker.parameters(), lr=0.001)
    
    best_test_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        attacker.train()
        optimizer.zero_grad()
        pred = attacker(X_train)
        loss = F.mse_loss(pred, y_train_norm)
        loss.backward()
        optimizer.step()
        
        # Validation
        attacker.eval()
        with torch.no_grad():
            test_pred = attacker(X_test)
            test_loss = F.mse_loss(test_pred, y_test_norm)
            if test_loss < best_test_loss:
                best_test_loss = test_loss.item()
    
    # Final evaluation
    attacker.eval()
    with torch.no_grad():
        pred_test = attacker(X_test)
        
        # Denormalize
        pred_denorm = pred_test * y_std.to(device) + y_mean.to(device)
        
        # Compute metrics
        mse = mean_squared_error(y_test.cpu().numpy(), pred_denorm.cpu().numpy())
        
        # R² score (per feature, then average)
        r2_scores = []
        for i in range(features.shape[1]):
            r2 = r2_score(y_test[:, i].cpu().numpy(), pred_denorm[:, i].cpu().numpy())
            r2_scores.append(r2)
        avg_r2 = np.mean(r2_scores)
        
        # Baseline: predict mean
        baseline_pred = y_mean.expand_as(y_test)
        baseline_mse = mean_squared_error(y_test.cpu().numpy(), baseline_pred.cpu().numpy())
        
        # Normalized MSE (relative to data variance)
        data_var = y_test.var().item()
        normalized_mse = mse / data_var if data_var > 0 else mse
    
    return {
        "mse": mse,
        "normalized_mse": normalized_mse,
        "avg_r2": avg_r2,
        "baseline_mse": baseline_mse,
        "improvement_over_baseline": (baseline_mse - mse) / baseline_mse if baseline_mse > 0 else 0,
        "n_test_samples": len(test_idx),
        "n_features": features.shape[1]
    }


def main():
    """Run complete privacy analysis."""
    print("=" * 70)
    print("PRIVACY ANALYSIS: Embedding Inversion Resistance")
    print("=" * 70)
    
    config = get_config()
    device = torch.device(config.experiment.device)
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading Elliptic dataset...")
    dataset = EllipticDataset(root=str(config.data.data_root))
    data = dataset.load()
    print(f"Nodes: {data.num_nodes}, Features: {data.num_features}")
    
    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Train GNN and extract embeddings
    print("\nTraining GNN and extracting embeddings...")
    features, embeddings = train_gnn_and_extract_embeddings(data, config, device)
    print(f"Features shape: {features.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Train inversion attacker
    print("\nTraining inversion attacker (MLP)...")
    results = train_inversion_attacker(features, embeddings, device=device)
    
    # Results
    print("\n" + "=" * 70)
    print("INVERSION ATTACK RESULTS")
    print("=" * 70)
    print(f"Test MSE:           {results['mse']:.4f}")
    print(f"Normalized MSE:     {results['normalized_mse']:.4f}")
    print(f"Average R²:         {results['avg_r2']:.4f}")
    print(f"Baseline MSE:       {results['baseline_mse']:.4f}")
    print(f"Improvement:        {results['improvement_over_baseline']*100:.1f}%")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("PRIVACY INTERPRETATION")
    print("=" * 70)
    
    if results['avg_r2'] < 0.3:
        privacy_verdict = "STRONG"
        print("✓ STRONG PRIVACY: Embeddings are NOT easily invertible.")
        print("  The attacker cannot reconstruct original features with high fidelity.")
        print("  GNN's non-linear transformations provide computational privacy.")
    elif results['avg_r2'] < 0.6:
        privacy_verdict = "MODERATE"
        print("⚠ MODERATE PRIVACY: Some feature information is leaked.")
        print("  Consider adding differential privacy noise to embeddings.")
    else:
        privacy_verdict = "WEAK"
        print("❌ WEAK PRIVACY: Embeddings can be inverted to recover features.")
        print("  This is a privacy concern. Additional defenses needed.")
    
    results["privacy_verdict"] = privacy_verdict
    
    # Save results
    results_dir = Path(config.experiment.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"privacy_analysis_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    main()
