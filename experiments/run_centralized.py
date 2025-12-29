"""
Centralized GNN Baseline (Oracle Upper Bound)

This script trains a GNN on the FULL, unparsed Elliptic graph.
This represents the theoretical maximum performance achievable
if all VASPs shared their data centrally (no privacy constraints).

The purpose is to establish an oracle upper bound to contextualize
federated learning results.

Usage:
    python experiments/run_centralized.py
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import numpy as np
from scipy import stats

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from data.elliptic_loader import EllipticDataset
from models.gnn import create_model


SEEDS = [42, 123, 456, 789, 2024]
EPOCHS = 100  # More epochs since we train the full graph


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def train_centralized_gnn(
    data,
    seed: int,
    config,
    device: torch.device
) -> Dict[str, float]:
    """Train a centralized GNN on the full graph."""
    
    set_seed(seed)
    
    data = data.to(device)
    
    model = create_model(
        config.model.model_type,
        in_channels=data.num_features,
        hidden_channels=config.model.hidden_channels,
        num_layers=config.model.num_layers
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        weight_decay=5e-4
    )
    
    # Compute class weights for imbalanced dataset
    train_mask = data.train_mask & (data.y != -1)
    y_train = data.y[train_mask]
    if y_train.sum() > 0:
        pos_weight = (y_train == 0).sum().float() / (y_train == 1).sum().float()
    else:
        pos_weight = torch.tensor(10.0)
    
    best_val_f1 = 0
    best_test_metrics = {}
    patience = 20
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index)
        
        # Weighted cross-entropy
        train_mask = data.train_mask & (data.y != -1)
        weight = torch.tensor([1.0, pos_weight.item()], device=device)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask], weight=weight)
        
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            
            # Validation metrics
            val_mask = data.val_mask & (data.y != -1)
            if val_mask.sum() > 0:
                y_val = data.y[val_mask].cpu().numpy()
                pred_val = pred[val_mask].cpu().numpy()
                val_f1 = f1_score(y_val, pred_val, zero_division=0)
                
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                    
                    # Compute test metrics
                    test_mask = data.test_mask & (data.y != -1)
                    if test_mask.sum() > 0:
                        y_test = data.y[test_mask].cpu().numpy()
                        pred_test = pred[test_mask].cpu().numpy()
                        prob_test = F.softmax(out[test_mask], dim=1)[:, 1].cpu().numpy()
                        
                        best_test_metrics = {
                            "f1": f1_score(y_test, pred_test, zero_division=0),
                            "precision": precision_score(y_test, pred_test, zero_division=0),
                            "recall": recall_score(y_test, pred_test, zero_division=0),
                            "roc_auc": roc_auc_score(y_test, prob_test) if len(np.unique(y_test)) > 1 else 0.5
                        }
                else:
                    patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            break
    
    return best_test_metrics


def main():
    """Run centralized GNN experiments with multiple seeds."""
    print("=" * 70)
    print("CENTRALIZED GNN BASELINE (Oracle Upper Bound)")
    print("=" * 70)
    print(f"Seeds: {SEEDS}")
    print(f"Epochs: {EPOCHS}")
    
    config = get_config()
    device = torch.device(config.experiment.device)
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading Elliptic dataset (full graph)...")
    dataset = EllipticDataset(root=str(config.data.data_root))
    data = dataset.load()
    print(f"Nodes: {data.num_nodes}, Edges: {data.edge_index.shape[1]}")
    
    # Run with multiple seeds
    all_results = []
    
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        start_time = time.time()
        
        metrics = train_centralized_gnn(data, seed, config, device)
        elapsed = time.time() - start_time
        
        metrics["seed"] = seed
        metrics["time"] = elapsed
        all_results.append(metrics)
        
        print(f"  F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, "
              f"Recall: {metrics['recall']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Compute statistics
    f1_scores = [r["f1"] for r in all_results]
    precisions = [r["precision"] for r in all_results]
    recalls = [r["recall"] for r in all_results]
    aucs = [r["roc_auc"] for r in all_results]
    
    print("\n" + "=" * 70)
    print("CENTRALIZED GNN RESULTS (Oracle Upper Bound)")
    print("=" * 70)
    print(f"F1:        {np.mean(f1_scores):.4f} ± {np.std(f1_scores, ddof=1):.4f}")
    print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions, ddof=1):.4f}")
    print(f"Recall:    {np.mean(recalls):.4f} ± {np.std(recalls, ddof=1):.4f}")
    print(f"ROC-AUC:   {np.mean(aucs):.4f} ± {np.std(aucs, ddof=1):.4f}")
    
    # Save results
    results_dir = Path(config.experiment.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"centralized_gnn_{timestamp}.json"
    
    summary = {
        "method": "Centralized GNN (Oracle)",
        "description": "Trained on full graph with no privacy constraints",
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "statistics": {
            "f1_mean": np.mean(f1_scores),
            "f1_std": np.std(f1_scores, ddof=1),
            "precision_mean": np.mean(precisions),
            "precision_std": np.std(precisions, ddof=1),
            "recall_mean": np.mean(recalls),
            "recall_std": np.std(recalls, ddof=1),
            "roc_auc_mean": np.mean(aucs),
            "roc_auc_std": np.std(aucs, ddof=1)
        },
        "raw_results": all_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    return summary


if __name__ == "__main__":
    main()
