"""
Baseline Experiments

Runs and compares baseline models:
1. Local XGBoost (Jullum et al. 2020 proxy)
2. Local GNN (single silo, cut edges)
3. Centralized GNN (oracle upper bound)

Usage:
    python experiments/run_baseline.py --strategy temporal --clients 3
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from data.elliptic_loader import EllipticDataset
from data.partitioner import create_federated_data, GraphPartitioner
from models.gnn import create_model
from models.xgboost_baseline import train_xgboost_baseline
from experiments.metrics import compute_metrics, find_optimal_threshold, MetricsTracker


def train_centralized_gnn(
    data,
    config,
    device,
    num_epochs: int = 100,
    verbose: bool = True
) -> dict:
    """
    Train centralized GNN (oracle upper bound).
    
    This represents the best possible performance if all data
    were shared centrally - no federation, no privacy.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("Training Centralized GNN (Oracle Upper Bound)")
        print("=" * 60)
    
    model = create_model(
        config.model.model_type,
        in_channels=data.num_features,
        hidden_channels=config.model.hidden_channels,
        num_layers=config.model.num_layers
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.federated.learning_rate,
        weight_decay=5e-4
    )
    
    data = data.to(device)
    
    # Compute class weight
    train_mask = data.train_mask & (data.y != -1)
    y_train = data.y[train_mask]
    pos_weight = ((y_train == 0).sum() / (y_train == 1).sum().clamp(min=1)).item()
    pos_weight = torch.tensor([pos_weight]).to(device)
    
    best_val_f1 = 0
    best_metrics = {}
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        
        logits, _ = model(data.x, data.edge_index)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[train_mask],
            data.y[train_mask].float(),
            pos_weight=pos_weight
        )
        loss.backward()
        optimizer.step()
        
        # Evaluate
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits, _ = model(data.x, data.edge_index)
                probs = torch.sigmoid(logits)
            
            # Validation metrics
            val_mask = data.val_mask & (data.y != -1)
            y_true = data.y[val_mask].cpu().numpy()
            y_prob = probs[val_mask].cpu().numpy()
            y_pred = (y_prob > 0.5).astype(int)
            
            metrics = compute_metrics(y_true, y_pred, y_prob)
            
            if metrics['f1'] > best_val_f1:
                best_val_f1 = metrics['f1']
                
                # Compute test metrics at best validation
                test_mask = data.test_mask & (data.y != -1)
                y_true_test = data.y[test_mask].cpu().numpy()
                y_prob_test = probs[test_mask].cpu().numpy()
                
                opt_thresh, _ = find_optimal_threshold(y_true, y_prob, 'f1')
                y_pred_test = (y_prob_test >= opt_thresh).astype(int)
                
                best_metrics = compute_metrics(y_true_test, y_pred_test, y_prob_test)
                best_metrics['threshold'] = opt_thresh
            
            if verbose:
                print(f"Epoch {epoch+1:3d}: Loss={loss.item():.4f}, "
                      f"Val F1={metrics['f1']:.4f}, Best={best_val_f1:.4f}")
    
    return best_metrics


def train_local_gnn(
    data,
    silo,
    config,
    device,
    num_epochs: int = 100,
    verbose: bool = True
) -> dict:
    """
    Train GNN on single silo only (local model with cut edges).
    
    This shows the limitation of siloed AML systems.
    """
    partitioner = GraphPartitioner(num_clients=1)
    local_data = partitioner.get_silo_subgraph(data, silo)
    local_data = local_data.to(device)
    
    model = create_model(
        config.model.model_type,
        in_channels=local_data.num_features,
        hidden_channels=config.model.hidden_channels,
        num_layers=config.model.num_layers
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.federated.learning_rate,
        weight_decay=5e-4
    )
    
    # Class weight
    train_mask = local_data.train_mask & (local_data.y != -1)
    if train_mask.sum() == 0:
        return {'f1': 0, 'precision': 0, 'recall': 0}
    
    y_train = local_data.y[train_mask]
    pos_count = (y_train == 1).sum().clamp(min=1)
    neg_count = (y_train == 0).sum().clamp(min=1)
    pos_weight = torch.tensor([neg_count / pos_count]).to(device)
    
    best_val_f1 = 0
    best_metrics = {}
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        logits, _ = model(local_data.x, local_data.edge_index)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[train_mask],
            local_data.y[train_mask].float(),
            pos_weight=pos_weight
        )
        loss.backward()
        optimizer.step()
        
        # Evaluate periodically
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                logits, _ = model(local_data.x, local_data.edge_index)
                probs = torch.sigmoid(logits)
            
            val_mask = local_data.val_mask & (local_data.y != -1)
            if val_mask.sum() > 0:
                y_true = local_data.y[val_mask].cpu().numpy()
                y_prob = probs[val_mask].cpu().numpy()
                y_pred = (y_prob > 0.5).astype(int)
                
                metrics = compute_metrics(y_true, y_pred, y_prob)
                
                if metrics['f1'] > best_val_f1:
                    best_val_f1 = metrics['f1']
                    
                    test_mask = local_data.test_mask & (local_data.y != -1)
                    if test_mask.sum() > 0:
                        y_true_test = local_data.y[test_mask].cpu().numpy()
                        y_prob_test = probs[test_mask].cpu().numpy()
                        y_pred_test = (y_prob_test > 0.5).astype(int)
                        best_metrics = compute_metrics(y_true_test, y_pred_test, y_prob_test)
    
    return best_metrics


def run_baselines(args):
    """Run all baseline experiments."""
    print("=" * 60)
    print("FedGraph-AML Baseline Experiments")
    print("=" * 60)
    print(f"Strategy: {args.strategy}")
    print(f"Clients: {args.clients}")
    print(f"Seed: {args.seed}")
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    config = get_config()
    device = torch.device(config.experiment.device)
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading Elliptic dataset...")
    dataset = EllipticDataset(root=str(config.data.data_root))
    data = dataset.load()
    
    # Partition data
    print(f"\nPartitioning into {args.clients} silos ({args.strategy})...")
    client_data, silos, partition_stats = create_federated_data(
        data,
        num_clients=args.clients,
        strategy=args.strategy
    )
    
    print(f"Cross-silo edge ratio: {partition_stats['cross_edge_ratio']:.2%}")
    print(f"Boundary nodes per silo: {partition_stats['boundary_per_silo']}")
    
    results = {}
    
    # 1. XGBoost Baseline (Jullum proxy)
    print("\n" + "-" * 60)
    print("1. XGBoost Baseline (Jullum et al. 2020 proxy)")
    print("-" * 60)
    
    # Train on full data (simulating single centralized bank)
    # This is fairer than using partitioned silo data
    try:
        _, xgb_metrics = train_xgboost_baseline(
            data.to('cpu'),
            verbose=True
        )
        results['xgboost_silo0'] = xgb_metrics
    except Exception as e:
        print(f"  XGBoost training failed: {e}")
        results['xgboost_silo0'] = {'f1': 0, 'precision': 0, 'recall': 0, 'roc_auc': 0, 'pr_auc': 0}
    
    # 2. Local GNN (per silo)
    print("\n" + "-" * 60)
    print("2. Local GNN (per silo, cut edges)")
    print("-" * 60)
    
    local_gnn_results = []
    for i, silo in enumerate(silos):
        print(f"\nSilo {i}:")
        metrics = train_local_gnn(data, silo, config, device, num_epochs=50, verbose=False)
        local_gnn_results.append(metrics)
        print(f"  F1={metrics.get('f1', 0):.4f}, "
              f"Precision={metrics.get('precision', 0):.4f}, "
              f"Recall={metrics.get('recall', 0):.4f}")
    
    # Average across silos
    avg_f1 = np.mean([m.get('f1', 0) for m in local_gnn_results])
    results['local_gnn_avg'] = {'f1': avg_f1}
    print(f"\nAverage Local GNN F1: {avg_f1:.4f}")
    
    # 3. Centralized GNN (Oracle)
    print("\n" + "-" * 60)
    print("3. Centralized GNN (Oracle Upper Bound)")
    print("-" * 60)
    
    centralized_metrics = train_centralized_gnn(data, config, device, num_epochs=100)
    results['centralized_gnn'] = centralized_metrics
    
    # Summary
    print("\n" + "=" * 60)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Method':<30} {'F1':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 60)
    print(f"{'XGBoost (Silo 0)':<30} {results['xgboost_silo0']['f1']:<10.4f} "
          f"{results['xgboost_silo0']['precision']:<10.4f} "
          f"{results['xgboost_silo0']['recall']:<10.4f}")
    print(f"{'Local GNN (Average)':<30} {results['local_gnn_avg']['f1']:<10.4f} - -")
    print(f"{'Centralized GNN (Oracle)':<30} {results['centralized_gnn']['f1']:<10.4f} "
          f"{results['centralized_gnn']['precision']:<10.4f} "
          f"{results['centralized_gnn']['recall']:<10.4f}")
    
    # Save results
    results_dir = Path(config.experiment.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"baseline_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'args': vars(args),
            'partition_stats': {k: v if not isinstance(v, list) else v 
                               for k, v in partition_stats.items()},
            'results': {k: {kk: float(vv) for kk, vv in v.items()} 
                       for k, v in results.items()}
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline experiments")
    parser.add_argument("--strategy", type=str, default="temporal",
                       choices=["temporal", "random", "metis"],
                       help="Partitioning strategy")
    parser.add_argument("--clients", type=int, default=3,
                       help="Number of clients/silos")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    run_baselines(args)
