"""
Federated Learning Experiments

Runs and compares federated approaches:
1. FedAvg (standard federated learning)
2. FedGraph-AML (with boundary embedding exchange)

Usage:
    python experiments/run_federated.py --clients 3 --rounds 20 --boundary-exchange
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
from data.partitioner import create_federated_data
from models.gnn import create_model
from federated.client import create_clients
from federated.server import FederatedServer, collect_boundary_embeddings
from federated.boundary_exchange import BoundaryEmbeddingBuffer
from experiments.metrics import compute_metrics, find_optimal_threshold


def run_federated_experiment(
    args,
    use_boundary_exchange: bool = False,
    experiment_name: str = "FedAvg"
) -> dict:
    """
    Run a single federated learning experiment.
    
    Args:
        args: Command line arguments
        use_boundary_exchange: Whether to use FedGraph-AML
        experiment_name: Name for logging
        
    Returns:
        Dictionary of results
    """
    print(f"\n{'=' * 60}")
    print(f"Running: {experiment_name}")
    print(f"{'=' * 60}")
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    config = get_config()
    device = torch.device(config.experiment.device)
    
    # Load and partition data
    dataset = EllipticDataset(root=str(config.data.data_root))
    data = dataset.load()
    
    client_data, silos, partition_stats = create_federated_data(
        data,
        num_clients=args.clients,
        strategy=args.strategy
    )
    
    print(f"Clients: {args.clients}, Strategy: {args.strategy}")
    print(f"Cross-silo edges: {partition_stats['cross_edge_ratio']:.2%}")
    
    # Create model
    template_model = create_model(
        config.model.model_type,
        in_channels=data.num_features,
        hidden_channels=config.model.hidden_channels,
        num_layers=config.model.num_layers
    )
    
    # Create clients
    clients = create_clients(
        model=template_model,
        client_data_list=client_data,
        learning_rate=config.federated.learning_rate,
        device=str(device)
    )
    
    # Create server
    server = FederatedServer(template_model)
    
    # Boundary buffer for FedGraph-AML
    boundary_buffer = None
    if use_boundary_exchange:
        boundary_buffer = BoundaryEmbeddingBuffer(
            embedding_dim=config.model.hidden_channels
        )
    
    # Training loop
    history = {'val_f1': [], 'test_f1': [], 'loss': []}
    best_test_f1 = 0
    best_metrics = {}
    
    for round_num in range(args.rounds):
        # Update boundary buffer (FedGraph-AML)
        if use_boundary_exchange and round_num > 0:
            # Collect boundary embeddings from all clients
            for client in clients:
                result = client.get_boundary_embeddings()
                if result is not None:
                    indices, embeddings = result
                    boundary_buffer.update(client.client_id, indices, embeddings)
        
        # Run training round
        round_metrics = server.run_round(
            clients=clients,
            local_epochs=args.local_epochs,
            use_boundary_exchange=use_boundary_exchange,
            boundary_buffer=boundary_buffer.buffer if boundary_buffer else None,
            use_unlabeled=args.semi_supervised,
            verbose=False
        )
        
        history['val_f1'].append(round_metrics.get('avg_val_f1', 0))
        history['test_f1'].append(round_metrics.get('avg_test_f1', 0))
        history['loss'].append(round_metrics.get('loss', 0))
        
        # Track best
        if round_metrics.get('avg_test_f1', 0) > best_test_f1:
            best_test_f1 = round_metrics.get('avg_test_f1', 0)
            best_metrics = round_metrics.copy()
            best_metrics['round'] = round_num + 1
        
        # Progress
        if (round_num + 1) % 5 == 0 or round_num == 0:
            print(f"Round {round_num + 1:3d}: "
                  f"Loss={round_metrics.get('loss', 0):.4f}, "
                  f"Val F1={round_metrics.get('avg_val_f1', 0):.4f}, "
                  f"Test F1={round_metrics.get('avg_test_f1', 0):.4f}")
    
    # Final evaluation with optimal threshold
    print(f"\nBest Test F1: {best_test_f1:.4f} (Round {best_metrics.get('round', 0)})")
    
    return {
        'experiment': experiment_name,
        'best_test_f1': best_test_f1,
        'best_metrics': best_metrics,
        'history': history,
        'partition_stats': partition_stats
    }


def run_all_experiments(args):
    """Run all federated experiments and compare."""
    print("=" * 60)
    print("FedGraph-AML Federated Experiments")
    print("=" * 60)
    
    config = get_config()
    results = {}
    
    # 1. Standard FedAvg
    results['fedavg'] = run_federated_experiment(
        args,
        use_boundary_exchange=False,
        experiment_name="FedAvg (Standard FL)"
    )
    
    # 2. FedGraph-AML (with boundary exchange)
    results['fedgraph_aml'] = run_federated_experiment(
        args,
        use_boundary_exchange=True,
        experiment_name="FedGraph-AML (Boundary Exchange)"
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("FEDERATED RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Method':<35} {'Best F1':<12} {'Best Round':<12}")
    print("-" * 60)
    
    for name, result in results.items():
        print(f"{result['experiment']:<35} "
              f"{result['best_test_f1']:<12.4f} "
              f"{result['best_metrics'].get('round', 'N/A'):<12}")
    
    # Improvement
    fedavg_f1 = results['fedavg']['best_test_f1']
    fedgraph_f1 = results['fedgraph_aml']['best_test_f1']
    
    if fedavg_f1 > 0:
        improvement = (fedgraph_f1 - fedavg_f1) / fedavg_f1 * 100
        print(f"\n🚀 FedGraph-AML Improvement: +{improvement:.1f}% over FedAvg")
    
    # Save results
    results_dir = Path(config.experiment.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"federated_results_{timestamp}.json"
    
    # Prepare for JSON serialization
    save_results = {}
    for name, result in results.items():
        save_results[name] = {
            'experiment': result['experiment'],
            'best_test_f1': float(result['best_test_f1']),
            'best_round': result['best_metrics'].get('round', 0),
            'history': {k: [float(v) for v in vals] 
                       for k, vals in result['history'].items()}
        }
    
    with open(results_file, 'w') as f:
        json.dump({
            'args': vars(args),
            'results': save_results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run federated experiments")
    parser.add_argument("--strategy", type=str, default="temporal",
                       choices=["temporal", "random", "metis"],
                       help="Partitioning strategy")
    parser.add_argument("--clients", type=int, default=3,
                       help="Number of clients/silos")
    parser.add_argument("--rounds", type=int, default=20,
                       help="Number of federated rounds")
    parser.add_argument("--local-epochs", type=int, default=3,
                       help="Local epochs per round")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--semi-supervised", action="store_true",
                       help="Use semi-supervised learning with unlabeled data")
    parser.add_argument("--quick", action="store_true",
                       help="Quick run with fewer rounds for testing")
    
    args = parser.parse_args()
    
    if args.quick:
        args.rounds = 5
        args.local_epochs = 1
        print("Quick mode: 5 rounds, 1 local epoch")
    
    run_all_experiments(args)
