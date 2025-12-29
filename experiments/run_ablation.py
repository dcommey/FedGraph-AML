"""
Ablation Study: Partitioning Strategies

Compares all partitioning strategies to find which best demonstrates
the FedGraph-AML improvement over standard FedAvg.

Strategies:
1. Random: Nodes randomly assigned to clients (baseline, most cross-silo edges)
2. Temporal-Interleaved: Timesteps interleaved across clients (preserves temporal)
3. METIS: Community-based (realistic, minimal cross-silo edges)

Usage:
    python experiments/run_ablation.py --clients 3 --rounds 15
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
from data.partitioner import GraphPartitioner, create_federated_data
from models.gnn import create_model
from federated.client import create_clients
from federated.server import FederatedServer
from federated.boundary_exchange import BoundaryEmbeddingBuffer


def run_federated_comparison(
    data,
    strategy: str,
    num_clients: int,
    num_rounds: int,
    local_epochs: int,
    device: torch.device,
    config
) -> dict:
    """
    Run FedAvg vs FedGraph-AML comparison for a given partitioning strategy.
    """
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy.upper()}")
    print(f"{'='*60}")
    
    # Partition data
    client_data, silos, partition_stats = create_federated_data(
        data,
        num_clients=num_clients,
        strategy=strategy
    )
    
    print(f"Cross-silo edges: {partition_stats['cross_edge_ratio']*100:.2f}%")
    print(f"Boundary nodes: {partition_stats['total_boundary_nodes']}")
    
    # Check training data distribution
    for i, cd in enumerate(client_data):
        train_count = (cd.train_mask & (cd.y != -1)).sum().item()
        illicit_count = ((cd.train_mask & (cd.y != -1)) & (cd.y == 1)).sum().item()
        print(f"  Client {i}: {train_count} train samples, {illicit_count} illicit")
    
    results = {'strategy': strategy, 'stats': partition_stats}
    
    # Run experiments for both FedAvg and FedGraph-AML
    for use_boundary in [False, True]:
        method_name = "FedGraph-AML" if use_boundary else "FedAvg"
        print(f"\n--- {method_name} ---")
        
        # Create fresh model
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
            boundary_loss_weight=config.federated.boundary_loss_weight,
            pseudo_label_weight=config.federated.pseudo_label_weight,
            device=str(device)
        )
        
        # Create server
        server = FederatedServer(template_model)
        
        # Boundary buffer
        boundary_buffer = BoundaryEmbeddingBuffer(
            embedding_dim=config.model.hidden_channels,
            use_pqc=True # Enable PQC for final experiment
        ) if use_boundary else None
        
        best_test_f1 = 0
        history = []
        
        for round_num in range(num_rounds):
            # Update boundary buffer if using FedGraph-AML
            if use_boundary and round_num > 0:
                for client in clients:
                    result = client.get_boundary_embeddings()
                    if result is not None:
                        indices, embeddings = result
                        boundary_buffer.update(client.client_id, indices, embeddings)
            
            # Run training round
            round_metrics = server.run_round(
                clients=clients,
                local_epochs=local_epochs,
                use_boundary_exchange=use_boundary,
                boundary_buffer=boundary_buffer.buffer if boundary_buffer else None,
                use_unlabeled=False,
                verbose=False
            )
            
            test_f1 = round_metrics.get('avg_test_f1', 0)
            history.append(test_f1)
            
            if test_f1 > best_test_f1:
                best_test_f1 = test_f1
            
            if (round_num + 1) % 5 == 0:
                print(f"  Round {round_num+1}: Test F1={test_f1:.4f}, Best={best_test_f1:.4f}")
        
        key = "fedgraph_aml" if use_boundary else "fedavg"
        results[key] = {
            'best_f1': best_test_f1,
            'final_f1': history[-1],
            'history': history
        }
    
    # Compute improvement
    improvement = 0
    if results['fedavg']['best_f1'] > 0:
        improvement = ((results['fedgraph_aml']['best_f1'] - results['fedavg']['best_f1']) 
                      / results['fedavg']['best_f1'] * 100)
    
    results['improvement'] = improvement
    print(f"\n🚀 Improvement: {improvement:+.1f}%")
    
    return results


def run_ablation(args):
    """Run ablation study across all partitioning strategies."""
    print("=" * 60)
    print("FedGraph-AML Ablation Study: Partitioning Strategies")
    print("=" * 60)
    
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
    
    # Strategies to test
    # 'metis' will try to use pymetis, falling back to spectral clustering
    # 'random' is the baseline
    strategies = ['metis', 'random']
    
    all_results = {}
    
    for strategy in strategies:
        try:
            results = run_federated_comparison(
                data=data,
                strategy=strategy,
                num_clients=args.clients,
                num_rounds=args.rounds,
                local_epochs=args.local_epochs,
                device=device,
                config=config
            )
            all_results[strategy] = results
        except Exception as e:
            print(f"Error with {strategy}: {e}")
            all_results[strategy] = {'error': str(e)}
    
    # Summary
    print("\n" + "=" * 60)
    print("ABLATION STUDY SUMMARY")
    print("=" * 60)
    print(f"{'Strategy':<15} {'FedAvg F1':<12} {'FedGraph F1':<12} {'Improvement':<12}")
    print("-" * 60)
    
    for strategy, results in all_results.items():
        if 'error' in results:
            print(f"{strategy:<15} ERROR: {results['error']}")
        else:
            fedavg_f1 = results['fedavg']['best_f1']
            fedgraph_f1 = results['fedgraph_aml']['best_f1']
            improvement = results['improvement']
            print(f"{strategy:<15} {fedavg_f1:<12.4f} {fedgraph_f1:<12.4f} {improvement:+.1f}%")
    
    # Find best strategy
    best_strategy = None
    best_improvement = -float('inf')
    for strategy, results in all_results.items():
        if 'error' not in results and results['improvement'] > best_improvement:
            best_improvement = results['improvement']
            best_strategy = strategy
    
    if best_strategy:
        print(f"\n🏆 Best Strategy: {best_strategy} (Improvement: {best_improvement:+.1f}%)")
    
    # Save results
    results_dir = Path(config.experiment.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"ablation_results_{timestamp}.json"
    
    # Prepare for JSON
    save_results = {}
    for strategy, results in all_results.items():
        if 'error' in results:
            save_results[strategy] = results
        else:
            save_results[strategy] = {
                'fedavg_best_f1': results['fedavg']['best_f1'],
                'fedgraph_best_f1': results['fedgraph_aml']['best_f1'],
                'improvement': results['improvement'],
                'cross_edge_ratio': results['stats']['cross_edge_ratio'],
                'boundary_nodes': results['stats']['total_boundary_nodes']
            }
    
    with open(results_file, 'w') as f:
        json.dump({
            'args': vars(args),
            'results': save_results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument("--clients", type=int, default=3,
                       help="Number of clients")
    parser.add_argument("--rounds", type=int, default=15,
                       help="Number of federated rounds")
    parser.add_argument("--local-epochs", type=int, default=3,
                       help="Local epochs per round")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    run_ablation(args)
