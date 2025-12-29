"""
Ablation Study: Hyperparameter Sensitivity Analysis

Varies key hyperparameters to understand their impact on performance:
1. boundary_loss_weight (λ): Weight of boundary alignment loss
2. num_clients (K): Number of VASPs in the federation
3. embedding_dim: Dimension of GNN hidden layers

Usage:
    python experiments/ablation_study.py
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import numpy as np

import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from data.elliptic_loader import EllipticDataset
from data.partitioner import create_federated_data
from models.gnn import create_model
from federated.client import create_clients
from federated.server import FederatedServer
from federated.boundary_exchange import BoundaryEmbeddingBuffer


# Fixed seed for reproducibility within ablation
SEED = 42
NUM_ROUNDS = 15
LOCAL_EPOCHS = 3


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(
    data,
    num_clients: int,
    boundary_loss_weight: float,
    embedding_dim: int,
    config,
    device: torch.device
) -> Dict[str, float]:
    """Run a single experiment with given hyperparameters."""
    
    set_seed(SEED)
    
    # Create model with specified embedding dim
    template_model = create_model(
        config.model.model_type,
        in_channels=data.num_features,
        hidden_channels=embedding_dim,
        num_layers=config.model.num_layers
    )
    
    # Partition data
    client_data, silos, partition_stats = create_federated_data(
        data,
        num_clients=num_clients,
        strategy="metis"
    )
    
    # Create clients with specified boundary weight
    clients = create_clients(
        model=template_model,
        client_data_list=client_data,
        learning_rate=config.federated.learning_rate,
        boundary_loss_weight=boundary_loss_weight,
        pseudo_label_weight=config.federated.pseudo_label_weight,
        device=str(device)
    )
    
    server = FederatedServer(template_model)
    
    boundary_buffer = BoundaryEmbeddingBuffer(
        embedding_dim=embedding_dim,
        use_pqc=True
    )
    
    best_f1 = 0
    history = []
    
    for round_num in range(NUM_ROUNDS):
        # Update boundary buffer
        if round_num > 0:
            for client in clients:
                result = client.get_boundary_embeddings()
                if result is not None:
                    indices, embeddings = result
                    boundary_buffer.update(client.client_id, indices, embeddings)
        
        # Run training round
        round_metrics = server.run_round(
            clients=clients,
            local_epochs=LOCAL_EPOCHS,
            use_boundary_exchange=True,
            boundary_buffer=boundary_buffer.buffer,
            use_unlabeled=False,
            verbose=False
        )
        
        test_f1 = round_metrics.get('avg_test_f1', 0)
        history.append(test_f1)
        
        if test_f1 > best_f1:
            best_f1 = test_f1
    
    return {
        "best_f1": best_f1,
        "final_f1": history[-1] if history else 0,
        "cross_edge_ratio": partition_stats["cross_edge_ratio"],
        "boundary_nodes": partition_stats["total_boundary_nodes"]
    }


def ablation_boundary_weight(data, config, device) -> Dict[str, Any]:
    """Ablation on boundary loss weight (λ)."""
    print("\n" + "=" * 60)
    print("ABLATION: Boundary Loss Weight (λ)")
    print("=" * 60)
    
    weights = [0.01, 0.05, 0.1, 0.2, 0.5]
    results = []
    
    for weight in weights:
        print(f"  λ = {weight}...", end=" ", flush=True)
        res = run_experiment(
            data=data,
            num_clients=3,
            boundary_loss_weight=weight,
            embedding_dim=128,
            config=config,
            device=device
        )
        res["lambda"] = weight
        results.append(res)
        print(f"F1 = {res['best_f1']:.4f}")
    
    # Find best
    best = max(results, key=lambda x: x["best_f1"])
    print(f"\n  🏆 Best λ = {best['lambda']} (F1 = {best['best_f1']:.4f})")
    
    return {"ablation": "boundary_loss_weight", "results": results, "best": best}


def ablation_num_clients(data, config, device) -> Dict[str, Any]:
    """Ablation on number of clients (K)."""
    print("\n" + "=" * 60)
    print("ABLATION: Number of Clients (K)")
    print("=" * 60)
    
    client_counts = [2, 3, 5, 8]
    results = []
    
    for k in client_counts:
        print(f"  K = {k}...", end=" ", flush=True)
        res = run_experiment(
            data=data,
            num_clients=k,
            boundary_loss_weight=0.1,
            embedding_dim=128,
            config=config,
            device=device
        )
        res["num_clients"] = k
        results.append(res)
        print(f"F1 = {res['best_f1']:.4f} (cross-edge: {res['cross_edge_ratio']:.2%})")
    
    # Find best
    best = max(results, key=lambda x: x["best_f1"])
    print(f"\n  🏆 Best K = {best['num_clients']} (F1 = {best['best_f1']:.4f})")
    
    return {"ablation": "num_clients", "results": results, "best": best}


def ablation_embedding_dim(data, config, device) -> Dict[str, Any]:
    """Ablation on embedding dimension."""
    print("\n" + "=" * 60)
    print("ABLATION: Embedding Dimension")
    print("=" * 60)
    
    dims = [64, 128, 256]
    results = []
    
    for dim in dims:
        print(f"  dim = {dim}...", end=" ", flush=True)
        res = run_experiment(
            data=data,
            num_clients=3,
            boundary_loss_weight=0.1,
            embedding_dim=dim,
            config=config,
            device=device
        )
        res["embedding_dim"] = dim
        results.append(res)
        print(f"F1 = {res['best_f1']:.4f}")
    
    # Find best
    best = max(results, key=lambda x: x["best_f1"])
    print(f"\n  🏆 Best dim = {best['embedding_dim']} (F1 = {best['best_f1']:.4f})")
    
    return {"ablation": "embedding_dim", "results": results, "best": best}


def main():
    """Run all ablation studies."""
    print("=" * 70)
    print("ABLATION STUDY: Hyperparameter Sensitivity Analysis")
    print("=" * 70)
    print(f"Fixed seed: {SEED}")
    
    config = get_config()
    device = torch.device(config.experiment.device)
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading Elliptic dataset...")
    dataset = EllipticDataset(root=str(config.data.data_root))
    data = dataset.load()
    
    all_ablations = []
    
    # Run ablations
    all_ablations.append(ablation_boundary_weight(data, config, device))
    all_ablations.append(ablation_num_clients(data, config, device))
    all_ablations.append(ablation_embedding_dim(data, config, device))
    
    # Save results
    results_dir = Path(config.experiment.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"ablation_study_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            "config": {
                "seed": SEED,
                "num_rounds": NUM_ROUNDS,
                "local_epochs": LOCAL_EPOCHS
            },
            "ablations": all_ablations
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    # Summary
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)
    for abl in all_ablations:
        param = list(abl["best"].keys())[0] if abl["best"] else "unknown"
        # Get the hyperparameter name (not best_f1 or other metrics)
        for key in abl["best"]:
            if key not in ["best_f1", "final_f1", "cross_edge_ratio", "boundary_nodes"]:
                param = key
                break
        print(f"  {abl['ablation']}: Best {param} = {abl['best'].get(param, 'N/A')}, F1 = {abl['best']['best_f1']:.4f}")
    
    return all_ablations


if __name__ == "__main__":
    main()
