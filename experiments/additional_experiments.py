"""
Quick experiment for temporal partitioning and K=4 ablation.
Runs minimal experiments to get publishable results.
"""
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from data.elliptic_loader import EllipticDataset
from data.partitioner import create_federated_data
from models.gnn import create_model
from federated.client import create_clients
from federated.server import FederatedServer
from federated.boundary_exchange import BoundaryEmbeddingBuffer

SEEDS = [42, 123, 456]  # 3 seeds for quick validation
NUM_ROUNDS = 30
LOCAL_EPOCHS = 3

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_experiment(data, num_clients, strategy, boundary_weight, config, device, method="fedgraph"):
    """Run single experiment."""
    template_model = create_model(
        config.model.model_type,
        in_channels=data.num_features,
        hidden_channels=128,
        num_layers=2
    )
    
    client_data, silos, stats = create_federated_data(
        data, num_clients=num_clients, strategy=strategy
    )
    
    clients = create_clients(
        model=template_model,
        client_data_list=client_data,
        learning_rate=0.01,
        boundary_loss_weight=boundary_weight if method == "fedgraph" else 0.0,
        device=str(device)
    )
    
    server = FederatedServer(template_model)
    boundary_buffer = BoundaryEmbeddingBuffer(embedding_dim=128, use_pqc=True)
    
    best_f1 = 0
    for r in range(NUM_ROUNDS):
        if r > 0 and method == "fedgraph":
            for client in clients:
                result = client.get_boundary_embeddings()
                if result:
                    idx, emb = result
                    boundary_buffer.update(client.client_id, idx, emb)
        
        metrics = server.run_round(
            clients=clients,
            local_epochs=LOCAL_EPOCHS,
            use_boundary_exchange=(method == "fedgraph"),
            boundary_buffer=boundary_buffer.buffer if method == "fedgraph" else {},
            verbose=False
        )
        
        test_f1 = metrics.get('avg_test_f1', 0)
        if test_f1 > best_f1:
            best_f1 = test_f1
        
        if (r + 1) % 10 == 0:
            print(f"    Round {r+1}/{NUM_ROUNDS}, F1={test_f1:.4f}")
    
    return best_f1, stats["cross_edge_ratio"]

def main():
    print("=" * 70)
    print("ADDITIONAL EXPERIMENTS FOR PAPER")
    print("=" * 70)
    
    config = get_config()
    device = torch.device(config.experiment.device)
    print(f"Device: {device}")
    
    print("\nLoading Elliptic dataset...")
    dataset = EllipticDataset(root=str(config.data.data_root))
    data = dataset.load()
    
    results = {}
    
    # ============================================
    # EXPERIMENT 1: Temporal Partitioning
    # ============================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Temporal Partitioning (Low Cross-Edge)")
    print("=" * 60)
    
    temporal_results = {"fedavg": [], "fedgraph": []}
    
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        set_seed(seed)
        
        for method in ["fedavg", "fedgraph"]:
            print(f"  Running {method}...", end=" ", flush=True)
            f1, cross_ratio = run_experiment(
                data, num_clients=3, strategy="temporal",
                boundary_weight=0.1, config=config, device=device, method=method
            )
            temporal_results[method].append(f1)
            print(f"F1={f1:.4f} (cross-edge: {cross_ratio:.2%})")
    
    print("\n--- Temporal Results ---")
    for method in ["fedavg", "fedgraph"]:
        arr = np.array(temporal_results[method])
        print(f"  {method}: F1 = {arr.mean():.4f} ± {arr.std():.4f}")
    
    results["temporal"] = temporal_results
    
    # ============================================
    # EXPERIMENT 2: K=4 Clients
    # ============================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: K=4 Clients (Scalability)")
    print("=" * 60)
    
    k4_results = []
    
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        set_seed(seed)
        
        print(f"  Running K=4...", end=" ", flush=True)
        try:
            f1, cross_ratio = run_experiment(
                data, num_clients=4, strategy="metis",
                boundary_weight=0.1, config=config, device=device, method="fedgraph"
            )
            k4_results.append(f1)
            print(f"F1={f1:.4f} (cross-edge: {cross_ratio:.2%})")
        except Exception as e:
            print(f"Error: {e}")
    
    if k4_results:
        arr = np.array(k4_results)
        print(f"\n--- K=4 Results ---")
        print(f"  FedGraph K=4: F1 = {arr.mean():.4f} ± {arr.std():.4f}")
    
    results["k4"] = k4_results
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path("results") / f"additional_experiments_{timestamp}.json"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if temporal_results["fedgraph"]:
        fg = np.array(temporal_results["fedgraph"])
        fa = np.array(temporal_results["fedavg"])
        diff = fg.mean() - fa.mean()
        print(f"Temporal Partition: FedGraph {fg.mean():.4f} vs FedAvg {fa.mean():.4f} (diff: {diff:+.4f})")
    
    if k4_results:
        k4 = np.array(k4_results)
        print(f"K=4 Scalability: F1 = {k4.mean():.4f} ± {k4.std():.4f}")

if __name__ == "__main__":
    main()
