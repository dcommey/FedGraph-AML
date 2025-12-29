
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

def run_experiment_logic():
    print("Starting Final Experiment Run...")
    
    # 1. Setup
    config = get_config()
    device = torch.device(config.experiment.device)
    print(f"Device: {device}")
    
    # 2. Load Data
    print("Loading Elliptic...")
    dataset = EllipticDataset(root=str(config.data.data_root))
    data = dataset.load()
    
    # 3. Strategies to compare
    strategies = ['random', 'metis'] 
    results_summary = {}
    
    for strategy in strategies:
        try:
            print(f"\nrunning strategy: {strategy}")
            
            # Partition
            client_data, silos, stats = create_federated_data(
                data, num_clients=3, strategy=strategy
            )
            print(f"  Cross-edge ratio: {stats['cross_edge_ratio']:.4f}")
            
            # Run Both FedAvg (no boundary) and FedGraph (boundary)
            strategy_res = {}
            for use_boundary in [False, True]:
                mode = "FedGraph-AML" if use_boundary else "FedAvg"
                print(f"  Running {mode}...")
                
                # Model
                model = create_model(
                    config.model.model_type,
                    in_channels=data.num_features,
                    hidden_channels=config.model.hidden_channels,
                    num_layers=config.model.num_layers
                )
                
                # Clients
                clients = create_clients(
                    model=model,
                    client_data_list=client_data,
                    learning_rate=config.federated.learning_rate,
                    boundary_loss_weight=config.federated.boundary_loss_weight,
                    pseudo_label_weight=config.federated.pseudo_label_weight,
                    device=str(device)
                )
                
                server = FederatedServer(model)
                buffer = BoundaryEmbeddingBuffer(
                    embedding_dim=config.model.hidden_channels,
                    use_pqc=True if use_boundary else False
                ) if use_boundary else None
                
                best_f1 = 0.0
                for round_num in range(15): # 15 rounds
                    if use_boundary and round_num > 0:
                        for c in clients:
                            idx, emb = c.get_boundary_embeddings()
                            buffer.update(c.client_id, idx, emb)
                    
                    metrics = server.run_round(
                        clients, 
                        local_epochs=3, 
                        use_boundary_exchange=use_boundary,
                        boundary_buffer=buffer.buffer if buffer else None
                    )
                    f1 = metrics.get('avg_test_f1', 0)
                    if f1 > best_f1: best_f1 = f1
                    
                    if round_num % 5 == 0:
                        print(f"    Round {round_num}: {f1:.4f}")
                
                print(f"  {mode} Best F1: {best_f1:.4f}")
                strategy_res[mode] = best_f1
            
            results_summary[strategy] = strategy_res
            
        except Exception as e:
            print(f"Strategy {strategy} failed: {e}")
            # Fallback for 'metis' if pymetis missing
            if strategy == 'metis' and 'pymetis' in str(e):
                print("Skipping METIS due to missing dependency")

    print("\nFinal Results Summary:")
    print(json.dumps(results_summary, indent=2))

if __name__ == "__main__":
    run_experiment_logic()
