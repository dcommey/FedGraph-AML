
"""
Sparse Experiment: Low-Connectivity Test
Addresses Reviewer Comment #1: Testing if FedGraph-VASP outperforms FedAvg
when natural cross-silo connectivity is low (e.g. temporal partitioning).
"""
import sys
import json
import time
from pathlib import Path
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

# Configuration for fast but meaningful test
SEED = 42
NUM_ROUNDS = 30  
LOCAL_EPOCHS = 3
NUM_CLIENTS = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_experiment(strategy="temporal"):
    print(f"Running Sparse Experiment with Strategy: {strategy}")
    set_seed(SEED)
    
    # Load Data
    dataset = EllipticDataset(root="./data/elliptic")
    data = dataset.load()
    
    # Create Partition
    print("Partitioning data...")
    client_data, silos, stats = create_federated_data(
        data, num_clients=NUM_CLIENTS, strategy="temporal"
    )
    
    # Log partition stats
    for i, silo in enumerate(silos):
        print(f"Silo {i}: {silo.num_nodes} nodes, {silo.num_boundary_nodes} boundary nodes")
    
    config = get_config()
    # OVERRIDE: Boost boundary loss weight for sparse setting
    config.federated.boundary_loss_weight = 5.0
    print(f"Overridden boundary_loss_weight: {config.federated.boundary_loss_weight}")
    
    results = {}
    
    # --- FEDAVG ---
    print("\nRunning FedAvg...")
    fedavg_model = create_model(
        config.model.model_type, data.num_features, 
        config.model.hidden_channels, config.model.num_layers
    ).to(DEVICE)
    
    clients = create_clients(fedavg_model, client_data, config.federated.learning_rate, 
                             config.federated.boundary_loss_weight, config.federated.pseudo_label_weight, 
                             str(DEVICE))
    server = FederatedServer(fedavg_model)
    
    history_fedavg = []
    for r in range(NUM_ROUNDS):
        metrics = server.run_round(clients, LOCAL_EPOCHS, use_boundary_exchange=False, verbose=False)
        f1 = metrics.get('avg_test_f1', 0)
        history_fedavg.append(f1)
        if r % 5 == 0:
            print(f"Round {r}: {f1:.4f}")
            
    results['fedavg'] = history_fedavg
    print(f"FedAvg Fin: {history_fedavg[-1]:.4f}")
    
    # --- FEDGRAPH ---
    print("\nRunning FedGraph...")
    fedgraph_model = create_model(
        config.model.model_type, data.num_features, 
        config.model.hidden_channels, config.model.num_layers
    ).to(DEVICE)
    
    clients = create_clients(fedgraph_model, client_data, config.federated.learning_rate, 
                             config.federated.boundary_loss_weight, config.federated.pseudo_label_weight, 
                             str(DEVICE))
    server = FederatedServer(fedgraph_model)
    boundary_buffer = BoundaryEmbeddingBuffer(embedding_dim=config.model.hidden_channels, use_pqc=False)
    
    history_fedgraph = []
    for r in range(NUM_ROUNDS):
        if r > 0:
             for client in clients:
                result = client.get_boundary_embeddings()
                if result is not None:
                    indices, embeddings = result
                    boundary_buffer.update(client.client_id, indices, embeddings)

        metrics = server.run_round(clients, LOCAL_EPOCHS, use_boundary_exchange=True, boundary_buffer=boundary_buffer.buffer, verbose=False)
        f1 = metrics.get('avg_test_f1', 0)
        history_fedgraph.append(f1)
        if r % 5 == 0:
             print(f"Round {r}: {f1:.4f}")
             
    results['fedgraph'] = history_fedgraph
    print(f"FedGraph Fin: {history_fedgraph[-1]:.4f}")
    
    # Save results
    with open(f"results/sparse_experiment_{strategy}.json", 'w') as f:
        json.dump(results, f, indent=2)
        
if __name__ == "__main__":
    run_experiment("temporal")
