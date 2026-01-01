
"""
Get Convergence History (Corrected)
Runs a short (20 round) experiment to capture per-round F1 scores for plotting.
Saves results to results/convergence_history.json
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from data.elliptic_loader import EllipticDataset
from data.partitioner import create_federated_data
from models.gnn import create_model
from federated.client import create_clients
from federated.server import FederatedServer
from federated.boundary_exchange import BoundaryEmbeddingBuffer

# Configuration
SEED = 42
NUM_ROUNDS = 50
LOCAL_EPOCHS = 3
NUM_CLIENTS = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    print(f"Generating REAL convergence data (Rounds={NUM_ROUNDS}, Seed={SEED})...")
    set_seed(SEED)
    
    # Load Data
    dataset = EllipticDataset(root="./data/elliptic")
    data = dataset.load()
    
    # Create Partition
    print("Partitioning data...")
    client_data, silos, _ = create_federated_data(
        data, num_clients=NUM_CLIENTS, strategy="metis"
    )
    
    config = get_config()
    history_data = {
        "rounds": list(range(1, NUM_ROUNDS + 1)),
        "local": [],
        "fedavg": [],
        "fedgraph": []
    }
    
    # --- 1. LOCAL BASELINE (Average of 3 continuous training processes) ---
    print("\nRunning Local Baseline...")
    local_models = []
    optimizers = []
    
    # Initialize separate models for each client
    for i in range(NUM_CLIENTS):
        m = create_model(
            config.model.model_type, data.num_features, 
            config.model.hidden_channels, config.model.num_layers
        ).to(DEVICE)
        local_models.append(m)
        optimizers.append(torch.optim.Adam(m.parameters(), lr=config.federated.learning_rate))
    
    for r in range(NUM_ROUNDS):
        round_f1s = []
        for i in range(NUM_CLIENTS):
            cd = client_data[i].to(DEVICE)
            model = local_models[i]
            opt = optimizers[i]
            
            # Train LOCAL_EPOCHS
            model.train()
            
            # Calculate pos_weight for this client
            labeled_mask = (cd.y >= 0)
            train_mask = cd.train_mask.bool() & labeled_mask
            if train_mask.sum() > 0:
                train_y = cd.y[train_mask]
                num_pos = (train_y == 1).sum().item()
                num_neg = (train_y == 0).sum().item()
                if num_pos > 0:
                    pos_weight = torch.tensor([num_neg / num_pos]).to(DEVICE)
                else:
                    pos_weight = torch.tensor([1.0]).to(DEVICE)
            else:
                 pos_weight = torch.tensor([1.0]).to(DEVICE)

            for _ in range(LOCAL_EPOCHS):
                opt.zero_grad()
                out, _ = model(cd.x, cd.edge_index)
                
                if train_mask.sum() > 0:
                    loss = F.binary_cross_entropy_with_logits(
                        out[train_mask], 
                        cd.y[train_mask].float(),
                        pos_weight=pos_weight
                    )
                    loss.backward()
                    opt.step()
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                out, _ = model(cd.x, cd.edge_index)
                pred = (out > 0).long()
                labeled_mask = (cd.y >= 0).bool()
                test_mask = cd.test_mask.bool() & labeled_mask
                
                if test_mask.sum() > 0:
                    from sklearn.metrics import f1_score
                    y_true = cd.y[test_mask].cpu().numpy()
                    y_pred = pred[test_mask].cpu().numpy()
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                    round_f1s.append(f1)
                else:
                    round_f1s.append(0.0)
                    
        avg_f1 = np.mean(round_f1s) if round_f1s else 0.0
        history_data["local"].append(avg_f1)
        print(f"Local Round {r+1}: {avg_f1:.4f}")

    # --- 2. FEDAVG ---
    print("\nRunning FedAvg...")
    fedavg_model = create_model(
        config.model.model_type, data.num_features, 
        config.model.hidden_channels, config.model.num_layers
    ).to(DEVICE)
    
    clients = create_clients(fedavg_model, client_data, config.federated.learning_rate, 
                             config.federated.boundary_loss_weight, config.federated.pseudo_label_weight, 
                             str(DEVICE))
    server = FederatedServer(fedavg_model)
    
    for r in range(NUM_ROUNDS):
        # Use server.run_round to handle aggregation logic
        metrics = server.run_round(
            clients=clients,
            local_epochs=LOCAL_EPOCHS,
            use_boundary_exchange=False,
            verbose=False
        )
        avg_f1 = metrics.get('avg_test_f1', 0.0)
        history_data["fedavg"].append(avg_f1)
        print(f"FedAvg Round {r+1}: {avg_f1:.4f}")
        
    # --- 3. FEDGRAPH ---
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
    
    for r in range(NUM_ROUNDS):
        # Update boundary buffer
        if r > 0:
             for client in clients:
                result = client.get_boundary_embeddings()
                if result is not None:
                    indices, embeddings = result
                    boundary_buffer.update(client.client_id, indices, embeddings)

        metrics = server.run_round(
            clients=clients,
            local_epochs=LOCAL_EPOCHS,
            use_boundary_exchange=True,
            boundary_buffer=boundary_buffer.buffer,
            verbose=False
        )
        avg_f1 = metrics.get('avg_test_f1', 0.0)
        history_data["fedgraph"].append(avg_f1)
        print(f"FedGraph Round {r+1}: {avg_f1:.4f}")
        
    # Save Results
    output_path = "results/convergence_history.json"
    with open(output_path, 'w') as f:
        json.dump(history_data, f, indent=2)
    print(f"\nSaved convergence history to {output_path}")

if __name__ == "__main__":
    main()
