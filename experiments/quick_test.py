"""
Quick Test: Verify all components work before full experiment.
Uses random partitioning (fast) with minimal iterations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F

print("Step 1: Imports...")
from config import get_config
from data.elliptic_loader import EllipticDataset
from data.partitioner import create_federated_data
from models.gnn import create_model
from federated.client import create_clients
from federated.server import FederatedServer
from federated.boundary_exchange import BoundaryEmbeddingBuffer
print("  OK")

print("\nStep 2: Load data...")
config = get_config()
device = torch.device(config.experiment.device)
print(f"  Device: {device}")

dataset = EllipticDataset(root=str(config.data.data_root))
data = dataset.load()
print(f"  Nodes: {data.num_nodes}, Edges: {data.edge_index.shape[1]}")

print("\nStep 3: Partition (random)...")
client_data, silos, stats = create_federated_data(
    data, num_clients=3, strategy="random"
)
print(f"  Cross-edge ratio: {stats['cross_edge_ratio']:.2%}")
print(f"  Boundary nodes: {stats['total_boundary_nodes']}")

print("\nStep 4: Test model forward pass...")
model = create_model(
    config.model.model_type,
    in_channels=data.num_features,
    hidden_channels=config.model.hidden_channels,
    num_layers=config.model.num_layers
).to(device)

cd = client_data[0].to(device)
out, emb = model(cd.x, cd.edge_index, return_embeddings=True)
print(f"  Output shape: {out.shape}, dtype: {out.dtype}")
print(f"  Embedding shape: {emb.shape if emb is not None else 'None'}")

print("\nStep 5: Test loss computation...")
labeled_mask = (cd.y >= 0).bool()
train_mask = cd.train_mask.bool() & labeled_mask
train_idx = torch.where(train_mask)[0]
print(f"  Train samples: {len(train_idx)}")

if len(train_idx) > 0:
    loss = F.binary_cross_entropy_with_logits(
        out[train_idx], 
        cd.y[train_idx].float()
    )
    print(f"  Loss: {loss.item():.4f}")

print("\nStep 6: Test clients and server...")
clients = create_clients(
    model=model,
    client_data_list=client_data,
    learning_rate=0.01,
    boundary_loss_weight=0.1,
    pseudo_label_weight=0.1,
    device=str(device)
)
server = FederatedServer(model)
print(f"  Created {len(clients)} clients")

print("\nStep 7: Test one training round (FedAvg)...")
round_metrics = server.run_round(
    clients=clients,
    local_epochs=1,
    use_boundary_exchange=False,
    boundary_buffer=None,
    use_unlabeled=False,
    verbose=True
)
print(f"  Test F1: {round_metrics.get('avg_test_f1', 0):.4f}")

print("\nStep 8: Test boundary buffer (no PQC)...")
buffer = BoundaryEmbeddingBuffer(embedding_dim=128, use_pqc=False)
for client in clients:
    result = client.get_boundary_embeddings()
    if result is not None:
        indices, embeddings = result
        buffer.update(client.client_id, indices, embeddings)
        print(f"  Client {client.client_id}: {len(indices)} boundary embeddings")

print("\nStep 9: Test one training round (FedGraph with boundary)...")
round_metrics = server.run_round(
    clients=clients,
    local_epochs=1,
    use_boundary_exchange=True,
    boundary_buffer=buffer.buffer,
    use_unlabeled=False,
    verbose=True
)
print(f"  Test F1: {round_metrics.get('avg_test_f1', 0):.4f}")

print("\n" + "=" * 50)
print("ALL TESTS PASSED!")
print("=" * 50)
