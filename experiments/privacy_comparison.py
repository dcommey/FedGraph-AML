"""
Gradient vs Embedding Privacy Comparison Experiment

Compares:
1. Gradient Inversion Attack on FedAvg (how much can we reconstruct from shared gradients)
2. Embedding Inversion Attack on FedGraph (how much can we reconstruct from shared embeddings)

Purpose: Quantify the privacy advantage of sharing embeddings vs gradients
"""
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import r2_score, mean_squared_error

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from data.elliptic_loader import EllipticDataset
from data.partitioner import create_federated_data
from models.gnn import create_model
from federated.client import create_clients
from federated.server import FederatedServer


SEEDS = [42, 123, 456]
NUM_ROUNDS = 10
LOCAL_EPOCHS = 3


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FeatureReconstructor(nn.Module):
    """MLP to reconstruct original features from gradients or embeddings."""
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


def extract_gradient_features(model, data, client_data, device, num_samples=1000):
    """
    Extract gradient information that would be shared in FedAvg.
    Returns flattened gradients and corresponding node features.
    """
    model.to(device)
    model.train()
    
    # Get a batch of nodes
    train_mask = client_data['train_mask']
    train_indices = train_mask.nonzero(as_tuple=True)[0]
    
    if len(train_indices) > num_samples:
        selected = train_indices[torch.randperm(len(train_indices))[:num_samples]]
    else:
        selected = train_indices
    
    node_features = []
    node_gradients = []
    
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    
    for idx in selected:
        idx = idx.item()
        # Skip unlabeled nodes (label == 2 or -1)
        if y[idx].item() not in [0, 1]:
            continue
            
        model.zero_grad()
        
        # Forward pass
        out = model(x, edge_index)
        # Handle tuple output (logits, embeddings)
        if isinstance(out, tuple):
            out = out[0]
        
        # Compute loss for this node - use nll_loss for compatibility
        target = y[idx].long()
        logits = out[idx]
        # Handle binary vs multi-class
        if logits.dim() == 0:
            # Binary output, convert to 2-class
            logits = torch.stack([-logits, logits])
        log_probs = F.log_softmax(logits.unsqueeze(0), dim=1)
        loss = F.nll_loss(log_probs, target.unsqueeze(0))
        loss.backward()
        
        # Collect gradients from first layer (most informative about input)
        first_layer_grad = None
        for name, param in model.named_parameters():
            if 'convs.0' in name and 'weight' in name and param.grad is not None:
                first_layer_grad = param.grad.clone().flatten()
                break
        
        if first_layer_grad is not None:
            node_features.append(x[idx].cpu().numpy())
            node_gradients.append(first_layer_grad.cpu().numpy())
    
    return np.array(node_features), np.array(node_gradients)


def extract_embedding_features(model, data, client_data, device, num_samples=1000):
    """
    Extract embeddings that would be shared in FedGraph.
    Returns embeddings and corresponding node features.
    """
    model.to(device)
    model.eval()
    
    train_mask = client_data['train_mask']
    train_indices = train_mask.nonzero(as_tuple=True)[0]
    
    if len(train_indices) > num_samples:
        selected = train_indices[torch.randperm(len(train_indices))[:num_samples]]
    else:
        selected = train_indices
    
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    
    with torch.no_grad():
        embeddings = model.get_embeddings(x, edge_index)
    
    node_features = x[selected].cpu().numpy()
    node_embeddings = embeddings[selected].cpu().numpy()
    
    return node_features, node_embeddings


def train_reconstructor(features, representations, input_dim, output_dim, device, epochs=100):
    """Train MLP to reconstruct features from representations (gradients or embeddings)."""
    # Split train/test
    n = len(features)
    perm = np.random.permutation(n)
    train_idx = perm[:int(0.8*n)]
    test_idx = perm[int(0.8*n):]
    
    X_train = torch.FloatTensor(representations[train_idx]).to(device)
    y_train = torch.FloatTensor(features[train_idx]).to(device)
    X_test = torch.FloatTensor(representations[test_idx]).to(device)
    y_test = torch.FloatTensor(features[test_idx]).to(device)
    
    model = FeatureReconstructor(input_dim, output_dim).to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_test)
            val_loss = criterion(val_pred, y_test).item()
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        pred = model(X_test).cpu().numpy()
        true = y_test.cpu().numpy()
    
    r2 = r2_score(true.flatten(), pred.flatten())
    mse = mean_squared_error(true.flatten(), pred.flatten())
    
    # Normalize MSE by variance
    normalized_mse = mse / np.var(true.flatten())
    
    return {
        "r2": r2,
        "mse": mse,
        "normalized_mse": normalized_mse
    }


def main():
    print("=" * 70)
    print("GRADIENT vs EMBEDDING PRIVACY COMPARISON")
    print("=" * 70)
    
    config = get_config()
    device = torch.device(config.experiment.device)
    print(f"Device: {device}")
    
    print("\nLoading Elliptic dataset...")
    dataset = EllipticDataset(root=str(config.data.data_root))
    data = dataset.load()
    
    gradient_results = []
    embedding_results = []
    
    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        set_seed(seed)
        
        # Create model and partition data
        model = create_model(
            config.model.model_type,
            in_channels=data.num_features,
            hidden_channels=128,
            num_layers=2
        )
        
        client_data, silos, stats = create_federated_data(
            data, num_clients=3, strategy="metis"
        )
        
        # Train model briefly
        print("  Training model...")
        clients = create_clients(
            model=model,
            client_data_list=client_data,
            learning_rate=0.01,
            boundary_loss_weight=0.1,
            device=str(device)
        )
        server = FederatedServer(model)
        
        for r in range(NUM_ROUNDS):
            server.run_round(clients=clients, local_epochs=LOCAL_EPOCHS, verbose=False)
        
        # Get trained model
        trained_model = server.global_model
        
        # Use first client's data for privacy analysis
        test_data = client_data[0]
        
        print("  Extracting gradients...")
        grad_features, gradients = extract_gradient_features(
            trained_model, data, test_data, device, num_samples=500
        )
        
        print("  Extracting embeddings...")
        emb_features, embeddings = extract_embedding_features(
            trained_model, data, test_data, device, num_samples=500
        )
        
        if len(gradients) > 50 and len(embeddings) > 50:
            print("  Training gradient reconstructor...")
            grad_attack = train_reconstructor(
                grad_features, gradients,
                input_dim=gradients.shape[1],
                output_dim=grad_features.shape[1],
                device=device
            )
            gradient_results.append(grad_attack)
            print(f"    Gradient R² = {grad_attack['r2']:.4f}")
            
            print("  Training embedding reconstructor...")
            emb_attack = train_reconstructor(
                emb_features, embeddings,
                input_dim=embeddings.shape[1],
                output_dim=emb_features.shape[1],
                device=device
            )
            embedding_results.append(emb_attack)
            print(f"    Embedding R² = {emb_attack['r2']:.4f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("PRIVACY COMPARISON RESULTS")
    print("=" * 70)
    
    if gradient_results:
        grad_r2 = np.array([r['r2'] for r in gradient_results])
        grad_mse = np.array([r['normalized_mse'] for r in gradient_results])
        print(f"\nGradient Inversion (FedAvg Attack Surface):")
        print(f"  R² = {grad_r2.mean():.4f} ± {grad_r2.std():.4f}")
        print(f"  Normalized MSE = {grad_mse.mean():.4f} ± {grad_mse.std():.4f}")
    
    if embedding_results:
        emb_r2 = np.array([r['r2'] for r in embedding_results])
        emb_mse = np.array([r['normalized_mse'] for r in embedding_results])
        print(f"\nEmbedding Inversion (FedGraph Attack Surface):")
        print(f"  R² = {emb_r2.mean():.4f} ± {emb_r2.std():.4f}")
        print(f"  Normalized MSE = {emb_mse.mean():.4f} ± {emb_mse.std():.4f}")
    
    if gradient_results and embedding_results:
        privacy_gain = (grad_r2.mean() - emb_r2.mean()) / grad_r2.mean() * 100
        print(f"\n🔒 Privacy Improvement: {privacy_gain:.1f}% reduction in information leakage")
    
    # Save results
    results = {
        "gradient_inversion": gradient_results,
        "embedding_inversion": embedding_results,
        "summary": {
            "gradient_r2_mean": float(grad_r2.mean()) if gradient_results else None,
            "gradient_r2_std": float(grad_r2.std()) if gradient_results else None,
            "embedding_r2_mean": float(emb_r2.mean()) if embedding_results else None,
            "embedding_r2_std": float(emb_r2.std()) if embedding_results else None,
        }
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path("results") / f"privacy_comparison_{timestamp}.json"
    results_file.parent.mkdir(exist_ok=True)
    
    # Custom encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    main()
