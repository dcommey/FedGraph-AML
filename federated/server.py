"""
Federated Learning Server

Coordinates federated learning across multiple clients.
Implements FedAvg aggregation and orchestrates training rounds.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import copy
from collections import OrderedDict


class FederatedServer:
    """
    Federated learning server that coordinates training across clients.
    
    Responsibilities:
    1. Initialize global model
    2. Distribute model to clients
    3. Aggregate client updates (FedAvg)
    4. Track global metrics
    """
    
    def __init__(
        self,
        model: nn.Module,
        aggregation: str = "fedavg"
    ):
        """
        Initialize the federated server.
        
        Args:
            model: Global model template
            aggregation: Aggregation strategy ("fedavg", "fedprox")
        """
        self.global_model = copy.deepcopy(model)
        self.aggregation = aggregation
        self.round_num = 0
        self.history = {
            'train_loss': [],
            'val_f1': [],
            'test_f1': []
        }
        
    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        """Get current global model weights."""
        return {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}
    
    def aggregate_weights(
        self,
        client_weights: List[Dict[str, torch.Tensor]],
        client_num_samples: Optional[List[int]] = None
    ):
        """
        Aggregate client model weights using FedAvg.
        
        Args:
            client_weights: List of state dicts from clients
            client_num_samples: Optional list of sample counts for weighted averaging
        """
        if len(client_weights) == 0:
            return
        
        # Compute weights for averaging
        if client_num_samples is not None:
            total_samples = sum(client_num_samples)
            weights = [n / total_samples for n in client_num_samples]
        else:
            weights = [1.0 / len(client_weights)] * len(client_weights)
        
        # Aggregate
        aggregated = OrderedDict()
        
        for key in client_weights[0].keys():
            # Weighted sum
            aggregated[key] = sum(
                w * client_weights[i][key].float()
                for i, w in enumerate(weights)
            )
        
        # Update global model
        self.global_model.load_state_dict(aggregated)
        self.round_num += 1
    
    def run_round(
        self,
        clients: List,  # List[FederatedClient]
        local_epochs: int = 1,
        use_boundary_exchange: bool = False,
        boundary_buffer: Optional[Dict] = None,
        use_unlabeled: bool = False,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Run one federated learning round.
        
        Args:
            clients: List of FederatedClient objects
            local_epochs: Number of local training epochs per client
            use_boundary_exchange: Whether to use FedGraph-AML boundary exchange
            boundary_buffer: Shared boundary embedding buffer
            use_unlabeled: Whether to use semi-supervised learning
            verbose: Print progress
            
        Returns:
            Dictionary with round metrics
        """
        # Distribute global model to all clients
        global_weights = self.get_global_weights()
        for client in clients:
            client.set_model_weights(global_weights)
        
        # Local training
        client_metrics = []
        client_weights = []
        client_samples = []
        
        for client in clients:
            # Train locally
            for epoch in range(local_epochs):
                metrics = client.train_epoch(
                    boundary_buffer=boundary_buffer if use_boundary_exchange else None,
                    use_unlabeled=use_unlabeled
                )
            
            # Collect weights
            client_weights.append(client.get_model_weights())
            client_samples.append(metrics.get('train_samples', 1))
            client_metrics.append(metrics)
        
        # Aggregate
        self.aggregate_weights(client_weights, client_samples)
        
        # Evaluate
        eval_metrics = self.evaluate_clients(clients)
        
        # Track history
        avg_loss = sum(m['loss'] for m in client_metrics) / len(client_metrics)
        self.history['train_loss'].append(avg_loss)
        self.history['val_f1'].append(eval_metrics.get('avg_val_f1', 0))
        self.history['test_f1'].append(eval_metrics.get('avg_test_f1', 0))
        
        if verbose:
            print(f"Round {self.round_num}: Loss={avg_loss:.4f}, "
                  f"Val F1={eval_metrics.get('avg_val_f1', 0):.4f}, "
                  f"Test F1={eval_metrics.get('avg_test_f1', 0):.4f}")
        
        return {
            'loss': avg_loss,
            **eval_metrics
        }
    
    def evaluate_clients(
        self,
        clients: List  # List[FederatedClient]
    ) -> Dict[str, float]:
        """
        Evaluate all clients and aggregate metrics.
        
        Args:
            clients: List of FederatedClient objects
            
        Returns:
            Dictionary with aggregated metrics
        """
        # Update clients with latest global model
        global_weights = self.get_global_weights()
        
        all_metrics = []
        for client in clients:
            client.set_model_weights(global_weights)
            metrics = client.evaluate()
            all_metrics.append(metrics)
        
        # Aggregate metrics
        result = {}
        
        for key in ['val_f1', 'val_precision', 'val_recall', 
                    'test_f1', 'test_precision', 'test_recall']:
            values = [m.get(key, 0) for m in all_metrics]
            if values:
                result[f'avg_{key}'] = sum(values) / len(values)
                result[f'min_{key}'] = min(values)
                result[f'max_{key}'] = max(values)
        
        return result
    
    def get_best_round(self, metric: str = 'val_f1') -> int:
        """Get the round with best validation metric."""
        if metric not in self.history or not self.history[metric]:
            return 0
        return max(range(len(self.history[metric])), 
                   key=lambda i: self.history[metric][i])


def run_federated_training(
    model: nn.Module,
    clients: List,  # List[FederatedClient]
    num_rounds: int = 20,
    local_epochs: int = 3,
    use_boundary_exchange: bool = False,
    use_unlabeled: bool = False,
    verbose: bool = True
) -> Tuple[nn.Module, Dict]:
    """
    Convenience function to run full federated training.
    
    Args:
        model: Initial model
        clients: List of FederatedClient objects
        num_rounds: Number of federated rounds
        local_epochs: Local epochs per round
        use_boundary_exchange: Enable FedGraph-AML
        use_unlabeled: Enable semi-supervised learning
        verbose: Print progress
        
    Returns:
        Tuple of (trained global model, training history)
    """
    server = FederatedServer(model)
    
    # Boundary buffer for FedGraph-AML
    boundary_buffer = {} if use_boundary_exchange else None
    
    for round_num in range(num_rounds):
        # Update boundary buffer from previous round
        if use_boundary_exchange and round_num > 0:
            boundary_buffer = collect_boundary_embeddings(clients)
        
        # Run round
        server.run_round(
            clients=clients,
            local_epochs=local_epochs,
            use_boundary_exchange=use_boundary_exchange,
            boundary_buffer=boundary_buffer,
            use_unlabeled=use_unlabeled,
            verbose=verbose
        )
    
    return server.global_model, server.history


def collect_boundary_embeddings(
    clients: List  # List[FederatedClient]
) -> Dict[int, torch.Tensor]:
    """
    Collect boundary embeddings from all clients.
    
    This is the core of FedGraph-AML: clients share embeddings
    for boundary nodes so other clients can use them for
    cross-silo message passing.
    
    Args:
        clients: List of FederatedClient objects
        
    Returns:
        Dictionary mapping global node indices to embeddings
    """
    buffer = {}
    
    for client in clients:
        result = client.get_boundary_embeddings()
        if result is not None:
            indices, embeddings = result
            for idx, emb in zip(indices.tolist(), embeddings):
                buffer[idx] = emb.cpu()
    
    return buffer
