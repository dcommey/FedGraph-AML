"""
Federated Learning Client

Handles local training on a single silo/institution.
Supports boundary embedding extraction for FedGraph-AML.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Dict, Optional, Tuple, List
import copy


class FederatedClient:
    """
    A federated learning client representing a single VASP/institution.
    
    Each client:
    1. Holds a local subgraph of transactions
    2. Trains a local GNN model
    3. Extracts embeddings for boundary nodes (for cross-silo exchange)
    4. Sends model weights to the server for aggregation
    """
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        local_data: Data,
        learning_rate: float = 0.01,
        weight_decay: float = 5e-4,
        boundary_loss_weight: float = 0.1,
        pseudo_label_weight: float = 0.1,
        device: str = "cuda"
    ):
        """
        Initialize a federated client.
        
        Args:
            client_id: Unique identifier for this client
            model: GNN model (will be copied for local training)
            local_data: PyG Data object for this client's subgraph
            learning_rate: Learning rate for local optimizer
            weight_decay: L2 regularization
            boundary_loss_weight: Weight for boundary alignment loss
            pseudo_label_weight: Weight for unlabeled data loss
            device: Device for training
        """
        self.client_id = client_id
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.boundary_loss_weight = boundary_loss_weight
        self.pseudo_label_weight = pseudo_label_weight
        
        # Copy model for local training
        self.model = copy.deepcopy(model).to(self.device)
        
        # Store local data
        self.local_data = local_data.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Boundary node information
        self.boundary_indices = getattr(local_data, 'boundary_local_indices', None)
        self.boundary_embeddings = None
        
        # Class weights for imbalanced data
        self._compute_class_weights()
        
    def _compute_class_weights(self):
        """Compute class weights from local data."""
        y = self.local_data.y
        train_mask = self.local_data.train_mask
        y_train = y[train_mask]
        
        # Only consider labeled data
        labeled = y_train[y_train != -1]
        if len(labeled) == 0:
            self.pos_weight = torch.tensor([1.0]).to(self.device)
            return
            
        num_pos = (labeled == 1).sum().float()
        num_neg = (labeled == 0).sum().float()
        
        if num_pos > 0:
            self.pos_weight = torch.tensor([num_neg / num_pos]).to(self.device)
        else:
            self.pos_weight = torch.tensor([1.0]).to(self.device)
    
    def train_epoch(
        self,
        boundary_buffer: Optional[Dict[int, torch.Tensor]] = None,
        use_unlabeled: bool = False
    ) -> Dict[str, float]:
        """
        Train for one local epoch.
        
        Args:
            boundary_buffer: Optional dict mapping global node IDs to embeddings
                            from other clients (for FedGraph-AML)
            use_unlabeled: Whether to use pseudo-labels for unlabeled nodes
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        data = self.local_data
        
        # Forward pass
        logits, embeddings = self.model(
            data.x, 
            data.edge_index,
            return_embeddings=True
        )
        
        # Store boundary embeddings for exchange
        if self.boundary_indices is not None and len(self.boundary_indices) > 0:
            self.boundary_embeddings = embeddings[self.boundary_indices].detach().clone()
        
        # Compute loss on labeled training data
        train_mask = data.train_mask & (data.y != -1)
        
        if train_mask.sum() == 0:
            return {"loss": 0.0, "train_samples": 0}
        
        loss = F.binary_cross_entropy_with_logits(
            logits[train_mask],
            data.y[train_mask].float(),
            pos_weight=self.pos_weight
        )
        
        # === FEDGRAPH-AML: Boundary Embedding Alignment ===
        # This is the CORE NOVELTY: align local boundary embeddings with foreign ones
        boundary_loss = 0.0
        if boundary_buffer is not None and self.boundary_indices is not None and len(self.boundary_indices) > 0:
            # Get global indices of boundary nodes
            global_indices = data.global_node_indices[self.boundary_indices]
            
            # OPTIMIZED: Vectorized lookup instead of per-node loop
            # Only process a sample of boundary nodes if too many (for speed)
            max_boundary_samples = 1000
            if len(global_indices) > max_boundary_samples:
                # Random sample for efficiency
                sample_idx = torch.randperm(len(global_indices))[:max_boundary_samples]
                global_indices_sample = global_indices[sample_idx]
                boundary_indices_sample = self.boundary_indices[sample_idx]
            else:
                global_indices_sample = global_indices
                boundary_indices_sample = self.boundary_indices
            
            # Find matching embeddings from buffer (vectorized)
            matched_local_idx = []
            matched_foreign_list = []
            
            global_list = global_indices_sample.tolist()
            for i, global_idx in enumerate(global_list):
                if global_idx in boundary_buffer:
                    foreign_emb = boundary_buffer[global_idx]
                    if isinstance(foreign_emb, torch.Tensor):
                        matched_local_idx.append(boundary_indices_sample[i])
                        matched_foreign_list.append(foreign_emb)
            
            # Compute alignment loss (cosine similarity)
            if len(matched_local_idx) > 0:
                local_emb = embeddings[torch.tensor(matched_local_idx, device=self.device)]
                foreign_emb = torch.stack(matched_foreign_list).to(self.device)
                
                # Normalize embeddings
                local_norm = F.normalize(local_emb, p=2, dim=1)
                foreign_norm = F.normalize(foreign_emb, p=2, dim=1)
                
                # Cosine similarity loss (want them to be similar)
                cosine_sim = (local_norm * foreign_norm).sum(dim=1)
                boundary_loss = (1 - cosine_sim).mean()
                
                # Add boundary loss with weight
                loss = loss + self.boundary_loss_weight * boundary_loss
        
        # Optional: Semi-supervised loss on unlabeled data
        if use_unlabeled and hasattr(data, 'train_unlabeled_mask'):
            unlabeled_mask = data.train_unlabeled_mask
            if unlabeled_mask.sum() > 0:
                # Use high-confidence pseudo-labels
                with torch.no_grad():
                    probs = torch.sigmoid(logits[unlabeled_mask])
                    high_conf = (probs > 0.9) | (probs < 0.1)
                    pseudo_labels = (probs > 0.5).float()
                
                if high_conf.sum() > 0:
                    pseudo_loss = F.binary_cross_entropy_with_logits(
                        logits[unlabeled_mask][high_conf],
                        pseudo_labels[high_conf]
                    )
                    loss = loss + self.pseudo_label_weight * pseudo_loss
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "train_samples": train_mask.sum().item(),
            "boundary_loss": float(boundary_loss) if isinstance(boundary_loss, float) else boundary_loss.item()
        }
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate on local validation/test data.
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        data = self.local_data
        
        with torch.no_grad():
            logits, _ = self.model(data.x, data.edge_index)
            probs = torch.sigmoid(logits)
        
        metrics = {}
        
        for split in ['val', 'test']:
            mask_attr = f'{split}_mask'
            if hasattr(data, mask_attr):
                mask = getattr(data, mask_attr) & (data.y != -1)
                if mask.sum() > 0:
                    y_true = data.y[mask].cpu()
                    y_pred = (probs[mask] > 0.5).cpu().float()
                    y_prob = probs[mask].cpu()
                    
                    # Binary metrics
                    tp = ((y_pred == 1) & (y_true == 1)).sum().float()
                    fp = ((y_pred == 1) & (y_true == 0)).sum().float()
                    fn = ((y_pred == 0) & (y_true == 1)).sum().float()
                    
                    precision = tp / (tp + fp + 1e-8)
                    recall = tp / (tp + fn + 1e-8)
                    f1 = 2 * precision * recall / (precision + recall + 1e-8)
                    
                    metrics[f'{split}_f1'] = f1.item()
                    metrics[f'{split}_precision'] = precision.item()
                    metrics[f'{split}_recall'] = recall.item()
        
        return metrics
    
    def get_model_weights(self) -> Dict[str, torch.Tensor]:
        """Get model weights for aggregation."""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
    
    def set_model_weights(self, weights: Dict[str, torch.Tensor]):
        """Set model weights from aggregated global model."""
        self.model.load_state_dict({
            k: v.to(self.device) for k, v in weights.items()
        })
    
    def get_boundary_embeddings(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get boundary node embeddings for cross-silo exchange.
        
        Returns:
            Tuple of (global_node_indices, embeddings) or None
        """
        if self.boundary_embeddings is None:
            return None
            
        # Get global indices of boundary nodes
        global_indices = self.local_data.global_node_indices[self.boundary_indices]
        
        return global_indices, self.boundary_embeddings


def create_clients(
    model: nn.Module,
    client_data_list: List[Data],
    learning_rate: float = 0.01,
    boundary_loss_weight: float = 0.1,
    pseudo_label_weight: float = 0.1,
    device: str = "cuda"
) -> List[FederatedClient]:
    """
    Create federated clients from a list of local data objects.
    
    Args:
        model: Template model to copy for each client
        client_data_list: List of PyG Data objects, one per client
        learning_rate: Learning rate for all clients
        device: Device for training
        
    Returns:
        List of FederatedClient objects
    """
    clients = []
    
    for i, local_data in enumerate(client_data_list):
        client = FederatedClient(
            client_id=i,
            model=model,
            local_data=local_data,
            learning_rate=learning_rate,
            boundary_loss_weight=boundary_loss_weight,
            pseudo_label_weight=pseudo_label_weight,
            device=device
        )
        clients.append(client)
    
    return clients
