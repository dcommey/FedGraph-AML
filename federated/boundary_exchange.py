
"""
Boundary Embedding Exchange for FedGraph-AML

THE CORE NOVELTY: This module implements the privacy-preserving
boundary embedding exchange mechanism that differentiates FedGraph-AML
from standard Federated Learning approaches like FedSage+.

Key differences:
- FedSage+: GENERATES (hallucinates) missing neighbors locally
- FedGraph-AML: EXCHANGES embeddings for known boundary nodes

Steps:
1. PSI: Identify shared/boundary nodes across silos
2. Exchange: Share embeddings for these nodes (not raw features!)
3. Aggregate: Use foreign embeddings in local message passing
"""

import torch
import torch.nn as nn
from typing import Dict, List, Set, Tuple, Optional, Any
import hashlib
from torch_geometric.data import Data
from federated.security import PostQuantumTunnel

class PrivateSetIntersection:
    """
    Simulates Private Set Intersection (PSI) for identifying
    boundary nodes across silos without revealing other nodes.
    """
    
    def __init__(self, salt: str = "fedgraph_aml_2024"):
        self.salt = salt
    
    def hash_identifier(self, node_id: int) -> str:
        data = f"{self.salt}:{node_id}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def find_intersection(self, set_a: Set[int], set_b: Set[int]) -> Set[int]:
        # In simulation, we just compute the intersection
        return set_a & set_b
    
    def identify_boundary_nodes(self, client_data_list: List[Data], global_node_count: int) -> Dict[int, List[int]]:
        # Collect which nodes each client has
        client_nodes = {}
        for i, data in enumerate(client_data_list):
            if hasattr(data, 'global_node_indices'):
                client_nodes[i] = set(data.global_node_indices.tolist())
            else:
                client_nodes[i] = set(range(data.num_nodes))
        
        # Find pairwise intersections
        boundary_nodes = {i: set() for i in range(len(client_data_list))}
        
        num_clients = len(client_data_list)
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                shared = self.find_intersection(client_nodes[i], client_nodes[j])
                boundary_nodes[i].update(shared)
                boundary_nodes[j].update(shared)
        
        return {k: list(v) for k, v in boundary_nodes.items()}


class BoundaryEmbeddingBuffer:
    """
    Manages the exchange of boundary node embeddings between silos.
    Now secured with Post-Quantum Cryptography (Kyber-512 + AES-GCM).
    """
    
    def __init__(self, embedding_dim: int = 128, use_pqc: bool = False):
        self.embedding_dim = embedding_dim
        self.buffer: Dict[int, Any] = {} 
        self.version = 0
        self.use_pqc = use_pqc
        self.tunnel = PostQuantumTunnel() if use_pqc else None
    
    def update(self, client_id: int, global_indices: torch.Tensor, embeddings: torch.Tensor):
        for idx, emb in zip(global_indices.tolist(), embeddings):
            emb_cpu = emb.cpu()
            
            if self.use_pqc:
                # Encrypt
                stored_data = self.tunnel.encrypt_embedding(emb_cpu)
            else:
                stored_data = emb_cpu
                
            self.buffer[idx] = stored_data
        
        self.version += 1
    
    def get_embeddings(self, global_indices: torch.Tensor, device: torch.device = torch.device('cpu')) -> Tuple[torch.Tensor, torch.Tensor]:
        found_indices = []
        embeddings = []
        
        for idx in global_indices.tolist():
            if idx in self.buffer:
                stored_data = self.buffer[idx]
                
                if self.use_pqc:
                    # Decrypt
                    decrypted = self.tunnel.decrypt_embedding(stored_data)
                    if decrypted is not None:
                        found_indices.append(idx)
                        embeddings.append(decrypted)
                else:
                    found_indices.append(idx)
                    embeddings.append(stored_data)
        
        if not embeddings:
            return torch.tensor([]), torch.tensor([])
        
        return (
            torch.tensor(found_indices, device=device),
            torch.stack(embeddings).to(device)
        )
    
    def clear(self):
        self.buffer.clear()
        self.version += 1


class CrossSiloAggregator(nn.Module):
    """
    Aggregates local embeddings with cross-silo boundary embeddings.
    """
    
    def __init__(self, embedding_dim: int = 128, aggregation: str = "attention"):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.aggregation = aggregation
        
        if aggregation == "attention":
            self.attention = nn.Linear(embedding_dim * 2, 1)
        elif aggregation == "concat":
            self.projection = nn.Linear(embedding_dim * 2, embedding_dim)
    
    def forward(self, local_embeddings: torch.Tensor, foreign_embeddings: Optional[torch.Tensor] = None, local_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        if foreign_embeddings is None or local_indices is None:
            return local_embeddings
        
        updated = local_embeddings.clone()
        
        if self.aggregation == "mean":
            updated[local_indices] = (local_embeddings[local_indices] + foreign_embeddings) / 2
            
        elif self.aggregation == "attention":
            local_emb = local_embeddings[local_indices]
            combined = torch.cat([local_emb, foreign_embeddings], dim=-1)
            weights = torch.sigmoid(self.attention(combined))
            updated[local_indices] = weights * local_emb + (1 - weights) * foreign_embeddings
            
        elif self.aggregation == "concat":
            local_emb = local_embeddings[local_indices]
            combined = torch.cat([local_emb, foreign_embeddings], dim=-1)
            updated[local_indices] = self.projection(combined)
        
        return updated
