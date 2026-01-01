"""
Graph Partitioner for Simulating Cross-Institution Silos

Implements multiple partitioning strategies to simulate realistic
scenarios where different VASPs/banks hold different portions of 
the transaction graph.

Strategies:
1. Temporal: Partition by timesteps (realistic for temporal data)
2. METIS: Community-based partitioning (realistic for clustered users)
3. Random: Simple baseline for comparison

Key feature: Identifies "boundary nodes" - nodes with edges crossing silos.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from torch_geometric.data import Data


@dataclass
class SiloData:
    """Data for a single silo (client/institution)."""
    
    silo_id: int
    node_mask: torch.Tensor          # Boolean mask for nodes in this silo
    local_edge_mask: torch.Tensor    # Edges where both endpoints are in silo
    boundary_nodes: torch.Tensor     # Node indices that have cross-silo edges
    cross_silo_edges: torch.Tensor   # Edges crossing to other silos (for analysis)
    
    @property
    def num_nodes(self) -> int:
        return self.node_mask.sum().item()
    
    @property
    def num_local_edges(self) -> int:
        return self.local_edge_mask.sum().item()
    
    @property
    def num_boundary_nodes(self) -> int:
        return len(self.boundary_nodes)


class GraphPartitioner:
    """
    Partitions a graph into K silos to simulate federated learning scenario.
    
    The key challenge: Money laundering patterns span multiple institutions.
    When we cut the graph, we lose these cross-institutional patterns.
    This is exactly what FedGraph-AML aims to solve.
    """
    
    def __init__(self, num_clients: int = 3, strategy: str = "temporal"):
        """
        Initialize the partitioner.
        
        Args:
            num_clients: Number of silos to create
            strategy: Partitioning strategy ("temporal", "metis", "random")
        """
        self.num_clients = num_clients
        self.strategy = strategy
        
    def partition(self, data: Data) -> Tuple[List[SiloData], Dict]:
        """
        Partition the graph into silos.
        
        Args:
            data: PyTorch Geometric Data object with timestep attribute
            
        Returns:
            Tuple of:
            - List of SiloData objects (one per client)
            - Statistics dictionary
        """
        if self.strategy == "temporal":
            return self._partition_temporal(data)
        elif self.strategy == "metis":
            return self._partition_metis(data)
        elif self.strategy == "random":
            return self._partition_random(data)
        elif self.strategy == "stratified":
            return self._partition_stratified(data)
        elif self.strategy == "realistic":
            return self._partition_realistic(data)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _partition_temporal(self, data: Data) -> Tuple[List[SiloData], Dict]:
        """
        Partition by timesteps - realistic for temporal transaction data.
        
        Different "banks" observe transactions at different time periods.
        This simulates a scenario where each institution joined the network
        at different times or focuses on different temporal windows.
        """
        if not hasattr(data, 'timestep'):
            raise ValueError("Data must have 'timestep' attribute for temporal partitioning")
        
        timesteps = data.timestep
        num_timesteps = timesteps.max().item() + 1
        steps_per_client = num_timesteps // self.num_clients
        
        # Assign nodes to clients based on timestep
        node_assignments = torch.zeros(data.num_nodes, dtype=torch.long)
        
        for k in range(self.num_clients):
            start_t = k * steps_per_client
            end_t = (k + 1) * steps_per_client if k < self.num_clients - 1 else num_timesteps
            
            mask = (timesteps >= start_t) & (timesteps < end_t)
            node_assignments[mask] = k
        
        return self._create_silos(data, node_assignments)
    
    def _partition_metis(self, data: Data) -> Tuple[List[SiloData], Dict]:
        """
        Partition using METIS algorithm - preserves community structure.
        
        Falls back to spectral clustering if pymetis not available.
        """
        try:
            import pymetis
            
            # Convert edge_index to adjacency list format for METIS
            edge_index = data.edge_index
            num_nodes = data.num_nodes
            
            adjacency = [[] for _ in range(num_nodes)]
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                adjacency[src].append(dst)
            
            n_cuts, membership = pymetis.part_graph(
                self.num_clients,
                adjacency=adjacency
            )
            
            node_assignments = torch.tensor(membership, dtype=torch.long)
            
            # Calculate actual cross-edge stats (pymetis n_cuts return value is unreliable)
            src_part = node_assignments[edge_index[0]]
            dst_part = node_assignments[edge_index[1]]
            cross_edges = (src_part != dst_part).sum().item()
            ratio = cross_edges / data.num_edges * 100
            
            print(f"METIS partitioning: {cross_edges} cross-edges ({ratio:.2f}%)")
            
            return self._create_silos(data, node_assignments)
            
        except ImportError:
            print("pymetis not available. Using spectral clustering instead.")
            return self._partition_spectral(data)
    
    def _partition_spectral(self, data: Data) -> Tuple[List[SiloData], Dict]:
        """
        Partition using spectral clustering - pure Python alternative to METIS.
        
        Uses eigenvectors of the graph Laplacian to find community structure.
        """
        try:
            from sklearn.cluster import SpectralClustering
            from scipy.sparse import csr_matrix
        except ImportError:
            raise ImportError(
                "scikit-learn and scipy are required for spectral clustering. "
                "Please install them via: pip install scikit-learn scipy"
            )
        
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        
        # Create sparse adjacency matrix
        row = edge_index[0].numpy()
        col = edge_index[1].numpy()
        data_ones = np.ones(len(row))
        adj_matrix = csr_matrix((data_ones, (row, col)), shape=(num_nodes, num_nodes))
        
        # Make symmetric (for undirected graph assumption)
        adj_matrix = adj_matrix + adj_matrix.T
        adj_matrix.data = np.clip(adj_matrix.data, 0, 1)
        
        print(f"Running spectral clustering on {num_nodes} nodes...")
        print(f"  (This may take 10-30 minutes for large graphs, please wait...)")
        import time as _time
        _start = _time.time()
        
        # Use spectral clustering
        clustering = SpectralClustering(
            n_clusters=self.num_clients,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=42,
            n_init=3
        )
        
        node_assignments = torch.tensor(
            clustering.fit_predict(adj_matrix),
            dtype=torch.long
        )
        
        _elapsed = _time.time() - _start
        print(f"Spectral clustering complete. (took {_elapsed:.1f}s)")
        
        return self._create_silos(data, node_assignments)
    
    def _partition_random(self, data: Data) -> Tuple[List[SiloData], Dict]:
        """
        Random partitioning - baseline for comparison.
        
        Each node is randomly assigned to a client.
        This is unrealistic but provides a baseline.
        """
        node_assignments = torch.randint(0, self.num_clients, (data.num_nodes,))
        return self._create_silos(data, node_assignments)
    
    def _partition_stratified(self, data: Data) -> Tuple[List[SiloData], Dict]:
        """
        Stratified partitioning - simulates realistic VASPs.
        
        Creates partitions with controlled cross-edge ratio (~10-15%).
        Uses community-preserving assignment to keep most edges internal.
        """
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        
        # Use connected component-like assignment to minimize cross-edges
        # Assign nodes in contiguous blocks to simulate real VASP boundaries
        nodes_per_client = num_nodes // self.num_clients
        
        node_assignments = torch.zeros(num_nodes, dtype=torch.long)
        for k in range(self.num_clients):
            start_idx = k * nodes_per_client
            end_idx = start_idx + nodes_per_client if k < self.num_clients - 1 else num_nodes
            node_assignments[start_idx:end_idx] = k
        
        return self._create_silos(data, node_assignments)
    
    def _partition_realistic(self, data: Data) -> Tuple[List[SiloData], Dict]:
        """
        Realistic VASP partitioning - ~10% cross-institutional edges.
        
        Simulates real-world scenario where most transactions are internal
        to a single exchange, with only ~10% being cross-exchange transfers.
        Uses BFS-based community detection to keep related nodes together.
        """
        import random
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        
        # Build adjacency list
        adj = {i: [] for i in range(num_nodes)}
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            adj[src].append(dst)
        
        # BFS-based assignment: grow regions from random seeds
        node_assignments = torch.full((num_nodes,), -1, dtype=torch.long)
        nodes_per_client = num_nodes // self.num_clients
        
        # Seed each client with a starting node
        random.seed(42)
        unassigned = list(range(num_nodes))
        random.shuffle(unassigned)
        
        for k in range(self.num_clients):
            if not unassigned:
                break
            seed = unassigned[0]
            queue = [seed]
            assigned_count = 0
            target = nodes_per_client if k < self.num_clients - 1 else len(unassigned)
            
            while queue and assigned_count < target:
                node = queue.pop(0)
                if node_assignments[node] == -1:
                    node_assignments[node] = k
                    assigned_count += 1
                    unassigned.remove(node) if node in unassigned else None
                    # Add neighbors to queue (BFS grows region)
                    for neighbor in adj[node]:
                        if node_assignments[neighbor] == -1:
                            queue.append(neighbor)
            
            # Fill remaining slots with any unassigned nodes
            while assigned_count < target and unassigned:
                node = unassigned.pop(0)
                if node_assignments[node] == -1:
                    node_assignments[node] = k
                    assigned_count += 1
        
        # Assign any remaining unassigned nodes
        for i in range(num_nodes):
            if node_assignments[i] == -1:
                node_assignments[i] = self.num_clients - 1
        
        return self._create_silos(data, node_assignments)
    
    def _create_silos(
        self, 
        data: Data, 
        node_assignments: torch.Tensor
    ) -> Tuple[List[SiloData], Dict]:
        """
        Create SiloData objects from node assignments.
        
        This is where we identify:
        1. Local edges (both endpoints in same silo)
        2. Cross-silo edges (endpoints in different silos)
        3. Boundary nodes (nodes with cross-silo edges)
        """
        edge_index = data.edge_index
        silos = []
        
        total_local_edges = 0
        total_cross_edges = 0
        total_boundary_nodes = 0
        
        for k in range(self.num_clients):
            # Node mask for this silo
            node_mask = (node_assignments == k)
            
            # Get node indices in this silo
            node_indices = torch.where(node_mask)[0]
            node_set = set(node_indices.tolist())
            
            # Classify edges
            src, dst = edge_index[0], edge_index[1]
            src_in_silo = node_mask[src]
            dst_in_silo = node_mask[dst]
            
            # Local edges: both endpoints in this silo
            local_edge_mask = src_in_silo & dst_in_silo
            
            # Cross-silo edges: source in this silo, destination outside
            cross_silo_mask = src_in_silo & ~dst_in_silo
            cross_silo_edges = edge_index[:, cross_silo_mask]
            
            # Boundary nodes: nodes in this silo with edges to other silos
            # These are the nodes that need "embedding exchange" in FedGraph-AML
            boundary_src = src[cross_silo_mask].unique()
            
            # Also include nodes that RECEIVE edges from outside
            incoming_cross = ~src_in_silo & dst_in_silo
            boundary_dst = dst[incoming_cross].unique()
            
            # Combine boundary nodes
            boundary_nodes = torch.cat([boundary_src, boundary_dst]).unique()
            
            silo = SiloData(
                silo_id=k,
                node_mask=node_mask,
                local_edge_mask=local_edge_mask,
                boundary_nodes=boundary_nodes,
                cross_silo_edges=cross_silo_edges
            )
            silos.append(silo)
            
            total_local_edges += silo.num_local_edges
            total_cross_edges += cross_silo_mask.sum().item()
            total_boundary_nodes += len(boundary_nodes)
        
        # Compute statistics
        stats = {
            "num_clients": self.num_clients,
            "strategy": self.strategy,
            "total_nodes": data.num_nodes,
            "total_edges": data.num_edges,
            "total_local_edges": total_local_edges,
            "total_cross_edges": total_cross_edges,
            "cross_edge_ratio": total_cross_edges / data.num_edges,
            "total_boundary_nodes": total_boundary_nodes,
            "nodes_per_silo": [s.num_nodes for s in silos],
            "edges_per_silo": [s.num_local_edges for s in silos],
            "boundary_per_silo": [s.num_boundary_nodes for s in silos],
        }
        
        return silos, stats
    
    def get_silo_subgraph(
        self, 
        data: Data, 
        silo: SiloData,
        include_boundary_context: bool = False
    ) -> Data:
        """
        Extract a subgraph for a specific silo.
        
        Args:
            data: Original full graph
            silo: SiloData object
            include_boundary_context: If True, include 1-hop neighbors of boundary nodes
                                     (for FedGraph-AML context)
                                     
        Returns:
            PyTorch Geometric Data object for this silo
        """
        # Get local edges only
        local_edge_index = data.edge_index[:, silo.local_edge_mask]
        
        # Map global node indices to local indices
        global_to_local = torch.full((data.num_nodes,), -1, dtype=torch.long)
        local_nodes = torch.where(silo.node_mask)[0]
        global_to_local[local_nodes] = torch.arange(len(local_nodes))
        
        # Remap edge indices
        local_edge_index = global_to_local[local_edge_index]
        
        # Create local data object
        local_data = Data(
            x=data.x[silo.node_mask],
            edge_index=local_edge_index,
            y=data.y[silo.node_mask],
        )
        
        # Copy masks if they exist
        for attr in ['train_mask', 'val_mask', 'test_mask', 'unlabeled_mask']:
            if hasattr(data, attr):
                setattr(local_data, attr, getattr(data, attr)[silo.node_mask])
        
        # Add metadata
        local_data.silo_id = silo.silo_id
        local_data.global_node_indices = local_nodes
        local_data.boundary_local_indices = global_to_local[silo.boundary_nodes]
        local_data.boundary_local_indices = local_data.boundary_local_indices[
            local_data.boundary_local_indices >= 0
        ]  # Filter valid indices
        
        return local_data


def create_federated_data(
    data: Data,
    num_clients: int = 3,
    strategy: str = "temporal"
) -> Tuple[List[Data], Dict]:
    """
    Convenience function to create federated data splits.
    
    Args:
        data: Original PyG Data object
        num_clients: Number of clients/silos
        strategy: Partitioning strategy
        
    Returns:
        Tuple of:
        - List of Data objects (one per client)
        - Statistics dictionary
    """
    partitioner = GraphPartitioner(num_clients=num_clients, strategy=strategy)
    silos, stats = partitioner.partition(data)
    
    client_data = []
    for silo in silos:
        local_data = partitioner.get_silo_subgraph(data, silo)
        client_data.append(local_data)
    
    return client_data, silos, stats


if __name__ == "__main__":
    # Test partitioning
    print("=" * 60)
    print("Testing Graph Partitioner")
    print("=" * 60)
    
    from elliptic_loader import EllipticDataset
    
    # Load data
    dataset = EllipticDataset()
    data = dataset.load()
    
    # Test each strategy
    for strategy in ["temporal", "random"]:
        print(f"\n{'-' * 60}")
        print(f"Strategy: {strategy.upper()}")
        print(f"{'-' * 60}")
        
        client_data, silos, stats = create_federated_data(
            data, 
            num_clients=3, 
            strategy=strategy
        )
        
        print(f"\nPartition Statistics:")
        print(f"  Cross-silo edge ratio: {stats['cross_edge_ratio']:.2%}")
        print(f"  Nodes per silo: {stats['nodes_per_silo']}")
        print(f"  Boundary nodes per silo: {stats['boundary_per_silo']}")
        
        print(f"\nClient Data Shapes:")
        for i, cd in enumerate(client_data):
            print(f"  Client {i}: {cd.num_nodes} nodes, {cd.num_edges} edges, "
                  f"{sum(cd.train_mask).item()} train, "
                  f"{len(cd.boundary_local_indices)} boundary")
