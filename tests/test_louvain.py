import torch
from torch_geometric.data import Data
from data.partitioner import GraphPartitioner
import networkx as nx

def test_louvain():
    print("Generating synthetic stochastic block model graph...")
    # Generate SBM to simulate community structure
    # 3 communities, 100 nodes each.
    # Intra-prob: 0.1, Inter-prob: 0.01
    sizes = [100, 100, 100, 100] # 400 nodes
    probs = [[0.1, 0.005, 0.005, 0.005],
             [0.005, 0.1, 0.005, 0.005],
             [0.005, 0.005, 0.1, 0.005],
             [0.005, 0.005, 0.005, 0.1]]
    
    g = nx.stochastic_block_model(sizes, probs, seed=42)
    x = torch.randn(400, 16)
    
    # Convert to PyG
    edge_index = torch.tensor(list(g.edges)).t().contiguous()
    # Make undirected
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    data = Data(x=x, edge_index=edge_index, num_nodes=400)
    data.num_edges = edge_index.shape[1]
    
    print(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges")
    
    # Partition
    partitioner = GraphPartitioner(num_clients=4, strategy="louvain") # Match sizes len
    silos, stats = partitioner.partition(data)
    
    print("\nResults:")
    print(f"Cross-edge ratio: {stats['cross_edge_ratio']:.2%}")
    print(f"Nodes per silo: {stats['nodes_per_silo']}")
    
    # Check if we are in the target range (5-15%)
    ratio = stats['cross_edge_ratio']
    if 0.05 <= ratio <= 0.15:
        print("SUCCESS: Ratio is within 5-15% target.")
    else:
        print(f"WARNING: Ratio {ratio:.2%} is outside 5-15%.")

if __name__ == "__main__":
    test_louvain()
