# FedGraph-VASP

**Privacy-Preserving Federated Graph Learning with Post-Quantum Security for Cross-Institutional Anti-Money Laundering**

A novel federated learning framework that enables multiple Virtual Asset Service Providers (VASPs) to collaboratively detect money laundering without sharing raw transaction data, secured with post-quantum cryptography.

## Key Innovations

1. **Boundary Embedding Exchange** - Unlike FedSage+ which *generates* missing neighbors, FedGraph-VASP *exchanges* encrypted embeddings for boundary nodes, providing authentic cross-silo topology information.

2. **Post-Quantum Security** - All embedding exchanges are secured with NIST-standardized Kyber-512 + AES-256-GCM, protecting against future quantum adversaries.

3. **Privacy-Preserving** - Only compressed, non-invertible embeddings are shared - no raw transaction data or model gradients leave institutional boundaries.

## Installation

### Windows (CPU/CUDA)
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/FedGraph-AML.git
cd FedGraph-AML

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyG and other dependencies
pip install torch-geometric
pip install -r requirements.txt
```

### Ubuntu/Linux (with PyMETIS for better partitioning)
```bash
# Install METIS library first
sudo apt-get update
sudo apt-get install libmetis-dev

# Clone and setup
git clone https://github.com/YOUR_USERNAME/FedGraph-AML.git
cd FedGraph-AML

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch
pip install torch torchvision torchaudio

# Install PyG 
pip install torch-geometric

# Install requirements with PyMETIS
pip install -r requirements-linux.txt
```

## Quick Start

```bash
# Quick test (3 rounds, minimal epochs)
python main.py --quick

# Run rigorous evaluation (5 seeds, 30 rounds)
python experiments/rigorous_evaluation.py

# Generate figures for paper
python experiments/generate_figures.py
```

## Partitioning Strategies

The choice of graph partitioning significantly impacts results:

| Strategy | Cross-Edges | Use Case |
|----------|-------------|----------|
| `stratified` | ~66% | High connectivity baseline |
| `realistic` | ~20-30% | BFS-based community detection |
| `metis` | ~5-10% | **Best for Linux** - minimizes edge cuts |

To use METIS partitioning (Linux only):
```python
# In experiments/rigorous_evaluation.py, change:
partition_strategy = "metis"  # Requires pymetis
```

## Project Structure

```
FedGraph-AML/
├── data/
│   ├── elliptic_loader.py    # Elliptic dataset loader
│   └── partitioner.py        # Graph partitioning strategies
│
├── models/
│   └── gnn.py                # GraphSAGE models
│
├── federated/
│   ├── client.py             # Federated client
│   ├── server.py             # Federated server (FedAvg)
│   └── boundary_exchange.py  # PQC-secured embedding exchange
│
├── experiments/
│   ├── rigorous_evaluation.py # Main experiment script
│   └── generate_figures.py    # Publication figures
│
├── paper/                     # LaTeX paper source
└── results/                   # Experiment outputs (JSON)
```

## Results (Elliptic Bitcoin Dataset)

| Method | F1-Score | vs Local |
|--------|----------|----------|
| Local GNN | 0.47 ± 0.02 | - |
| FedAvg | 0.61 ± 0.01 | +32% |
| **FedGraph-VASP** | **0.62 ± 0.01** | **+32%** |

Both federated methods achieve comparable performance to centralized GCN (0.61) while preserving privacy.

## Hardware Requirements

- **Minimum**: CPU with 8GB RAM
- **Recommended**: NVIDIA GPU with 4GB+ VRAM
- **For METIS**: Linux with libmetis-dev

## Citation

```bibtex
@inproceedings{fedgraph-vasp,
  title={FedGraph-VASP: Privacy-Preserving Federated Graph Learning 
         with Post-Quantum Security for Cross-Institutional Anti-Money Laundering},
  author={Commey, Daniel and Alsenani, Yousef and Crosby, Garth V.},
  booktitle={IEEE International Conference on Blockchain and Cryptocurrency (ICBC)},
  year={2025}
}
```

## References

- Weber et al. (2019). "Anti-money laundering in Bitcoin: Experimenting with graph convolutional networks"
- McMahan et al. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data"
- CRYSTALS-Kyber: NIST Post-Quantum Cryptography Standard
