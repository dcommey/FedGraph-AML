# FedGraph-VASP

**Privacy-Preserving Federated Graph Learning with Post-Quantum Security for Cross-Institutional Anti-Money Laundering**

FedGraph-VASP is a federated learning framework enabling Virtual Asset Service Providers (VASPs) to collaboratively detect money laundering patterns without sharing raw transaction data. It addresses the "cross-chain blind spot" by exchanging encrypted boundary embeddings, secured against future quantum threats using a hybrid Kyber-512 + AES-256-GCM tunnel.

## Key Features

-   **Boundary Embedding Exchange**: Shares compressed, non-invertible topological information for cross-silo edges.
-   **Post-Quantum Security**: Implements NIST-standardized ML-KEM (Kyber-512) for long-term data protection.
-   **Privacy-First**: No raw features or personally identifiable information leave the institution.

## Results on Elliptic Dataset

| Method | F1-Score | vs Local |
|--------|----------|----------|
| Local GNN | 0.48 ± 0.01 | - |
| FedAvg | 0.63 ± 0.01 | +30% |
| **FedGraph-VASP** | **0.63 ± 0.01** | **+30%** |

*FedGraph-VASP achieves parity with centralized baselines while preserving strict data localization.*

## Installation

### Prerequisites
-   Python 3.8+
-   PyTorch 1.12+
-   CUDA 11.8+ (optional, for GPU acceleration)

### Setup
```bash
git clone https://github.com/<your-username>/FedGraph-AML.git
cd FedGraph-AML

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install -r requirements.txt
```

> **Note for Linux Users:** To use METIS partitioning (recommended), install `libmetis-dev`:
> `sudo apt-get install libmetis-dev`

## Usage

```bash
# 1. Run rigorous evaluation (5 seeds, 50 rounds)
python experiments/rigorous_evaluation.py

# 2. Generate publication figures
python experiments/generate_figures.py
```

## Citation

Citation will be added upon acceptance.
