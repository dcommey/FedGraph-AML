# FedGraph-VASP

Public code release for FedGraph-VASP, a privacy-preserving federated graph learning framework for cross-institutional anti-money laundering.

FedGraph-VASP allows collaborating Virtual Asset Service Providers (VASPs) to train graph models without sharing raw transaction data. The framework combines boundary embedding exchange with a post-quantum secure tunnel based on Kyber-512 and AES-256-GCM.

## What this repository includes

- Federated training logic in `federated/`
- GNN and baseline models in `models/`
- Experiment drivers in `experiments/`
- Data loaders and partitioning utilities in `data/`
- Tests and configuration needed to reproduce the code workflows

## What is intentionally excluded from GitHub

- Raw and processed datasets under `data/elliptic/` and `data/ethereum/`
- Generated outputs under `results/`
- Manuscript folders, figures, and submission artifacts
- Local virtual environments, caches, logs, and LaTeX build products

> The `data/` Python package is part of the source tree and contains loader code only. No dataset files are committed.

## Reported snapshot

| Partition | Local GNN | FedSage+ | FedAvg | **FedGraph-VASP** |
| --------- | --------- | -------- | ------ | ------------------ |
| Louvain (seed 42, sparse cross-silo edges) | 0.336 | 0.411 | 0.438 | **0.446** |
| METIS (seed 42, high-connectivity reference) | 0.387 | 0.464 | 0.603 | **0.604** |

In the five-seed METIS study, FedAvg ($0.626 \pm 0.008$) and FedGraph-VASP ($0.620 \pm 0.009$) were statistically indistinguishable ($p = 0.119$).

## Installation

### Prerequisites

- Python 3.8+
- PyTorch-compatible environment
- Optional GPU support through CUDA

### Setup

```bash
git clone https://github.com/dcommey/FedGraph-AML.git
cd FedGraph-AML

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

For Linux METIS support, install the system package first:

```bash
sudo apt-get install libmetis-dev
```

## Dataset setup

Download the third-party datasets referenced in the manuscript and place them locally in:

- `data/elliptic/`
- `data/ethereum/`

## Reproducing core workflows

```bash
# Core federated run
python experiments/run_federated.py

# Multi-seed evaluation
python experiments/rigorous_evaluation.py

# Privacy and PQC measurements
python experiments/privacy_analysis.py
python experiments/pqc_benchmark.py
```

## Citation

If the journal version is not yet available, cite the public code release:

> Daniel Commey, Matilda Nkoom, Yousef Alsenani, Sena G. Hounsinou, and Garth V. Crosby. *FedGraph-AML: FedGraph-VASP implementation*. GitHub repository, 2026. <https://github.com/dcommey/FedGraph-AML>
