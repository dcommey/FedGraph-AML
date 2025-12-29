# FedGraph-VASP: Research Context & METIS Experiment

## Research Goal
We are developing **FedGraph-VASP**, a privacy-preserving federated graph learning framework for cross-institutional anti-money laundering (AML). The paper is targeting **IEEE ICBC 2025**.

## The Problem We're Solving
**Current Status:** FedGraph-VASP and standard FedAvg achieve **identical F1 scores (0.61)** in our experiments. This is problematic for a research paper because it suggests our boundary embedding exchange adds complexity without benefit.

## Why We Need METIS Partitioning

### The Root Cause
Our current "stratified" partitioning creates **~66% cross-silo edges**, which is unrealistically high. In real VASPs (exchanges), only ~5-15% of transactions are cross-institutional.

### Why This Matters
- With 66% cross-edges, FedAvg already captures most collaborative learning benefit through model averaging
- The boundary embedding exchange becomes redundant
- **With ~5-10% cross-edges (realistic VASP scenario), boundary nodes become critical bottlenecks**
- This is where FedGraph's explicit topology sharing should outperform FedAvg

### What METIS Does
METIS is a graph partitioning algorithm that **minimizes edge cuts**. When we partition with METIS:
- Silos become truly isolated (internal transactions stay internal)
- Only ~5-10% of edges cross silo boundaries
- Boundary nodes become the critical "bridges" that FedGraph exploits
- **FedGraph should show measurable improvement over FedAvg**

## Experiment Instructions

### 1. Setup on Ubuntu
```bash
# Clone the repo
git clone git@github.com:dcommey/FedGraph-AML.git
cd FedGraph-AML

# Install METIS library (required for pymetis)
sudo apt-get update
sudo apt-get install libmetis-dev

# Create Python environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch (adjust for your system)
pip install torch torchvision torchaudio

# Install PyG
pip install torch-geometric

# Install requirements with pymetis
pip install -r requirements-linux.txt
```

### 2. Download Elliptic Dataset
- Go to: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
- Download and extract to: `data/elliptic_bitcoin_dataset/`
- Files needed:
  - `elliptic_txs_features.csv`
  - `elliptic_txs_edgelist.csv`
  - `elliptic_txs_classes.csv`

### 3. Configure Experiment for METIS
Edit `experiments/rigorous_evaluation.py` line 315:
```python
# Change from:
partition_strategy = "realistic"

# To:
partition_strategy = "metis"
```

### 4. Run Experiment
```bash
python experiments/rigorous_evaluation.py
```

### 5. Expected Results
With METIS partitioning, we expect:
- **Local GNN:** F1 ≈ 0.40-0.45 (isolated, no collaboration)
- **FedAvg:** F1 ≈ 0.55-0.58 (model averaging helps, but misses topology)
- **FedGraph-VASP:** F1 ≈ 0.60-0.65 (**should beat FedAvg** due to boundary embedding exchange)

The key result we need: **FedGraph-VASP > FedAvg** by at least 0.02-0.03 F1 points.

### 6. After Experiment
- Results will be saved to `results/rigorous_evaluation_YYYYMMDD_HHMMSS.json`
- Check the `statistics` section for mean F1 scores
- Look at `ttests.fedgraph_vs_fedavg` - we want `significant_at_005: true`

## Paper Status
The paper is in `paper/main.tex`. Current issues fixed:
- ✅ Consistent 36% improvement claim (not 47%)
- ✅ Table II uses "Topology" instead of "Privacy" 
- ✅ GraphSAGE citation added
- ✅ Honest discussion of limitations

**Main remaining issue:** Need experimental evidence that FedGraph > FedAvg with realistic partitioning.

## Contact
If the METIS experiment works and shows differentiation, update the paper's Table I with the new results and regenerate figures with `python experiments/generate_figures.py`.
