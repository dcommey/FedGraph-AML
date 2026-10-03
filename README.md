# FedGraph-VASP

This repository contains the code for the paper *FedGraph-VASP: Utility, Communication, and Recipient Exposure in Cross-Silo Transaction Screening*.

The study splits a transaction graph among three simulated owners. Each owner trains a GraphSAGE model. A coordinator sends first-layer node representations along the edges that cross owners. The study compares this exchange with FedAvg, local models, pooled models, graph-free models and a FedGCN adapter.

The study uses two data sources:

- IBM HI-Small, a synthetic anti-money-laundering data set. This is the primary source.
- Elliptic, a Bitcoin transaction data set. This is an exploratory source.

## Main result

| Source | Ownership | Step exchange minus FedAvg (AP) |
|---|---|---|
| IBM | Random | +0.0029 |
| IBM | Louvain | −0.0050 |
| Elliptic | Random | +0.0173 |
| Elliptic | Louvain | +0.0033 |

Graph-free tree models have the highest AP on both sources. The file `results/analysis/summary.json` contains all values.

## Repository contents

| Folder | Contents |
|---|---|
| `data/` | Data loaders, temporal splits and the IBM graph construction |
| `models/` | GraphSAGE with routed messages, MLP and the FedGCN adapter |
| `federated/` | Message format for the authenticated transport test |
| `experiments/` | Training, search, gates, interventions, attacks, audits and figures |
| `scripts/` | Setup, data download and the complete run sequence |
| `tests/` | Regression tests |
| `external/` | The FedGCN model file (MIT license) and the CDLA data license |
| `results/analysis/` | Per-run metrics and summaries from the 768 audited runs |

## Requirements

- Python 3.11
- An NVIDIA GPU with CUDA 12.1 for training
- PyTorch 2.4.1 and PyTorch Geometric 2.6.1

The file `requirements.txt` gives the direct dependencies. The file `requirements-lock-gpu.txt` gives the full training environment.

## Install

1. Create the environment:

   ```bash
   bash scripts/setup_environment.sh python3.11
   ```

2. Download the data. The IBM download uses the Kaggle API, so set up your Kaggle credentials first.

   ```bash
   bash scripts/fetch_data.sh
   ```

   The script stops if the IBM file hash is different from the study file.

## Run the tests

```bash
.venv/bin/python -m pytest
```

The data tests need the downloaded data.

## Make the figures

The figures use only the files in `results/analysis/`. A GPU is not necessary.

```bash
python experiments/make_manuscript_figures.py --analysis results/analysis --output figures
```

## Run the complete study

The study uses two GPU hosts. The first host selects the configurations. The second host repeats the main runs with the same configurations.

1. On the first GPU host, run all steps:

   ```bash
   bash scripts/run_study.sh gtx
   ```

2. Copy these four files from `results/gtx/` to the second host:
   `elliptic-main-gate-v2.json`, `ibm-main-gate-v2.json`, `elliptic-trees-gate-v2.json` and `ibm-trees-gate-v2.json`.

3. On the second GPU host, run the main grid with the copied gates:

   ```bash
   bash scripts/run_study.sh rtx path/to/gates
   ```

4. Put both result folders below one root, for example `results/study/gtx` and `results/study/rtx`.

5. Make the summaries:

   ```bash
   python experiments/summarize_refinements.py --root results/study --output results/analysis
   python experiments/summarize_refinement_secondary.py --root results/study/gtx --main-summary results/analysis/summary.json --output results/analysis
   ```

Each gate records the SHA-256 hash of the source files and the data. The runner stops if a hash changes. Do not edit the files in the gate list. If you change them, start a new study.

## Run the transport test

The transport test sends frozen representations between two hosts over mutual TLS 1.3.

1. Make new test credentials in a private folder outside the repository:

   ```bash
   python scripts/create_transport_probe_credentials.py --directory /tmp/fedgraph-keys
   ```

2. Prepare the snapshot from a trained Elliptic checkpoint:

   ```bash
   python experiments/refinement_transport.py --mode prepare --checkpoint path/to/checkpoint
   ```

3. Start the server on the source host. Start the client on the recipient host. Use the same `--run` and `--port` values on both hosts. Give the server address with `--host`.

Do not put the private keys in the repository.

## Data and licenses

- The FedGCN model file comes from https://github.com/yh-yao/FedGCN at commit `378438d`. It has the MIT license in `external/FedGCN/LICENSE`.
- The IBM data come from https://github.com/IBM/AML-Data under the CDLA-Sharing-1.0 license. The license text is in `external/CDLA-Sharing-1.0.txt`.
- The Elliptic data come from the provider through PyTorch Geometric. The provider terms apply.

This repository does not contain the raw data files.

See `THIRD_PARTY_NOTICES.md` for more information.
