# Third-party notices

## FedGCN

The file `external/FedGCN/src/gnn_models.py` comes from https://github.com/yh-yao/FedGCN at commit 378438d0a5dbfa0f1cf859a3d6f3aaa4cff232a3. We did not change this file. The MIT license is in `external/FedGCN/LICENSE`.

The adapter in `models/fedgcn_reference.py` changes the task to directed, temporal binary classification. It does not use the encrypted mode of FedGCN.

## IBM AML data

The IBM HI-Small data come from https://github.com/IBM/AML-Data. Credit: Altman et al., *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models*, NeurIPS 2023.

The data license is the Community Data License Agreement – Sharing, Version 1.0 (https://cdla.dev/sharing-1-0/). The full text is in `external/CDLA-Sharing-1.0.txt`.

This repository does not contain the IBM transaction file. The study uses a fixed sample of 1–10 September 2022. Files that contain IBM row IDs or labels, together with model predictions, are derived data under CDLA-Sharing-1.0.

## Elliptic data

The Elliptic data come from the provider through the PyTorch Geometric loader. The provider terms apply. This repository does not contain the Elliptic files.
