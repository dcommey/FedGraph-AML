#!/usr/bin/env bash
# Download the Elliptic and IBM HI-Small source files and verify the IBM file hash.
# The IBM download uses the Kaggle API. Set up Kaggle credentials first.
set -euo pipefail
mkdir -p data/elliptic/raw data/ibm
(
  cd data/elliptic/raw
  for filename in elliptic_txs_features.csv elliptic_txs_edgelist.csv elliptic_txs_classes.csv; do
    if [[ ! -s "$filename" ]]; then
      curl --fail --location --retry 10 "https://data.pyg.org/datasets/elliptic/$filename.zip" -o "$filename.zip"
      unzip -o "$filename.zip"
    fi
  done
)
if [[ ! -s data/ibm/HI-Small_Trans.csv ]]; then
  curl --fail --location --retry 2 \
    'https://www.kaggle.com/api/v1/datasets/download/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/HI-Small_Trans.csv' \
    -o data/ibm/HI-Small_Trans.csv
fi
python3 - <<'PY'
import hashlib
from pathlib import Path
h = hashlib.sha256()
with Path('data/ibm/HI-Small_Trans.csv').open('rb') as f:
    for chunk in iter(lambda: f.read(1 << 20), b''):
        h.update(chunk)
expected = 'b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040'
if h.hexdigest() != expected:
    raise SystemExit('IBM file hash differs from the study file. Stop.')
print('IBM HI-Small file hash is correct.')
PY
