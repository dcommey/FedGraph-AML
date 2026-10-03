#!/usr/bin/env bash
# Run the complete study on one GPU host.
# Usage:
#   bash scripts/run_study.sh gtx          # first GPU: pilots, search, gates, main grid, secondary runs
#   bash scripts/run_study.sh rtx GATES    # second GPU: main grid with the gates from the first GPU
# GATES is a directory that contains the four gate files from the first GPU.
set -euo pipefail
hardware=${1:?give a hardware label, for example gtx}
gates=${2:-}
py=${PYTHON:-.venv/bin/python}
out="results/$hardware"
mkdir -p "$out" logs
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1

if [[ -z "$gates" ]]; then
  for ds in elliptic ibm; do
    "$py" -u experiments/refinement_study.py --phase pilot --dataset "$ds" --data-root "data/$ds" \
      --output "$out/$ds-pilot-v2" --rounds 5 --seeds 711 --ownership-seeds 20261003 --strategies random \
      --methods fedavg fresh centralized mlp fedmlp fedgcn > "logs/$hardware-$ds-pilot.log" 2>&1
    "$py" -u experiments/refinement_study.py --phase tune --dataset "$ds" --data-root "data/$ds" \
      --output "$out/$ds-tune-v2" --rounds 30 --seeds 711 --ownership-seeds 20261003 --strategies random \
      --methods fedavg centralized mlp fedmlp fedgcn > "logs/$hardware-$ds-tune.log" 2>&1
    "$py" -u experiments/refinement_trees.py --phase tune --dataset "$ds" --data-root "data/$ds" \
      --output "$out/$ds-trees-tune-v2" > "logs/$hardware-$ds-trees-tune.log" 2>&1
    "$py" experiments/build_tree_gate.py --tuning "$out/$ds-trees-tune-v2" --output "$out/$ds-trees-gate-v2.json"
    "$py" experiments/build_refinement_gate.py --dataset "$ds" --data-root "data/$ds" \
      --pilot "$out/$ds-pilot-v2" --tuning "$out/$ds-tune-v2" --output "$out/$ds-main-gate-v2.json"
  done
  gates="$out"
fi

for ds in elliptic ibm; do
  "$py" -u experiments/refinement_study.py --phase main --dataset "$ds" --data-root "data/$ds" \
    --output "$out/$ds-main-v2" --gate "$gates/$ds-main-gate-v2.json" > "logs/$hardware-$ds-main.log" 2>&1
  "$py" -u experiments/refinement_trees.py --phase main --dataset "$ds" --data-root "data/$ds" \
    --output "$out/$ds-trees-main-v2" --gate "$gates/$ds-trees-gate-v2.json" > "logs/$hardware-$ds-trees-main.log" 2>&1
done

if [[ "$gates" == "$out" ]]; then
  "$py" -u experiments/refinement_ablations.py --main-root "$out/elliptic-main-v2" \
    --gate "$out/elliptic-main-gate-v2.json" --output "$out/ablations-v2" > "logs/$hardware-ablations.log" 2>&1
  "$py" -u experiments/refinement_exposure.py --main-root "$out/elliptic-main-v2" \
    --gate "$out/elliptic-main-gate-v2.json" --output "$out/exposure-v2" > "logs/$hardware-exposure.log" 2>&1
  "$py" -u experiments/refinement_bank_ownership.py --gate "$out/ibm-main-gate-v2.json" \
    --output "$out/bank-ownership-v2" > "logs/$hardware-bank.log" 2>&1
  for ds in elliptic ibm; do
    "$py" experiments/refinement_data_audit.py --dataset "$ds" --main-root "$out/$ds-main-v2" \
      --gate "$out/$ds-main-gate-v2.json" --output "$out/$ds-data-audit-v2"
  done
fi
