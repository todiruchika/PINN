#!/usr/bin/env bash
# Train and score the data-driven PINN for several sensor counts and seeds, one after another.
#
#   experiments/run_data_driven.sh                       # N = 5,8,10 x seeds 0..4 (the protocol grid)
#   experiments/run_data_driven.sh "5 10" "0 1"          # N = 5,10 x seeds 0,1
#   EXTRA="--steps 50000" experiments/run_data_driven.sh "10" "0"
#   EXTRA="--w-phys 0" experiments/run_data_driven.sh    # the no-physics ablation (outputs end in _nophys)
#
# Each run writes results/data_driven_pinn/N<N>_seed<k>[_nophys]/ (checkpoints, CSVs, curves).
# Re-running a finished run overwrites it; a crashed run can be continued with
#   python -m src.train.data_driven --sensors N --seed k --resume <run dir>/last.pt
# Summarise afterwards with: python experiments/aggregate_data_driven.py
set -euo pipefail
cd "$(dirname "$0")/.."

SENSORS="${1:-5 8 10}"
SEEDS="${2:-0 1 2 3 4}"
EXTRA="${EXTRA:-}"

for n in $SENSORS; do
  for s in $SEEDS; do
    echo "=== N=$n seed=$s ${EXTRA} ==="
    # shellcheck disable=SC2086
    python -m src.train.data_driven --sensors "$n" --seed "$s" $EXTRA
  done
done
python experiments/aggregate_data_driven.py
