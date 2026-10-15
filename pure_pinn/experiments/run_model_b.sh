#!/usr/bin/env bash
# Model B arm of the experiment matrix (Sec. 7), run in parallel.
#
# One process per (chip, sensor count, seed) solve, JOBS at a time, each writing
# its own CSV under results/parts/ -- concurrent appends to a shared file are not
# atomic. Merge afterwards with:
#
#     python src/evaluate.py --merge results/parts --plot
#
# Usage:
#   experiments/run_model_b.sh [N_CHIPS] [SENSORS] [SEEDS] [JOBS] [CONFIG]
#
#   experiments/run_model_b.sh 20 5,8,10 0,1,2 4          # the full arm, subsampled
#   experiments/run_model_b.sh 3  5,8,10 0,1   4          # quick trend check
#
# THREADS_PER_JOB keeps the total thread count near the core count; oversubscribed
# BLAS threads make every solve slower without finishing any sooner. Wall-clock
# numbers in metrics.csv are only comparable across runs made at the same JOBS
# setting -- state the setting when reporting the efficiency column (Sec. 6).
set -euo pipefail

cd "$(dirname "$0")/.."

N_CHIPS="${1:-20}"
SENSORS="${2:-5,8,10}"
SEEDS="${3:-0,1,2}"
JOBS="${4:-4}"
CONFIG="${5:-experiments/configs/model_b_default.json}"
SPLIT="${SPLIT:-test}"
THREADS_PER_JOB="${THREADS_PER_JOB:-2}"
# OVERRIDE: JSON merged into the config, for ablation arms. TAG: filename suffix.
#   OVERRIDE='{"model":{"bc_mode":"soft"}}' TAG=soft experiments/run_model_b.sh 3 5 0
OVERRIDE="${OVERRIDE:-}"
TAG="${TAG:-}"
# CHIPS: explicit comma-separated sample ids, overriding the N_CHIPS subsample.
#   CHIPS=thermal_00652,thermal_00664 experiments/run_model_b.sh 2 5,8,10 0
CHIPS="${CHIPS:-}"
# FIGS=1 (default) also writes the true/predicted/error triptych per solve, so the
# figures in a report always correspond to rows in metrics.csv. FIGS=0 to skip.
FIGS="${FIGS:-1}"
PARTS="${PARTS:-results/parts}"
LOGS="results/logs"
mkdir -p "$PARTS" "$LOGS"

# Resolve the chip list once, so every (N, seed) combination sees the same chips
# and the exact list is recorded for the writeup.
if [ -z "$CHIPS" ]; then
  CHIPS=$(python -c "
import sys; sys.path.insert(0, 'src')
from train_pinn import select_chips
print(','.join(select_chips('$SPLIT', None, $N_CHIPS, None)))
")
fi
echo "$CHIPS" | tr ',' '\n' > "results/model_b_chips_${SPLIT}.txt"

n_chips=$(echo "$CHIPS" | tr ',' '\n' | wc -l | tr -d ' ')
n_sens=$(echo "$SENSORS" | tr ',' '\n' | wc -l | tr -d ' ')
n_seed=$(echo "$SEEDS" | tr ',' '\n' | wc -l | tr -d ' ')
echo "Model B: $((n_chips * n_sens * n_seed)) solves on split=$SPLIT, ${JOBS} at a time"
echo "chip list -> results/model_b_chips_${SPLIT}.txt"

export OMP_NUM_THREADS="$THREADS_PER_JOB"
export MKL_NUM_THREADS="$THREADS_PER_JOB"

for chip in $(echo "$CHIPS" | tr ',' ' '); do
  for n in $(echo "$SENSORS" | tr ',' ' '); do
    for s in $(echo "$SEEDS" | tr ',' ' '); do
      stem="${chip}_N${n}_seed${s}${TAG:+_$TAG}"
      extra=""
      [ -n "$OVERRIDE" ] && extra="--override '$OVERRIDE'"
      [ -n "$TAG" ] && extra="$extra --tag $TAG"
      [ "$FIGS" = "1" ] && extra="$extra --save-figs --save-fields"
      printf '%s\n' "python -u src/train_pinn.py --config $CONFIG --split $SPLIT --chips $chip \
--sensors $n --seeds $s --quiet $extra --metrics $PARTS/${stem}.csv --runs-dir results/runs \
> $LOGS/${stem}.log 2>&1"
    done
  done
done | xargs -P "$JOBS" -I{} bash -c '{}'

echo
echo "done. merge and summarise with:"
echo "  python src/evaluate.py --merge $PARTS --plot"
