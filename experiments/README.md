# Experiments

Naming convention: `experiments/<method>_<config>_seed<k>.py`

Examples:
- `experiments/pinn_baseline_seed0.py`
- `experiments/gappy_pod_n10_seed2.py`

No model implementations go here. Each experiment script imports from `src/` and writes results to `results/`.
