# Paper: hybrid DeepONet–PINN thermal map reconstruction

Draft manuscript for **IEEE Transactions on Instrumentation and Measurement** (Q1, JIF 7.0, hybrid open access, so no publication charge is required).

Build: `latexmk -pdf main.tex` (needs IEEEtran, TikZ, siunitx). Figures: `python scripts/make_figures.py`.

## Why this journal
The paper is about reconstructing a full field from a few noisy sensors (virtual sensing), with a realistic sensor model and measured IR data.
That is a measurement problem, which matches T-IM's scope. Backups from the same list:
- **Advanced Modeling and Simulation in Engineering Sciences** (Springer, Q1, JIF 3.2, open access with a fee): its scope is physics-informed and data-driven simulation. Needs the Springer Nature template.
- **IEEE Access** (Q2, JIF 4.2): faster review, if a Q1 attempt is rejected.

## Where every number comes from
| Paper item | Source (branch: file) |
|---|---|
| Table II (DeepONet, physics-informed DeepONet) | `smriti-hybrid(pinn+deeponet)`: `results/model_a/summary.md`, `results/model_c_summary.md` |
| Tables II-IV (PINN, blend, peak/hotspot, paired) | `smriti-hybrid(pinn+deeponet)`: `results/blend/summary.md`, `results/blend/paired_analysis.md` |
| Table V, Fig. 3 (simulated low data) | `smriti-hybrid(pinn+deeponet)`: `results/low_data/summary.md` |
| Fig. 2 | `smriti-hybrid(pinn+deeponet)`: `results/model_a/maps_n10_seed0.png` |
| Table VI, Fig. 4 (real, all test frames) | `github_dataset-hybrid`: `results/real_hybrid/summary_hybrid.csv`, `paired_comparison.csv`, `results/real_deeponet/summary_i5_i7.csv`, `comparison_maps_i5_i7_n8.png` |
| Tables VII-VIII, Fig. 5 (real, per-frame PINN, few workloads) | `github_dataset-hybrid`: `results/real_pinn_hybrid/summary.csv`, `paired_comparison.csv`, README; gappy POD and train-mean on sampled frames from `siya-real-fair`: `results/real_fbpinn/summary.csv` |
| Section IX, Table IX (other architectures) | `siya-fbpinn`: `results/comparison/summary.md`, `RESULTS.md`, `scarcity.md`, `results/phys_*/N10_seed0/results.json`; real FBPINN and FNO cost: `siya-real-fair` README; data-driven PINN: `data-driven-pinn`: `DATA_DRIVEN_PINN.md` §5 |
| Residual check (Section VIII-B) | `siya-fbpinn`: `results/comparison/RESULTS.md` §4.2 |
| Dataset, sensors, noise | `main`: `PROTOCOL.md` |

## Before submission (red TODOs in the PDF)
- [ ] Author order and corresponding author (currently the report's order, guide last; the core results are Smriti's).
- [ ] Confirm `t_die`, `h_eff` and the power ranges (PROTOCOL.md marks them as unsourced assumptions).
- [ ] Check every entry in `refs.bib` against the publisher record.
- [ ] Code/data availability statement and repository URL; acknowledgements.
- [ ] Read T-IM's author guidelines (page limit, graphical abstract, cover letter).
