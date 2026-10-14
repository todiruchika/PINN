"""Mean +/- std over seeds of the data-driven PINN runs.

Reads every ``results/data_driven_pinn/*/seed_*.csv`` (one row per run, written by
``python -m src.train.data_driven``), groups by ``config`` (for example
``data_driven_pinn_N5`` or ``data_driven_pinn_N5_nophys``) and prints a markdown
table. Scores are means over the evaluated chips, so the std is the seed-to-seed
variation, which is what ``PROTOCOL.md`` asks to report.

    python experiments/aggregate_data_driven.py
    python experiments/aggregate_data_driven.py --root results/data_driven_pinn --out results/data_driven_pinn/summary.md
"""
import argparse
from pathlib import Path

import pandas as pd

COLUMNS = ["relative_l2", "mae_C", "rmse_C", "peak_temp_abs_error_C", "hotspot_localization_cells",
           "source_relative_l2", "n_params", "train_seconds"]


def aggregate(root: Path) -> str:
    files = sorted(root.glob("*/seed_*.csv"))
    if not files:
        return f"No seed CSV files found under {root}/*/"
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    cols = [c for c in COLUMNS if c in df.columns]
    lines = ["| config | seeds | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for cfg, g in df.groupby("config"):
        cells = []
        for c in cols:
            std = g[c].std(ddof=1) if len(g) > 1 else float("nan")
            cells.append(f"{g[c].mean():.4f} +/- {std:.4f}" if len(g) > 1 else f"{g[c].mean():.4f}")
        lines.append(f"| {cfg} | {len(g)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/data_driven_pinn")
    ap.add_argument("--out", default=None, help="also write the table to this file")
    args = ap.parse_args()
    table = aggregate(Path(args.root))
    print(table)
    if args.out:
        Path(args.out).write_text(table + "\n")
