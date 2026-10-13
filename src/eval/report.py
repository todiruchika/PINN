"""Aggregate per-seed CSVs into mean +/- std markdown tables."""
import pandas as pd
from pathlib import Path


def aggregate_results(results_dir: Path, out_path: Path = None) -> str:
    """Load all seed CSVs from results_dir, return markdown table."""
    dfs = [pd.read_csv(p) for p in sorted(results_dir.glob("seed_*.csv"))]
    if not dfs:
        return "No seed CSV files found."

    combined = pd.concat(dfs)
    numeric_cols = combined.select_dtypes(include="number").columns
    mean = combined.groupby("config")[numeric_cols].mean()
    std = combined.groupby("config")[numeric_cols].std()

    rows = []
    for cfg in mean.index:
        row = {"config": cfg}
        for col in numeric_cols:
            row[col] = f"{mean.loc[cfg, col]:.4f} +/- {std.loc[cfg, col]:.4f}"
        rows.append(row)

    df_out = pd.DataFrame(rows)
    md = df_out.to_markdown(index=False)
    if out_path:
        out_path.write_text(md)
    return md
