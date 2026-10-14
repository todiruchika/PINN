from pathlib import Path
from typing import Dict, Optional

import pandas as pd

COLUMNS = ["peak_temp_C", "T_amb_C", "block_count", "thermal_violation"]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DELIVERED = REPO_ROOT / "data" / "raw" / "dataset" / "metadata.csv"
DEFAULT_EXTENDED = REPO_ROOT / "data" / "raw_extended"
DEFAULT_REPORT = REPO_ROOT / "results" / "validation" / "coverage_report.md"
SPLITS = ["train", "val", "test"]


def load_extended_metadata(root: Path) -> Optional[pd.DataFrame]:
    files = sorted(root.glob("*/metadata.csv")) if root.is_dir() else []
    if not files:
        return None
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def count_matrices(root: Path) -> Dict[str, int]:
    counts = {}
    for split in SPLITS:
        d = root / split / "matrices"
        counts[split] = len([p for p in d.glob("thermal_*.npy") if not p.stem.endswith("_Q")]) if d.is_dir() else 0
    return counts


def summarize(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out = {}
    for col in COLUMNS:
        s = df[col].astype(float)
        out[col] = {"n": int(s.count()), "min": s.min(), "median": s.median(),
                    "mean": s.mean(), "max": s.max()}
    return out


def build_report(delivered: pd.DataFrame, extended: Optional[pd.DataFrame],
                 matrix_counts: Dict[str, int]) -> str:
    lines = ["# Coverage Report: delivered 100 vs extended dataset", ""]
    d = summarize(delivered)

    if extended is None:
        lines += [
            "**Extended metadata not available.** No `*/metadata.csv` found under "
            "`data/raw_extended/`; per-split metadata is written only when a split finishes.",
            "",
            "Ground-truth matrices present so far (excluding `_Q` files):",
            "",
            "| Split | Matrices |",
            "|-------|----------|",
        ]
        lines += [f"| {s} | {matrix_counts.get(s, 0)} |" for s in SPLITS]
        lines += ["", "## Delivered 100 samples", "", _stats_table(d)]
        return "\n".join(lines) + "\n"

    e = summarize(extended)
    lines += [f"Delivered n = {d[COLUMNS[0]]['n']}, extended n = {e[COLUMNS[0]]['n']}.", ""]
    lines += ["| Column | Source | n | min | median | mean | max | Delivered range inside extended range |",
              "|--------|--------|---|-----|--------|------|-----|----------------------------------------|"]
    for col in COLUMNS:
        covered = e[col]["min"] <= d[col]["min"] and e[col]["max"] >= d[col]["max"]
        for name, s in (("delivered", d), ("extended", e)):
            row = s[col]
            tail = ("yes" if covered else "no") if name == "extended" else ""
            lines.append(f"| {col} | {name} | {row['n']} | {row['min']:.4g} | {row['median']:.4g} | "
                         f"{row['mean']:.4g} | {row['max']:.4g} | {tail} |")
    lines += ["", "For `thermal_violation`, min/median/max are 0/1 and mean is the violation rate."]
    return "\n".join(lines) + "\n"


def _stats_table(d: Dict[str, Dict[str, float]]) -> str:
    rows = ["| Column | n | min | median | mean | max |", "|--------|---|-----|--------|------|-----|"]
    for col in COLUMNS:
        r = d[col]
        rows.append(f"| {col} | {r['n']} | {r['min']:.4g} | {r['median']:.4g} | {r['mean']:.4g} | {r['max']:.4g} |")
    return "\n".join(rows)


def main() -> None:
    delivered = pd.read_csv(DEFAULT_DELIVERED)
    extended = load_extended_metadata(DEFAULT_EXTENDED)
    report = build_report(delivered, extended, count_matrices(DEFAULT_EXTENDED))
    DEFAULT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_REPORT.write_text(report, encoding="utf-8")
    print(f"Wrote {DEFAULT_REPORT}")


if __name__ == "__main__":
    main()
