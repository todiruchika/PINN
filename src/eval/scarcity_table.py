"""Table of the data-scarcity experiment: XPINN with and without physics on few training chips.

Reads results/scarce/<variant>_<chips>/N10_seed0/results.json and writes results/comparison/scarcity.md.

Usage (from the repo root):
    python -m src.eval.scarcity_table
"""
from __future__ import annotations

import json
from pathlib import Path

VARIANTS = {
    "xpinn_nophys": "no physics",
    "xpinn_phys": "physics, pointwise (tile edges masked)",
    "xpinn_pool16": "physics, 16x16 patch-averaged (tile edges masked)",
}
CHIPS = [25, 50]
# Same model trained on all 500 chips (seed 0, 6000 steps, batch 32): reference rows.
FULL_REFERENCE = {
    "no physics": Path("results/xpinn_fast_wphys0/N10_seed0/results.json"),
    "physics, pointwise (tile edges masked)": Path("results/xpinn_fast_seam/N10_seed0/results.json"),
    "physics, 16x16 patch-averaged (tile edges masked)": Path("results/xpinn_fast_seam_pool16/N10_seed0/results.json"),
}


def read(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> None:
    rows = ["| Physics term | 25 training chips | 50 training chips | 500 training chips |", "|---|---|---|---|"]
    for key, label in VARIANTS.items():
        cells = []
        for n in CHIPS:
            run = read(Path(f"results/scarce/{key}_{n}/N10_seed0/results.json"))
            cells.append(f"{run['test']['relative_l2'] * 100:.2f}%" if run else "not finished")
        full = read(FULL_REFERENCE[label])
        cells.append(f"{full['test']['relative_l2'] * 100:.2f}%" if full else "not finished")
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    out = Path("results/comparison/scarcity.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
