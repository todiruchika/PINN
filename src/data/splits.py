"""Read splits.csv and return sample ID lists for each split."""
import pandas as pd
from pathlib import Path

SPLITS_CSV = Path("data/raw/dataset/splits.csv")
SPLITS_DIR = Path("data/splits")


def load_splits(splits_csv: Path = SPLITS_CSV) -> dict:
    df = pd.read_csv(splits_csv)
    return {
        split: sorted(df.loc[df["split"] == split, "sample_id"].tolist())
        for split in ["train", "validation", "test"]
    }


def save_split_files(splits: dict, out_dir: Path = SPLITS_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, ids in splits.items():
        (out_dir / f"{split}.txt").write_text("\n".join(ids) + "\n")
