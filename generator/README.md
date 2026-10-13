# Generator

Run from the repository root. To add samples to an existing dataset, the size
arguments below are the number of **additional** samples for each split.

```bash
python generator/dataset_generator.py --train-size 420 --val-size 90 --test-size 90 --out-dir data/raw/dataset --seed 42 --append
```

With `--append`, existing samples and metadata are preserved, and new sample
names continue after the highest ID already present (for example, after
`thermal_00100`, the next ID is `thermal_00101`). Omit `--append` when creating
a new dataset from scratch.

Output goes to `data/raw/` (gitignored).
