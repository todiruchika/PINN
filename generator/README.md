# Generator

Run from the repository root. To add samples to an existing dataset, the size
arguments below are the number of **additional** samples for each split.

```bash
python generator/dataset_generator.py --train-size 420 --val-size 90 --test-size 90 --out-dir data/raw/dataset --seed 42 --append
```

With `--append`, existing samples and metadata are preserved, and new sample
names continue after the highest ID already present (for example, after
`thermal_00100`, the new IDs start at `thermal_00101`). The new IDs are
randomly assigned across train, validation, and test using the requested
per-split counts; they are not generated as three consecutive ID ranges.
Omit `--append` when creating a new dataset from scratch.

If an earlier interrupted run left sample files without matching metadata,
the generator stops instead of skipping those IDs or overwriting files. Move
or restore those incomplete samples before retrying.

To replace samples marked `thermal_violation=True` without changing their
sample IDs or split assignments, run:

```bash
python generator/dataset_generator.py --regenerate-violations --out-dir data/raw/dataset --seed 42 --max-attempts-per-sample 50
```

Each candidate uses a new floorplan and ambient temperature. A replacement is
saved only if it is at or below 125 °C and passes the validator's hard physical,
analytic-power, and sensor-noise checks. Fine-mesh replacements retain the
existing dataset's energy-balance behavior rather than being rejected by the
validator's 0.5% threshold, which the current fine-mesh data also exceeds.
Successful replacements are saved as they complete, so the command can be
rerun after an interruption. If no candidate passes within the attempt limit,
the command stops and reports the sample and failed checks.

Output goes to `data/raw/` (gitignored).
