"""Run an existing trainer on the real IR-camera data instead of the synthetic chips.

The trainers read their data through load_splits / load_metadata / load_split_arrays. This runner swaps those three
names for versions backed by src.data.real_thermal, then calls the trainer unchanged. The real chips have no known
heat source or material constants, so use --w-phys 0 --w-src 0 (the runner refuses to run otherwise).

Examples:
    python -m src.train.real_runner fno      --sensors 10 --steps 3000 --out results/real/fno --w-phys 0 --w-src 0
    python -m src.train.real_runner fbpinn   --sensors 10 --steps 6000 --batch 32 --points 1024 --out results/real/fbpinn --w-phys 0 --w-src 0
    python -m src.train.real_runner xpinn    ...
    python -m src.train.real_runner hybrid   --mode ml --sensors 10 ...
"""
from __future__ import annotations

import sys

from src.data.real_thermal import RealThermal


def main(argv: list[str]) -> None:
    kind, rest = argv[0], argv[1:]
    if kind in ("fno", "fbpinn", "xpinn"):
        if "--w-phys" not in rest or rest[rest.index("--w-phys") + 1] != "0" \
                or "--w-src" not in rest or rest[rest.index("--w-src") + 1] != "0":
            raise SystemExit("real data has no heat source: pass --w-phys 0 --w-src 0")
    real = RealThermal()

    def load_splits():
        return real.splits

    def load_metadata():
        return None

    def load_split_arrays(ids, n_sensors, meta):
        return real.arrays(ids, n_sensors)

    if kind == "fno":
        import src.train.fno_pino as module
        parse, run = module.parse_args, module.train
        targets = [module]
    elif kind in ("fbpinn", "xpinn"):
        import src.train.fbpinn as module
        import src.train.fno_pino as fno_module
        parse, run = module.parse_args, module.train
        targets = [module, fno_module]
        if kind == "xpinn":
            rest = ["--model", "xpinn"] + rest
    elif kind == "hybrid":
        import src.train.phys_hybrid as module
        parse, run = module.parse_args, module.train
        targets = [module]
        import src.train.fno_pino as fno_module
        targets.append(fno_module)
    else:
        raise SystemExit(f"unknown trainer {kind}")
    for target in targets:
        for name, fn in (("load_splits", load_splits), ("load_metadata", load_metadata),
                         ("load_split_arrays", load_split_arrays)):
            if hasattr(target, name):
                setattr(target, name, fn)
    run(parse(rest))


if __name__ == "__main__":
    main(sys.argv[1:])
