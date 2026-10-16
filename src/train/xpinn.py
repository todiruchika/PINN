"""Train and evaluate the sensor-conditioned XPINN (same trainer as FBPINN, model = xpinn).

Example:
    python -m src.train.xpinn --sensors 5 --seed 0 --steps 10000 --batch 32 --points 1024
"""
import sys

from src.train.fbpinn import parse_args, train

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--model" not in argv:
        argv = ["--model", "xpinn"] + argv
    if "--out" not in argv:
        argv = argv + ["--out", "results/xpinn"]
    train(parse_args(argv))
