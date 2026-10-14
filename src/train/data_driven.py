"""Train and evaluate the data-driven PINN: one network across all training chips.

The network (``src/models/conditional_pinn.py``) maps a chip's N noisy sensor
readings, plus its known ambient temperature, to the full temperature field and
to the heat-source field. Training combines

* **data** terms: the temperature and source heads against the ground-truth
  fields of the 500 training chips, and
* a **physics** term: the PDE residual of the predicted pair, evaluated at random
  collocation points (so it also constrains the field *between* grid nodes).

The physics weight can be set to 0 (``--w-phys 0``) for the purely data-driven
ablation. At test time only the sensor readings and ``T_amb`` are used.

Examples (from the repository root)::

    python -m src.train.data_driven --sensors 5
    python -m src.train.data_driven --sensors 10 --seed 1 --steps 40000
    python -m src.train.data_driven --sensors 5 --w-phys 0          # no-physics ablation
    python -m src.train.data_driven --sensors 5 --eval-only --out results/data_driven_pinn/N5_seed0

See ``DATA_DRIVEN_PINN.md`` for the full list of options and outputs.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.arrays import SplitArrays, load_split_arrays
from src.data.dataset import GRID
from src.eval import metrics as M
from src.models.conditional_pinn import ConditionalPINN
from src.physics.autodiff import Physics, pde_residual
from src.sensors.readers import sensor_mask

DTYPE = torch.float32
SENSOR_COUNTS = (5, 8, 10)


# --------------------------------------------------------------------------- #
# arguments
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Data-driven PINN: one network trained across chips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sensors", type=int, required=True, choices=SENSOR_COUNTS, help="number of sensors N")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="output dir (default results/data_driven_pinn/N<N>_seed<seed>[_nophys])")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0 ...")

    g = p.add_argument_group("optimisation")
    g.add_argument("--steps", type=int, default=30000)
    g.add_argument("--batch-chips", type=int, default=32, help="chips per step")
    g.add_argument("--points-data", type=int, default=1024, help="supervised grid points per chip per step")
    g.add_argument("--points-phys", type=int, default=256, help="PDE collocation points per chip per step")
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--lr-min", type=float, default=1e-5, help="cosine decay target")
    g.add_argument("--grad-clip", type=float, default=1.0, help="0 disables")

    g = p.add_argument_group("loss")
    g.add_argument("--w-temp", type=float, default=1.0, help="weight of the temperature data term")
    g.add_argument("--w-src", type=float, default=1.0, help="weight of the source data term")
    g.add_argument("--w-phys", type=float, default=0.01, help="weight of the PDE residual; 0 = data-driven ablation")
    g.add_argument("--phys-warmup", type=float, default=0.2, help="fraction of steps over which w-phys ramps up from 0")
    g.add_argument("--no-noise-aug", action="store_true",
                   help="train on the stored noisy readings instead of re-simulating sensor noise every step")

    g = p.add_argument_group("model")
    g.add_argument("--latent", type=int, default=64)
    g.add_argument("--branch-width", type=int, default=128)
    g.add_argument("--branch-depth", type=int, default=3)
    g.add_argument("--trunk-width", type=int, default=128)
    g.add_argument("--trunk-depth", type=int, default=4)
    g.add_argument("--activation", default="tanh", choices=["tanh", "silu", "gelu"])
    g.add_argument("--temp-scale", type=float, default=50.0, help="fixed temperature scale S in K")
    g.add_argument("--soft-neumann", action="store_true",
                   help="feed raw (x, y) to the trunk instead of cos(pi x), cos(pi y); edges are then not exactly adiabatic")

    g = p.add_argument_group("evaluation / bookkeeping")
    g.add_argument("--eval-every", type=int, default=1000, help="validate every K steps")
    g.add_argument("--eval-chips", type=int, default=0, help="validation chips used during training (0 = all 100)")
    g.add_argument("--log-every", type=int, default=100)
    g.add_argument("--eval-splits", default="validation,test", help="comma-separated splits scored at the end")
    g.add_argument("--save-figs", type=int, default=0, metavar="K", help="save true/pred/error pictures for K chips per eval split")
    g.add_argument("--resume", default=None, help="checkpoint (.pt) to continue training from")
    g.add_argument("--eval-only", action="store_true", help="skip training; score --ckpt (default <out>/best.pt)")
    g.add_argument("--ckpt", default=None, help="checkpoint for --eval-only")
    return p


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def default_out(args) -> Path:
    tag = "_nophys" if args.w_phys == 0 else ""
    return Path("results/data_driven_pinn") / f"N{args.sensors}_seed{args.seed}{tag}"


def config_name(args) -> str:
    return f"data_driven_pinn_N{args.sensors}" + ("_nophys" if args.w_phys == 0 else "")


# --------------------------------------------------------------------------- #
# training data on device
# --------------------------------------------------------------------------- #
def grid_xy(device) -> torch.Tensor:
    """``(16384, 2)`` points ``(x, y)`` in ``[0, 1]^2``, row-major like the matrices (row = y)."""
    axis = torch.linspace(0.0, 1.0, GRID, dtype=DTYPE, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)


class TrainData:
    """The training split as device tensors, normalised for the network."""

    def __init__(self, ds: SplitArrays, physics: Physics, temp_scale: float, device):
        t = lambda a: torch.as_tensor(a, dtype=DTYPE, device=device)
        self.n = len(ds)
        self.T_amb = t(ds.T_amb)                                          # (n,)
        self.sensors_true = t(ds.sensors_true)                            # (n, N)
        self.sensors_noisy = t(ds.sensors_noisy)                          # (n, N)
        self.offsets = t(ds.noise.offsets_C)
        self.sigma = float(ds.noise.sigma_C)
        self.quant = float(ds.noise.quant_step_C)
        T = t(ds.T).reshape(self.n, -1)
        Q = t(ds.Q).reshape(self.n, -1)
        self.theta = (T - self.T_amb[:, None]) / temp_scale                # (n, 16384)
        self.q_tilde = Q / physics.source_scale(temp_scale)                # (n, 16384)
        self.xy = grid_xy(device)

    def sensors(self, idx: torch.Tensor, noise_aug: bool, gen: torch.Generator) -> torch.Tensor:
        """Readings for chips ``idx``: fresh simulated noise, or the stored noisy ones."""
        if not noise_aug:
            return self.sensors_noisy[idx]
        true = self.sensors_true[idx]
        noise = torch.randn(true.shape, generator=gen, device=true.device, dtype=true.dtype)
        noisy = true + self.offsets + self.sigma * noise
        return torch.round(noisy / self.quant) * self.quant


# --------------------------------------------------------------------------- #
# one optimisation step
# --------------------------------------------------------------------------- #
def training_losses(model, data: TrainData, physics: Physics, args, w_phys: float, gen) -> dict:
    device = data.T_amb.device
    B, Pd, Pc = args.batch_chips, args.points_data, args.points_phys
    idx = torch.randint(0, data.n, (B,), generator=gen, device=device)
    T_amb = data.T_amb[idx]
    cond = model.condition(data.sensors(idx, not args.no_noise_aug, gen), T_amb)

    # ---- supervised terms on random grid nodes ---------------------------- #
    node = torch.randint(0, GRID * GRID, (B, Pd), generator=gen, device=device)
    theta_p, q_p = model(cond, data.xy[node])                              # (B, Pd)
    l_temp = torch.mean((theta_p - torch.gather(data.theta[idx], 1, node)) ** 2)
    l_src = torch.mean((q_p - torch.gather(data.q_tilde[idx], 1, node)) ** 2)
    total = args.w_temp * l_temp + args.w_src * l_src

    # ---- physics residual at continuous collocation points ---------------- #
    l_phys = torch.zeros((), device=device)
    if w_phys > 0:
        xy_c = torch.rand((B * Pc, 2), generator=gen, device=device, dtype=DTYPE)
        coef_rows = model.coefficients(cond).repeat_interleave(Pc, dim=0)  # (B*Pc, 2, L+1)
        res = pde_residual(model.rows_fn(coef_rows), xy_c,
                           T_amb.repeat_interleave(Pc)[:, None], physics, args.temp_scale)
        l_phys = torch.mean(res**2)
        total = total + w_phys * l_phys
    return {"total": total, "temp": l_temp, "src": l_src, "phys": l_phys}


def _join(history: pd.DataFrame, rows: list) -> pd.DataFrame:
    new = pd.DataFrame(rows)
    return new if history.empty else pd.concat([history, new], ignore_index=True)


def lr_at(step: int, total: int, lr: float, lr_min: float) -> float:
    """Cosine decay from ``lr`` to ``lr_min`` (closed form, so resuming needs no scheduler state)."""
    frac = min(max(step / max(total, 1), 0.0), 1.0)
    return lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * frac))


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model, ds: SplitArrays, physics: Physics, device, chunk: int = 50):
    """Predicted ``(T degC, Q W/m^2)`` for every chip, each ``(n, 128, 128)``, from the stored noisy readings."""
    model.eval()
    xy = grid_xy(device)
    T_amb = torch.as_tensor(ds.T_amb, dtype=DTYPE, device=device)
    sens = torch.as_tensor(ds.sensors_noisy, dtype=DTYPE, device=device)
    Ts, Qs = [], []
    for i in range(0, len(ds), chunk):
        sl = slice(i, i + chunk)
        theta, q = model(model.condition(sens[sl], T_amb[sl]), xy)       # (b, 16384)
        Ts.append((T_amb[sl, None] + model.temp_scale * theta).reshape(-1, GRID, GRID).cpu().numpy())
        Qs.append((q * physics.source_scale(model.temp_scale)).reshape(-1, GRID, GRID).cpu().numpy())
    return np.concatenate(Ts), np.concatenate(Qs)


def score(ds: SplitArrays, T_pred: np.ndarray, Q_pred: np.ndarray) -> pd.DataFrame:
    """Per-chip metrics on the non-sensor nodes, using the shared harness in ``src/eval/metrics.py``."""
    keep = ~sensor_mask(ds.positions_m)                                    # True = scored node
    rows = []
    for i, sid in enumerate(ds.ids):
        m = M.all_metrics(T_pred[i], ds.T[i], keep)
        m["peak_temp_abs_error_C"] = abs(m["peak_temp_error_C"])
        m["source_relative_l2"] = M.relative_l2(Q_pred[i], ds.Q[i], keep)
        rows.append({"sample_id": sid, **m})
    return pd.DataFrame(rows)


def mean_row(df: pd.DataFrame) -> dict:
    return {c: float(df[c].mean()) for c in df.columns if c != "sample_id"}


# --------------------------------------------------------------------------- #
# checkpoints / outputs
# --------------------------------------------------------------------------- #
def save_checkpoint(path: Path, model, opt, step: int, best: float, args) -> None:
    torch.save({"model": model.state_dict(), "model_cfg": model.cfg, "opt": opt.state_dict() if opt else None,
                "step": step, "best_val": best, "args": vars(args)}, path)


def load_model(path: Path, device) -> tuple[ConditionalPINN, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = ConditionalPINN.from_config(ck["model_cfg"]).to(device)
    model.load_state_dict(ck["model"])
    return model, ck


def save_triptychs(ds: SplitArrays, T_pred: np.ndarray, out_dir: Path, k: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    keep = ~sensor_mask(ds.positions_m)
    for i in range(min(k, len(ds))):
        true, pred = ds.T[i], T_pred[i]
        rel = M.relative_l2(pred, true, keep)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        lo, hi = float(true.min()), float(true.max())
        for ax, data, title, cmap, kw in (
            (axes[0], true, "ground truth", "inferno", dict(vmin=lo, vmax=hi)),
            (axes[1], pred, f"data-driven PINN (N={ds.n_sensors})", "inferno", dict(vmin=lo, vmax=hi)),
            (axes[2], np.abs(pred - true), f"|error|, rel L2 = {rel:.3f}", "viridis", {}),
        ):
            im = ax.imshow(data, origin="lower", extent=(0, 1, 0, 1), cmap=cmap, **kw)
            ax.set_title(title, fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046)
        sx, sy = ds.positions_m[:, 0] / 0.012, ds.positions_m[:, 1] / 0.012
        for ax in axes[:2]:
            ax.plot(sx, sy, "c^", ms=5, mec="k")
        fig.suptitle(f"{ds.ids[i]} ({ds.split}, T_amb={ds.T_amb[i]:.1f} degC)", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"{ds.ids[i]}_N{ds.n_sensors}.png", dpi=120)
        plt.close(fig)


def save_curves(history: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for col in ("temp", "src", "phys"):
        ax[0].semilogy(history["step"], history[col].clip(lower=1e-12), label=col)
    ax[0].set_xlabel("step"); ax[0].set_ylabel("train loss term"); ax[0].legend(); ax[0].grid(alpha=0.3)
    val = history.dropna(subset=["val_rel_l2"])
    ax[1].plot(val["step"], val["val_rel_l2"], marker="o")
    ax[1].set_xlabel("step"); ax[1].set_ylabel("validation relative L2"); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def run_training(args, model, physics, device, out: Path) -> float:
    """Train; return wall-clock seconds."""
    train = load_split_arrays("train", args.sensors)
    val = load_split_arrays("validation", args.sensors)
    if args.eval_chips:
        val = val.subset(args.eval_chips)
    data = TrainData(train, physics, args.temp_scale, device)
    print(f"train chips: {len(train)}  validation chips: {len(val)}  sensors: {args.sensors}  "
          f"params: {model.n_params:,}  device: {device}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    step, best = 0, float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if ck.get("opt"):
            opt.load_state_dict(ck["opt"])
        step, best = int(ck["step"]), float(ck["best_val"])
        gen.manual_seed(args.seed + step)            # do not replay the same batches after a restart
        print(f"resumed from {args.resume} at step {step} (best val rel L2 {best:.4f})")

    hist_path = out / "history.csv"
    history = pd.read_csv(hist_path) if (args.resume and hist_path.exists()) else pd.DataFrame(
        columns=["step", "total", "temp", "src", "phys", "lr", "val_rel_l2", "seconds"])
    rows = []
    t0 = time.perf_counter()
    start_step = step
    run_acc = {"total": 0.0, "temp": 0.0, "src": 0.0, "phys": 0.0, "n": 0}

    while step < args.steps:
        model.train()
        lr = lr_at(step, args.steps, args.lr, args.lr_min)
        for g in opt.param_groups:
            g["lr"] = lr
        warm = max(args.phys_warmup * args.steps, 1.0)
        w_phys = args.w_phys * min(1.0, step / warm)

        losses = training_losses(model, data, physics, args, w_phys, gen)
        opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        step += 1

        for k in ("total", "temp", "src", "phys"):
            run_acc[k] += float(losses[k].detach())
        run_acc["n"] += 1

        do_eval = step % args.eval_every == 0 or step == args.steps
        if step % args.log_every == 0 or do_eval:
            n = run_acc["n"]
            row = {"step": step, "lr": lr, "val_rel_l2": np.nan, "seconds": time.perf_counter() - t0,
                   **{k: run_acc[k] / n for k in ("total", "temp", "src", "phys")}}
            run_acc = {"total": 0.0, "temp": 0.0, "src": 0.0, "phys": 0.0, "n": 0}
            if do_eval:
                T_pred, Q_pred = predict(model, val, physics, device)
                df = score(val, T_pred, Q_pred)
                row["val_rel_l2"] = float(df["relative_l2"].mean())
                if row["val_rel_l2"] < best:
                    best = row["val_rel_l2"]
                    save_checkpoint(out / "best.pt", model, opt, step, best, args)
                save_checkpoint(out / "last.pt", model, opt, step, best, args)
            rows.append(row)
            eta = (time.perf_counter() - t0) / (step - start_step) * (args.steps - step)
            msg = (f"[{step:6d}/{args.steps}] loss={row['total']:.3e} temp={row['temp']:.3e} "
                   f"src={row['src']:.3e} phys={row['phys']:.3e} lr={lr:.1e}")
            if do_eval:
                msg += f" | val rel L2={row['val_rel_l2']:.4f} (best {best:.4f})"
            print(f"{msg} | {row['seconds']/60:.1f} min, ETA {eta/60:.1f} min", flush=True)
            if do_eval:
                _join(history, rows).to_csv(hist_path, index=False)

    history = _join(history, rows)
    history.to_csv(hist_path, index=False)
    try:
        save_curves(history, out / "curves.png")
    except Exception as exc:                          # plotting must never lose a finished run
        print(f"warning: could not save curves.png: {exc}")
    save_checkpoint(out / "last.pt", model, opt, step, best, args)
    return time.perf_counter() - t0


def run_evaluation(args, model, physics, device, out: Path, train_seconds: float | None) -> None:
    summary = {"config": config_name(args), "n_params": model.n_params, "seed": args.seed}
    if train_seconds is not None:
        summary["train_seconds"] = train_seconds
    seed_row = None
    for split in [s for s in args.eval_splits.split(",") if s]:
        ds = load_split_arrays(split, args.sensors)
        t0 = time.perf_counter()
        T_pred, Q_pred = predict(model, ds, physics, device)
        infer = time.perf_counter() - t0
        df = score(ds, T_pred, Q_pred)
        df.to_csv(out / f"per_chip_{split}.csv", index=False)
        mean = mean_row(df)
        mean["inference_seconds_per_chip"] = infer / len(ds)
        summary[split] = mean
        print(f"\n{split}: {len(ds)} chips, N={args.sensors} (non-sensor nodes)")
        for k in ("relative_l2", "mae_C", "rmse_C", "peak_temp_abs_error_C", "hotspot_localization_cells",
                  "source_relative_l2"):
            print(f"  {k:28s} {mean[k]:.4f}   (median {df[k].median():.4f})")
        if args.save_figs:
            save_triptychs(ds, T_pred, out / "figures" / split, args.save_figs)
        if split == "test" or seed_row is None:
            seed_row = {"config": config_name(args), "split": split, **mean,
                        "n_params": model.n_params, "train_seconds": train_seconds or float("nan")}
    if seed_row is not None:
        # one row per seed, in the layout src/eval/report.py aggregates
        pd.DataFrame([seed_row]).to_csv(out / f"seed_{args.seed}.csv", index=False)
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {out}/seed_{args.seed}.csv, per_chip_*.csv, summary.json")


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    physics = Physics()
    out = Path(args.out) if args.out else default_out(args)
    out.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        ckpt = Path(args.ckpt) if args.ckpt else out / "best.pt"
        model, ck = load_model(ckpt, device)
        if model.n_sensors != args.sensors:
            raise SystemExit(f"{ckpt} was trained with {model.n_sensors} sensors, not {args.sensors}")
        print(f"loaded {ckpt} (step {ck['step']}, best val rel L2 {ck['best_val']:.4f})")
        run_evaluation(args, model, physics, device, out, None)
        return

    model = ConditionalPINN(
        n_sensors=args.sensors, latent=args.latent, branch_width=args.branch_width,
        branch_depth=args.branch_depth, trunk_width=args.trunk_width, trunk_depth=args.trunk_depth,
        activation=args.activation, temp_scale=args.temp_scale, hard_neumann=not args.soft_neumann,
    ).to(device)
    with open(out / "args.json", "w") as fh:
        json.dump(vars(args), fh, indent=2)

    seconds = run_training(args, model, physics, device, out)
    best_ckpt = out / "best.pt"
    if best_ckpt.exists():                           # score the best-by-validation weights, not the last
        model, ck = load_model(best_ckpt, device)
        print(f"\nscoring best checkpoint (step {ck['step']}, val rel L2 {ck['best_val']:.4f})")
    run_evaluation(args, model, physics, device, out, seconds)


if __name__ == "__main__":
    main()
