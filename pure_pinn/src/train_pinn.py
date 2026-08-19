"""Train Model B: one physics-only PINN per (chip, sensor count, seed).

Model B never touches the train/val split and never sees a ground-truth field.
Its only inputs are that chip's N sparse sensor readings, its metadata
constants (k, T_boundary), and the governing equation. "Testing" therefore
means re-solving from scratch on every held-out chip and reporting the
distribution of errors, not a generalisation check.

Usage
-----
    python src/train_pinn.py --config experiments/configs/model_b_default.json \
        --split test --chips thermal_00501 --sensors 5 --seeds 0

    # full arm of the experiment matrix, subsampling the test set for cost
    python src/train_pinn.py --config experiments/configs/model_b_default.json \
        --split test --n-chips 20 --sensors 5,8,10 --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch  # type: ignore[import-not-found]

sys.path.insert(0, str(Path(__file__).resolve().parent))  # allow flat imports

from data_loader import (  # noqa: E402
    ChipSample, ChipScaler, chip_ids, field_from_points, grid_points, load_chip,
)
from evaluate import append_metrics_row, field_metrics, sensor_rmse, source_metrics  # noqa: E402
from losses import (  # noqa: E402
    annealed_weight, boundary_loss, data_loss, residual_loss, sample_boundary, sample_interior,
)
from models.pinn import build_pinn  # noqa: E402
from utils import deep_update, load_config, resolve_path, set_seed, timer  # noqa: E402

DTYPE = torch.float32


# --------------------------------------------------------------------------- #
# one solve
# --------------------------------------------------------------------------- #
def train_one_chip(chip: ChipSample, n_sensors: int, seed: int, config: dict, verbose: bool = True) -> dict:
    """Fit one PINN to one chip and return its metrics row plus artefacts."""
    set_seed(seed)
    device = torch.device(config.get("device", "cpu"))
    gen = torch.Generator(device="cpu").manual_seed(seed + 10_000)

    tr = config["training"]
    bc_mode = config["model"].get("bc_mode", "hard")

    # ---- inputs the model is allowed to see ------------------------------- #
    positions, temps = chip.sensors(n_sensors)
    if bc_mode == "hard":
        # boundary value enters by construction, so normalise around it
        scaler = ChipScaler.from_sensors(chip, temps)
    else:
        # soft/free arms are not handed T_boundary by construction; keep the
        # offset at 0 so the network has to represent the boundary level itself
        scaler = ChipScaler(offset=0.0, scale=max(float(np.max(temps)), 1.0))

    xy_data = torch.tensor(positions, dtype=DTYPE, device=device)
    t_data = torch.tensor(scaler.normalise(temps), dtype=DTYPE, device=device).reshape(-1, 1)
    t_boundary_norm = float(scaler.normalise(chip.T_boundary))

    # ---- model ------------------------------------------------------------ #
    model_cfg = json.loads(json.dumps(config["model"]))  # deep copy
    src_cfg = model_cfg.setdefault("source", {})
    count_key = {"gaussian": "n_blobs", "softrect": "n_blocks"}.get(src_cfg.get("kind", "gaussian"))
    if count_key:  # the mlp source has no blob count
        requested = src_cfg.pop("n_blobs", None)
        requested = src_cfg.pop("n_blocks", requested)
        if requested in (None, "oracle"):
            # oracle-count variant: use the chip's true hotspot count. Flagged as
            # an oracle because a deployed solver would not know M.
            requested = chip.hotspot_count
        src_cfg[count_key] = int(requested)
    model = build_pinn(model_cfg).to(device)

    params = list(model.parameters())
    w_r = float(tr.get("w_residual", 1.0))
    w_d = float(tr.get("w_data", 1.0))
    w_b = float(tr.get("w_boundary", 1.0))
    res_scale = float(tr.get("residual_scale", 1.0))
    scheme = tr.get("weight_scheme", "fixed")

    def losses_at(xy_col: torch.Tensor, xy_bnd: torch.Tensor | None):
        l_res = residual_loss(model.temperature_fn, model.source_fn, xy_col, residual_scale=res_scale)
        l_dat = data_loss(model.temperature_fn, xy_data, t_data)
        l_bnd = torch.zeros((), device=device)
        if bc_mode == "soft" and xy_bnd is not None:
            l_bnd = boundary_loss(model.temperature_fn, xy_bnd, t_boundary_norm)
        return l_res, l_dat, l_bnd

    def total(l_res, l_dat, l_bnd, wd, wb):
        return w_r * l_res + wd * l_dat + wb * l_bnd

    history: list[dict] = []
    best = {"loss": float("inf"), "state": None}

    def record(step: int, phase: str, l_res, l_dat, l_bnd, loss, wd):
        entry = {"step": step, "phase": phase, "loss": float(loss.detach()),
                 "residual": float(l_res.detach()), "data": float(l_dat.detach()),
                 "boundary": float(l_bnd.detach()), "w_data": float(wd)}
        history.append(entry)
        if entry["loss"] < best["loss"] and np.isfinite(entry["loss"]):
            best["loss"] = entry["loss"]
            best["state"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        return entry

    with timer() as train_time:
        # ------------------------- Adam ----------------------------------- #
        n_col = int(tr.get("n_collocation", 2048))
        n_bnd = int(tr.get("n_boundary", 256))
        adam_steps = int(tr.get("adam_steps", 5000))
        opt = torch.optim.Adam(params, lr=float(tr.get("lr", 1e-3)))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(adam_steps, 1), eta_min=float(tr.get("lr_min", 1e-5))
        )
        anneal_every = int(tr.get("anneal_every", 100))
        ema = float(tr.get("anneal_ema", 0.9))
        wd_cur = w_d

        for step in range(adam_steps):
            xy_col = sample_interior(n_col, gen, device, DTYPE)
            xy_bnd = sample_boundary(n_bnd, gen, device, DTYPE) if bc_mode == "soft" else None
            l_res, l_dat, l_bnd = losses_at(xy_col, xy_bnd)

            if scheme == "annealing" and step % anneal_every == 0:
                target = annealed_weight(w_r * l_res, l_dat, params)
                wd_cur = ema * wd_cur + (1 - ema) * float(np.clip(target, 1e-3, 1e6))

            loss = total(l_res, l_dat, l_bnd, wd_cur, w_b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if tr.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(params, float(tr["grad_clip"]))
            opt.step()
            sched.step()

            if step % int(tr.get("log_every", 500)) == 0 or step == adam_steps - 1:
                entry = record(step, "adam", l_res, l_dat, l_bnd, loss, wd_cur)
                if verbose:
                    print(f"  [adam {step:5d}] loss={entry['loss']:.4e} "
                          f"res={entry['residual']:.4e} data={entry['data']:.4e} w_d={wd_cur:.3g}")

        # ------------------------- L-BFGS --------------------------------- #
        lbfgs_steps = int(tr.get("lbfgs_steps", 0))
        if lbfgs_steps > 0:
            # L-BFGS needs a deterministic objective, so the collocation set is
            # fixed for this phase (resampling breaks its line search).
            xy_col = sample_interior(int(tr.get("n_collocation_lbfgs", 4096)), gen, device, DTYPE)
            xy_bnd = sample_boundary(n_bnd, gen, device, DTYPE) if bc_mode == "soft" else None
            opt2 = torch.optim.LBFGS(
                params, lr=float(tr.get("lbfgs_lr", 1.0)), max_iter=lbfgs_steps,
                history_size=int(tr.get("lbfgs_history", 50)),
                tolerance_grad=float(tr.get("lbfgs_tol_grad", 1e-11)),
                tolerance_change=float(tr.get("lbfgs_tol_change", 1e-12)),
                line_search_fn="strong_wolfe",
            )
            counter = {"n": 0}

            def closure():
                opt2.zero_grad(set_to_none=True)
                l_res, l_dat, l_bnd = losses_at(xy_col, xy_bnd)
                loss = total(l_res, l_dat, l_bnd, wd_cur, w_b)
                loss.backward()
                counter["n"] += 1
                if counter["n"] % int(tr.get("log_every_lbfgs", 100)) == 1:
                    entry = record(adam_steps + counter["n"], "lbfgs", l_res, l_dat, l_bnd, loss, wd_cur)
                    if verbose:
                        print(f"  [lbfgs {counter['n']:5d}] loss={entry['loss']:.4e} "
                              f"res={entry['residual']:.4e} data={entry['data']:.4e}")
                return loss

            opt2.step(closure)
            l_res, l_dat, l_bnd = losses_at(xy_col, xy_bnd)
            record(adam_steps + counter["n"] + 1, "lbfgs",
                   l_res, l_dat, l_bnd, total(l_res, l_dat, l_bnd, wd_cur, w_b), wd_cur)

    # restore the best iterate: L-BFGS can and does step uphill on this problem
    if tr.get("restore_best", True) and best["state"] is not None:
        model.load_state_dict(best["state"])

    # ---- evaluation on the full field (ground truth used here only) ------- #
    grid_n = int(config.get("eval", {}).get("grid", 128))
    with timer() as infer_time:
        pred_field = predict_field(model, scaler, grid_n, device)
    true_field = chip.field()
    if true_field.shape[0] != grid_n:
        raise ValueError(f"eval grid {grid_n} does not match matrix size {true_field.shape[0]}")

    row = {"model": "B", "bc_mode": bc_mode,
           "source_kind": config["model"]["source"].get("kind", "gaussian"),
           "split": chip.split, "sample_id": chip.sample_id,
           "n_sensors": n_sensors, "seed": seed}
    row.update(field_metrics(pred_field, true_field, chip.T_boundary))
    row["sensor_rmse"] = sensor_rmse(pred_field, positions, temps)
    row.update(evaluate_source(model, chip, scaler, grid_n, device))
    last = history[-1]
    row.update({"train_seconds": train_time["seconds"], "inference_seconds": infer_time["seconds"],
                "final_loss": last["loss"], "final_residual": last["residual"],
                "final_data_loss": last["data"]})

    return {"row": row, "history": history, "pred_field": pred_field,
            "scaler": {"offset": scaler.offset, "scale": scaler.scale}, "model": model}


# --------------------------------------------------------------------------- #
# prediction / source recovery
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_field(model, scaler: ChipScaler, n: int, device, chunk: int = 8192) -> np.ndarray:
    """Dense ``(n, n)`` temperature prediction in physical units."""
    pts = torch.tensor(grid_points(n), dtype=DTYPE, device=device)
    out = [model.temperature_fn(pts[i : i + chunk]).cpu().numpy().ravel() for i in range(0, len(pts), chunk)]
    return field_from_points(scaler.denormalise(np.concatenate(out)), n)


@torch.no_grad()
def predict_source(model, scaler: ChipScaler, k: float, n: int, device, chunk: int = 8192) -> np.ndarray:
    """Inferred ``Q(x, y)`` in physical W/m^2 on the ``(n, n)`` grid."""
    pts = torch.tensor(grid_points(n), dtype=DTYPE, device=device)
    out = [model.source_fn(pts[i : i + chunk]).cpu().numpy().ravel() for i in range(0, len(pts), chunk)]
    f_norm = np.concatenate(out)
    return field_from_points(f_norm / scaler.source_scale(k), n)  # Q = f_norm * k * scale


def evaluate_source(model, chip: ChipSample, scaler: ChipScaler, n: int, device) -> dict:
    """Validate the inferred source against metadata hotspots (Sec. 4 of the design)."""
    pred_Q = predict_source(model, scaler, chip.k, n, device)
    true_Q = chip.source_field(n)
    src = model.source
    centres = weights = None
    if hasattr(src, "centres"):
        centres = src.centres().detach().cpu().numpy()
        amps = src.amplitudes().detach().cpu().numpy()
        if hasattr(src, "widths"):
            weights = amps * src.widths().detach().cpu().numpy() ** 2      # integrated power
        else:
            areas = np.prod(4 * src.half_extents().detach().cpu().numpy(), axis=1)
            weights = amps * areas
    return source_metrics(pred_Q, true_Q, centres, weights,
                          chip.hotspot_centres(), chip.powers)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def parse_int_list(text: str) -> list[int]:
    return [int(v) for v in str(text).replace(" ", "").split(",") if v]


def select_chips(split: str, chips: str | None, n_chips: int | None, root, seed: int = 0) -> list[str]:
    """Explicit ids, or a deterministic subsample of the split (see Sec. 10 cost note)."""
    if chips:
        return [c for c in chips.replace(" ", "").split(",") if c]
    ids = chip_ids(root, split)
    if n_chips and n_chips < len(ids):
        rng = np.random.default_rng(seed)
        return [ids[i] for i in sorted(rng.choice(len(ids), size=n_chips, replace=False))]
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Model B (physics-only PINN), one solve per chip")
    ap.add_argument("--config", default="experiments/configs/model_b_default.json")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--chips", default=None, help="comma-separated sample ids; overrides --n-chips")
    ap.add_argument("--n-chips", type=int, default=None, help="deterministic subsample of the split")
    ap.add_argument("--sensors", default="5,8,10")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--metrics", default="results/metrics.csv")
    ap.add_argument("--runs-dir", default="results/runs")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--tag", default=None, help="suffix for run artefact filenames")
    ap.add_argument("--save-fields", action="store_true", help="store predicted fields as .npy")
    ap.add_argument("--save-figs", action="store_true", help="store true/pred/error triptychs")
    ap.add_argument("--override", default=None, help="JSON dict merged into the config")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    config = load_config(resolve_path(args.config))
    if args.override:
        config = deep_update(config, json.loads(args.override))

    ids = select_chips(args.split, args.chips, args.n_chips, args.data_root)
    sensors = parse_int_list(args.sensors)
    seeds = parse_int_list(args.seeds)
    runs_dir = resolve_path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""

    total_runs = len(ids) * len(sensors) * len(seeds)
    print(f"Model B: {total_runs} solves "
          f"({len(ids)} chips x {len(sensors)} sensor counts x {len(seeds)} seeds), split={args.split}")

    done = 0
    for sample_id in ids:
        chip = load_chip(sample_id, args.split, args.data_root)
        for n_sensors in sensors:
            for seed in seeds:
                done += 1
                print(f"[{done}/{total_runs}] {sample_id} N={n_sensors} seed={seed} "
                      f"(k={chip.k:.1f}, T_b={chip.T_boundary:.2f}, hotspots={chip.hotspot_count})")
                result = train_one_chip(chip, n_sensors, seed, config, verbose=not args.quiet)
                row = result["row"]
                append_metrics_row(row, args.metrics)
                print(f"      rel_L2={row['rel_l2']:.4f} max_abs={row['max_abs_err']:.1f} degC "
                      f"peak_loc_err={row['peak_loc_err']:.3f} bnd_mae={row['boundary_mae']:.3f} "
                      f"({row['train_seconds']:.1f} s)")

                stem = f"{sample_id}_N{n_sensors}_seed{seed}_{row['bc_mode']}_{row['source_kind']}{tag}"
                with open(runs_dir / f"{stem}.json", "w", encoding="utf-8") as fh:
                    json.dump({"row": row, "config": config, "history": result["history"],
                               "scaler": result["scaler"]}, fh, indent=2)
                if args.save_fields:
                    np.save(runs_dir / f"{stem}_pred.npy", result["pred_field"])
                if args.save_figs:
                    save_triptych(result["pred_field"], chip, row, stem)

    print(f"\nwrote {total_runs} rows to {resolve_path(args.metrics)}")


def save_triptych(pred: np.ndarray, chip: ChipSample, row: dict, stem: str) -> None:
    """True / predicted / absolute-error panels for the writeup."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    true = chip.field()
    err = np.abs(pred - true)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, data, title, cmap in (
        (axes[0], true, "ground truth", "inferno"),
        (axes[1], pred, f"Model B ({row['bc_mode']}, N={row['n_sensors']})", "inferno"),
        (axes[2], err, f"|error|, rel L2 = {row['rel_l2']:.3f}", "viridis"),
    ):
        vmax = true.max() if cmap == "inferno" else None
        im = ax.imshow(data, origin="lower", extent=(0, 1, 0, 1), cmap=cmap,
                       vmin=true.min() if cmap == "inferno" else None, vmax=vmax)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{chip.sample_id} (k={chip.k:.1f}, T_b={chip.T_boundary:.1f} degC)", fontsize=11)
    fig.tight_layout()
    out = resolve_path("results/figures") / f"{stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
