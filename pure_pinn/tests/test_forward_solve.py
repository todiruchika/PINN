"""Oracle forward solve: can the field network represent a dataset solution?

Model B has two error sources that field metrics alone cannot separate:

1. the *inverse* part -- N sparse sensors do not pin down the source field, and
2. the *representation/optimisation* part -- whether a tanh MLP with the hard
   Neumann embedding, trained on a strong-form residual, can reproduce a 128x128
   FEM field at all.

This diagnostic removes (1) by handing the network the ground-truth source
(the dataset's stored ``Q`` grid, already including its converged leakage) and
minimising the residual alone -- no sensor data. The resulting relative L2 error
is the floor Model B could reach with a perfectly recovered source, so it is the
right reference for interpreting Model B's numbers.

Uses ground-truth data by design; it is a diagnostic, not a model.

    python tests/test_forward_solve.py --chip thermal_00001 --split train --steps 3000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import ChipScaler, load_chip  # noqa: E402
from evaluate import field_metrics  # noqa: E402
from losses import ChipContext, Physics, residual_loss, sample_interior  # noqa: E402
from models.pinn import FixedGridSource, PhysicsOnlyPINN, TemperatureField  # noqa: E402
from train_pinn import predict_field  # noqa: E402
from utils import set_seed, timer  # noqa: E402


def forward_solve(chip, adam_steps: int, lbfgs_steps: int, width: int, depth: int,
                  n_collocation: int, seed: int, verbose: bool = True) -> dict:
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    device = torch.device("cpu")
    physics = Physics()

    # scale from the true peak here: this is a diagnostic, not a model run
    scaler = ChipScaler(offset=chip.T_amb, scale=float(chip.peak_temperature - chip.T_amb))
    ctx = ChipContext(T_amb=scaler.offset, scale=scaler.scale)

    field = TemperatureField(bc_mode="hard", width=width, depth=depth)
    source = FixedGridSource(chip.source_field(), q_scale=physics.h_eff * scaler.scale)
    model = PhysicsOnlyPINN(field=field, source=source, physics=physics).to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    def loss_at(xy):
        return residual_loss(model.temperature_fn, model.source_fn, xy, physics, ctx)

    with timer() as t:
        opt = torch.optim.Adam(params, lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(adam_steps, 1), eta_min=1e-5)
        for step in range(adam_steps):
            loss = loss_at(sample_interior(n_collocation, gen))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if verbose and step % max(adam_steps // 6, 1) == 0:
                print(f"  [adam {step:5d}] residual={float(loss.detach()):.4e}")

        if lbfgs_steps > 0:
            xy = sample_interior(4 * n_collocation, gen)
            opt2 = torch.optim.LBFGS(params, lr=1.0, max_iter=lbfgs_steps, history_size=50,
                                     line_search_fn="strong_wolfe")

            def closure():
                opt2.zero_grad(set_to_none=True)
                loss = loss_at(xy)
                loss.backward()
                return loss

            final = opt2.step(closure)
            if verbose:
                print(f"  [lbfgs end] residual={float(final.detach()):.4e}")

    pred = predict_field(model, scaler, 128, device)
    # no sensors are involved, so every node counts
    metrics = field_metrics(pred, chip.field(), np.zeros((128, 128), dtype=bool), chip.die_L)
    metrics["train_seconds"] = t["seconds"]
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Oracle-source forward solve (representation floor)")
    ap.add_argument("--chip", default="thermal_00001")
    ap.add_argument("--split", default="train")
    ap.add_argument("--steps", type=int, default=3000, help="Adam steps")
    ap.add_argument("--lbfgs", type=int, default=1000)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--collocation", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=0.15, help="pass threshold on relative L2")
    args = ap.parse_args()

    chip = load_chip(args.chip, args.split)
    print(f"{chip.sample_id}: T_amb={chip.T_amb:.2f} degC, {chip.block_count} blocks, "
          f"peak={chip.peak_temperature:.1f} degC")
    m = forward_solve(chip, args.steps, args.lbfgs, args.width, args.depth, args.collocation, args.seed)
    print(f"\nrelative L2      {m['rel_l2']:.4f}   <- representation/optimisation floor for Model B")
    print(f"max abs error    {m['max_abs_err']:.2f} degC")
    print(f"peak temp error  {m['peak_temp_err']:.2f} degC")
    print(f"hotspot loc err  {m['hotspot_loc_cells']:.1f} cells")
    print(f"boundary flux    {m['boundary_flux_mean']:.1f} K/m (mean |dT/dn| on the edges)")
    print(f"wall clock       {m['train_seconds']:.1f} s")

    ok = m["rel_l2"] < args.tol
    print(f"\n{'PASS' if ok else 'FAIL'}: rel L2 {'<' if ok else '>='} {args.tol}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
