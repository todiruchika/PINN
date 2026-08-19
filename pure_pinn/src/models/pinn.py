"""Model B -- physics-only PINN with hard boundary-condition enforcement.

Survey placement: soft/strong cell for the governing equation (there is no
other option -- see Sec. IV-A: hard enforcement applies to BC/IC and
essentially never to the PDE) and the *hard* column of Axis A for the
boundary condition, via the Lagaris-style construction.

    T_norm(x, y) = 0 + d(x, y) * NN(x, y),   d(x, y) = 16 x(1-x) y(1-y)

In normalised units the Dirichlet value is 0, so this satisfies
``T = T_boundary`` on the whole of ``dOmega`` by construction, for every chip,
with no boundary weight to tune. The factor 16 makes ``max d = 1`` so the
network output is not forced to be large just to reach realistic temperatures.

The source field ``f_norm`` is unknown and therefore trainable. Three
parametrisations are provided:

``gaussian``   isotropic Gaussian blobs (the study's default choice)
``softrect``   smoothed axis-aligned rectangles -- matches how the dataset was
               actually generated, so it is the matched-model arm of the
               source-parametrisation ablation
``mlp``        a free-form second network, no shape assumption
"""
from __future__ import annotations

import inspect
import math

import torch
from torch import nn


# --------------------------------------------------------------------------- #
# backbone
# --------------------------------------------------------------------------- #
ACTIVATIONS = {"tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU, "sin": None}


class Sine(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


def _activation(name: str) -> nn.Module:
    if name == "sin":
        return Sine()
    if name not in ACTIVATIONS:
        raise ValueError(f"unknown activation {name!r}; choose from {sorted(ACTIVATIONS)}")
    return ACTIVATIONS[name]()


class FourierFeatures(nn.Module):
    """Fixed random Fourier embedding ``[sin(2 pi B x), cos(2 pi B x)]``.

    Off by default: the plain tanh MLP is the vanilla baseline, and the
    embedding is a spectral-bias remedy worth ablating separately rather than
    smuggling into the default configuration.
    """

    def __init__(self, in_dim: int = 2, n_features: int = 32, sigma: float = 3.0):
        super().__init__()
        self.register_buffer("B", torch.randn(in_dim, n_features) * sigma)

    @property
    def out_dim(self) -> int:
        return 2 * self.B.shape[1]

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * (xy @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 1,
        width: int = 64,
        depth: int = 4,
        activation: str = "tanh",
        fourier: dict | None = None,
    ):
        super().__init__()
        self.embed = None
        if fourier:
            self.embed = FourierFeatures(
                in_dim=in_dim,
                n_features=int(fourier.get("n_features", 32)),
                sigma=float(fourier.get("sigma", 3.0)),
            )
            in_dim = self.embed.out_dim

        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(depth):
            layers += [nn.Linear(prev, width), _activation(activation)]
            prev = width
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        if self.embed is not None:
            xy = self.embed(xy)
        return self.net(xy)


# --------------------------------------------------------------------------- #
# temperature field
# --------------------------------------------------------------------------- #
def distance_function(xy: torch.Tensor) -> torch.Tensor:
    """``d(x, y) = 16 x(1-x) y(1-y)``: vanishes on ``dOmega``, peaks at 1 in the centre."""
    x, y = xy[:, 0:1], xy[:, 1:2]
    return 16.0 * x * (1.0 - x) * y * (1.0 - y)


class TemperatureField(nn.Module):
    """``T_norm(x, y)`` with hard, soft, or free boundary treatment.

    ``bc_mode``:

    ``hard``  ``T_norm = d(x, y) * NN(x, y)`` -- exact Dirichlet by construction
              (normalised units put the boundary value at 0).
    ``soft``  ``T_norm = NN(x, y)``, boundary imposed by a penalty term.
    ``free``  identical network to ``soft`` but trained with *no* boundary
              term, to test whether residual + interior sensors alone recover
              the boundary temperature.
    """

    def __init__(self, bc_mode: str = "hard", out_scale: float = 1.0, **mlp_kwargs):
        super().__init__()
        if bc_mode not in ("hard", "soft", "free"):
            raise ValueError(f"bc_mode must be hard|soft|free, got {bc_mode!r}")
        self.bc_mode = bc_mode
        self.out_scale = float(out_scale)
        self.net = MLP(in_dim=2, out_dim=1, **mlp_kwargs)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        raw = self.out_scale * self.net(xy)
        if self.bc_mode == "hard":
            return distance_function(xy) * raw
        return raw


# --------------------------------------------------------------------------- #
# source parametrisations -- all return f_norm = Q / (k * scale) >= 0
# --------------------------------------------------------------------------- #
class GaussianSource(nn.Module):
    """Sum of ``n_blobs`` isotropic Gaussians with trainable amplitude, centre, width."""

    def __init__(self, n_blobs: int = 6, init_amp: float = 10.0, init_width: float = 0.12,
                 min_width: float = 0.02, max_width: float = 0.5):
        super().__init__()
        self.n_blobs = int(n_blobs)
        self.min_width, self.max_width = float(min_width), float(max_width)
        # amplitudes are exponential so they can move across orders of magnitude
        self.log_amp = nn.Parameter(torch.full((self.n_blobs,), math.log(init_amp)))
        # centres live in (0, 1)^2 through a sigmoid; initialised on a jittered grid
        self.centre_raw = nn.Parameter(_init_centres(self.n_blobs))
        w0 = _inverse_scaled_sigmoid(init_width, min_width, max_width)
        self.width_raw = nn.Parameter(torch.full((self.n_blobs,), w0))

    # -- interpretable parameters ------------------------------------------- #
    def amplitudes(self) -> torch.Tensor:
        return torch.exp(self.log_amp)

    def centres(self) -> torch.Tensor:
        return torch.sigmoid(self.centre_raw)

    def widths(self) -> torch.Tensor:
        return self.min_width + (self.max_width - self.min_width) * torch.sigmoid(self.width_raw)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        d2 = torch.sum((xy[:, None, :] - self.centres()[None, :, :]) ** 2, dim=-1)  # (N, B)
        sig2 = (self.widths() ** 2)[None, :]
        return torch.sum(self.amplitudes()[None, :] * torch.exp(-d2 / (2.0 * sig2)), dim=1, keepdim=True)


class SoftRectSource(nn.Module):
    """Sum of smoothed axis-aligned rectangles -- matched to the data generator.

    Each block is ``amp * sigmoid((x-x0)/eps) sigmoid((x1-x)/eps) * (same in y)``,
    a differentiable indicator with trainable centre, half-extent, and amplitude.
    """

    def __init__(self, n_blocks: int = 6, init_amp: float = 10.0, init_half: float = 0.12,
                 sharpness: float = 60.0, min_half: float = 0.01, max_half: float = 0.5):
        super().__init__()
        self.n_blocks = int(n_blocks)
        self.sharpness = float(sharpness)
        self.min_half, self.max_half = float(min_half), float(max_half)
        self.log_amp = nn.Parameter(torch.full((self.n_blocks,), math.log(init_amp)))
        self.centre_raw = nn.Parameter(_init_centres(self.n_blocks))
        h0 = _inverse_scaled_sigmoid(init_half, min_half, max_half)
        self.half_raw = nn.Parameter(torch.full((self.n_blocks, 2), h0))

    def amplitudes(self) -> torch.Tensor:
        return torch.exp(self.log_amp)

    def centres(self) -> torch.Tensor:
        return torch.sigmoid(self.centre_raw)

    def half_extents(self) -> torch.Tensor:
        return self.min_half + (self.max_half - self.min_half) * torch.sigmoid(self.half_raw)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        centre, half = self.centres(), self.half_extents()
        delta = xy[:, None, :] - centre[None, :, :]                 # (N, B, 2)
        inside = torch.sigmoid(self.sharpness * (half[None] - delta.abs()))
        window = inside[..., 0] * inside[..., 1]                    # (N, B)
        return torch.sum(self.amplitudes()[None, :] * window, dim=1, keepdim=True)


class MLPSource(nn.Module):
    """Free-form source network with a softplus output so ``f_norm >= 0``."""

    def __init__(self, width: int = 32, depth: int = 3, activation: str = "tanh",
                 out_scale: float = 10.0, fourier: dict | None = None):
        super().__init__()
        self.net = MLP(in_dim=2, out_dim=1, width=width, depth=depth,
                       activation=activation, fourier=fourier)
        self.out_scale = float(out_scale)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.out_scale * nn.functional.softplus(self.net(xy))


class FixedRectSource(nn.Module):
    """Non-trainable exact rectangles from metadata -- **diagnostic only**.

    Used by ``tests/test_forward_solve.py`` to check that the field network plus
    hard BC can represent a dataset solution when the source is *known*. It uses
    ground-truth hotspot data and must never appear in a Model B run.
    """

    def __init__(self, rects, amplitudes, sharpness: float = 400.0):
        super().__init__()
        self.register_buffer("rects", torch.as_tensor(rects, dtype=torch.float32).reshape(-1, 4))
        self.register_buffer("amps", torch.as_tensor(amplitudes, dtype=torch.float32).reshape(-1))
        self.sharpness = float(sharpness)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        x, y = xy[:, 0:1], xy[:, 1:2]
        r = self.rects
        wx = torch.sigmoid(self.sharpness * (x - r[:, 0])) * torch.sigmoid(self.sharpness * (r[:, 2] - x))
        wy = torch.sigmoid(self.sharpness * (y - r[:, 1])) * torch.sigmoid(self.sharpness * (r[:, 3] - y))
        return torch.sum(self.amps[None, :] * wx * wy, dim=1, keepdim=True)


SOURCES = {"gaussian": GaussianSource, "softrect": SoftRectSource, "mlp": MLPSource}


def _init_params(cls) -> set[str]:
    return set(inspect.signature(cls.__init__).parameters) - {"self"}


def build_source(kind: str, **kwargs) -> nn.Module:
    """Instantiate a source parametrisation, tolerating sibling-specific keys.

    Configs are merged rather than replaced (``--override`` on top of a default
    config), so switching ``kind`` leaves behind keys that belong to another
    parametrisation -- e.g. ``init_width`` from ``gaussian`` when asking for
    ``softrect``. Those are dropped; keys belonging to no parametrisation at all
    are still an error, so typos do not pass silently.
    """
    if kind not in SOURCES:
        raise ValueError(f"unknown source {kind!r}; choose from {sorted(SOURCES)}")
    cls = SOURCES[kind]
    accepted = _init_params(cls)
    known_elsewhere = set().union(*(_init_params(c) for c in SOURCES.values())) - accepted
    unknown = set(kwargs) - accepted - known_elsewhere
    if unknown:
        raise ValueError(f"unknown option(s) {sorted(unknown)} for source {kind!r}; "
                         f"accepted: {sorted(accepted)}")
    return cls(**{k: v for k, v in kwargs.items() if k in accepted})


# --------------------------------------------------------------------------- #
# the assembled model
# --------------------------------------------------------------------------- #
class PhysicsOnlyPINN(nn.Module):
    """Model B: ``(x, y) -> T_norm``, plus a trainable source field.

    Holding both in one module keeps a single parameter list for the
    optimiser and one checkpoint per chip.
    """

    def __init__(self, field: TemperatureField, source: nn.Module):
        super().__init__()
        self.field = field
        self.source = source

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.field(xy)

    # convenient aliases matching the callables the loss module expects
    def temperature_fn(self, xy: torch.Tensor) -> torch.Tensor:
        return self.field(xy)

    def source_fn(self, xy: torch.Tensor) -> torch.Tensor:
        return self.source(xy)


def build_pinn(config: dict) -> PhysicsOnlyPINN:
    """Assemble Model B from the ``model`` block of a config dict."""
    field_cfg = dict(config.get("field", {}))
    source_cfg = dict(config.get("source", {}))
    kind = source_cfg.pop("kind", "gaussian")
    field = TemperatureField(bc_mode=config.get("bc_mode", "hard"), **field_cfg)
    return PhysicsOnlyPINN(field=field, source=build_source(kind, **source_cfg))


# --------------------------------------------------------------------------- #
# init helpers
# --------------------------------------------------------------------------- #
def _init_centres(n: int) -> torch.Tensor:
    """Jittered near-uniform starting centres, in pre-sigmoid space."""
    side = max(1, int(math.ceil(math.sqrt(n))))
    axis = (torch.arange(side, dtype=torch.float32) + 0.5) / side
    gx, gy = torch.meshgrid(axis, axis, indexing="xy")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)[:n]
    pts = pts + 0.05 * torch.randn_like(pts)
    pts = pts.clamp(0.05, 0.95)
    return torch.log(pts / (1.0 - pts))  # logit


def _inverse_scaled_sigmoid(value: float, lo: float, hi: float) -> float:
    """Pre-sigmoid value ``r`` such that ``lo + (hi-lo) sigmoid(r) == value``."""
    frac = min(max((value - lo) / (hi - lo), 1e-4), 1 - 1e-4)
    return math.log(frac / (1 - frac))
