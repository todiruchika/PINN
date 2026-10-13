"""Steady 2D heat conduction stubs.

BC and PDE form are unknown (see PROTOCOL.md blockers).
Do NOT implement until the boundary condition and governing equation are resolved.
"""


def pde_residual(T, Q, k, dx):
    """Compute PDE residual on interior nodes.

    For linear conduction: -k*nabla^2(T) - Q = 0.
    For leakage model: -k*nabla^2(T) - Q_dynamic - beta*(T-T_amb)*Q_leakage0 = 0.

    Raises:
        NotImplementedError: BC and PDE form must be confirmed first.
    """
    raise NotImplementedError("PDE residual: BC and governing equation unknown (BLOCKER)")


def apply_boundary_condition(T, T_amb):
    """Apply boundary condition to temperature field.

    Raises:
        NotImplementedError: Boundary condition is unknown (BLOCKER).
        Edge temperatures differ by up to 92 degC from T_amb.
    """
    raise NotImplementedError("Boundary condition unknown (BLOCKER)")


def hard_bc_ansatz(coords, T_amb):
    """Hard-constraint BC ansatz for PINN.

    Raises:
        NotImplementedError: Requires known BC (BLOCKER).
    """
    raise NotImplementedError("Hard BC ansatz: BC unknown (BLOCKER)")
