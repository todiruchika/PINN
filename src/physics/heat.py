"""Steady 2D heat conduction stubs.

The governing equation and boundary conditions are documented in PROTOCOL.md.
Residual and boundary-condition implementations remain intentionally pending.
"""


def pde_residual(T, Q, k, dx):
    """Compute PDE residual on interior nodes.

    The confirmed model is -div(t_die*k(T)*grad(T)) + h_eff*(T-T_amb) = q(x,y,T).

    Raises:
        NotImplementedError: residual discretisation is not implemented.
    """
    raise NotImplementedError("PDE residual implementation is pending")


def apply_boundary_condition(T, T_amb):
    """Apply boundary condition to temperature field.

    Raises:
        NotImplementedError: boundary-condition implementation is pending.
    """
    raise NotImplementedError("Boundary-condition implementation is pending")


def hard_bc_ansatz(coords, T_amb):
    """Hard-constraint BC ansatz for PINN.

    Raises:
        NotImplementedError: hard-constraint ansatz is not implemented.
    """
    raise NotImplementedError("Hard boundary-condition ansatz is pending")
