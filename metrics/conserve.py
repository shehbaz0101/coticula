"""Conservation / residual-style audits for PDE fields."""
from __future__ import annotations

import numpy as np


def discrete_energy_1d(u: np.ndarray, dx: float) -> np.ndarray:
    """Kinetic-style energy 0.5 * sum(u^2) * dx over last axis; u: (..., nx)."""
    u = np.asarray(u, dtype=np.float64)
    return 0.5 * np.sum(u**2, axis=-1) * dx


def discrete_energy_2d(u: np.ndarray, dx: float) -> np.ndarray:
    """0.5 * sum(u^2) * dx^2 over last two axes; u: (..., ny, nx)."""
    u = np.asarray(u, dtype=np.float64)
    return 0.5 * np.sum(u**2, axis=(-2, -1)) * (dx**2)


def energy_drift(energies: np.ndarray) -> dict:
    """Relative drift from t=0: (E_t - E_0) / (|E_0| + eps)."""
    e = np.asarray(energies, dtype=np.float64)
    e0 = e[..., 0:1]
    rel = (e - e0) / (np.abs(e0) + 1e-12)
    return {
        "rel_drift": rel,
        "max_abs_rel_drift": float(np.max(np.abs(rel))),
        "final_mean_rel_drift": float(np.mean(rel[..., -1])),
    }


def burgers_residual_l2(
    u: np.ndarray,
    dx: float,
    dt: float,
    nu: float | np.ndarray,
) -> float:
    """Simple FD residual of Burgers on interior time steps (periodic).

    u: (nt+1, nx) or (B, nt+1, nx). Returns mean |residual| L2 over batch/time.
    """
    u = np.asarray(u, dtype=np.float64)
    single = u.ndim == 2
    if single:
        u = u[None, ...]
    B, ntp1, nx = u.shape
    nt = ntp1 - 1
    if np.isscalar(nu):
        nu_arr = np.full(B, float(nu))
    else:
        nu_arr = np.asarray(nu, dtype=np.float64)

    # centered in time between n and n+1 using mid state
    residuals = []
    for b in range(B):
        ub = u[b]
        nub = nu_arr[b]
        res_t = []
        for n in range(nt):
            u_n, u_np = ub[n], ub[n + 1]
            u_mid = 0.5 * (u_n + u_np)
            ut = (u_np - u_n) / dt
            up = np.roll(u_mid, -1)
            um = np.roll(u_mid, 1)
            adv = (up**2 - um**2) / (4.0 * dx)
            visc = nub * (up - 2.0 * u_mid + um) / (dx**2)
            r = ut + adv - visc
            res_t.append(np.linalg.norm(r) / (np.linalg.norm(u_mid) + 1e-12))
        residuals.append(float(np.mean(res_t)))
    return float(np.mean(residuals))


def heat_residual_l2(
    u: np.ndarray,
    dx: float,
    dt: float,
    alpha: float | np.ndarray,
) -> float:
    """FD residual of heat equation (Dirichlet grid). u: (nt+1,n,n) or batched."""
    u = np.asarray(u, dtype=np.float64)
    single = u.ndim == 3
    if single:
        u = u[None, ...]
    B, ntp1, n, _ = u.shape
    nt = ntp1 - 1
    if np.isscalar(alpha):
        a_arr = np.full(B, float(alpha))
    else:
        a_arr = np.asarray(alpha, dtype=np.float64)

    residuals = []
    for b in range(B):
        ub = u[b]
        ab = a_arr[b]
        res_t = []
        for step in range(nt):
            u_n, u_np = ub[step], ub[step + 1]
            u_mid = 0.5 * (u_n + u_np)
            ut = (u_np - u_n) / dt
            lap = (
                np.roll(u_mid, 1, 0)
                + np.roll(u_mid, -1, 0)
                + np.roll(u_mid, 1, 1)
                + np.roll(u_mid, -1, 1)
                - 4.0 * u_mid
            ) / (dx**2)
            # mask boundaries
            r = ut - ab * lap
            r[0, :] = r[-1, :] = r[:, 0] = r[:, -1] = 0.0
            interior = u_mid[1:-1, 1:-1]
            res_t.append(
                np.linalg.norm(r[1:-1, 1:-1])
                / (np.linalg.norm(interior) + 1e-12)
            )
        residuals.append(float(np.mean(res_t)))
    return float(np.mean(residuals))


def audit_burgers(traj: np.ndarray, dx: float, dt: float, nu: np.ndarray) -> dict:
    e = discrete_energy_1d(traj, dx)  # (B, nt+1)
    drift = energy_drift(e)
    return {
        "residual_rel_l2": burgers_residual_l2(traj, dx, dt, nu),
        "energy_max_abs_rel_drift": drift["max_abs_rel_drift"],
        "energy_final_mean_rel_drift": drift["final_mean_rel_drift"],
    }


def audit_heat(traj: np.ndarray, dx: float, dt: float, alpha: np.ndarray) -> dict:
    e = discrete_energy_2d(traj, dx)
    drift = energy_drift(e)
    return {
        "residual_rel_l2": heat_residual_l2(traj, dx, dt, alpha),
        "energy_max_abs_rel_drift": drift["max_abs_rel_drift"],
        "energy_final_mean_rel_drift": drift["final_mean_rel_drift"],
    }
