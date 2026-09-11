"""Classical 1D viscous Burgers solver (periodic, FTCS / viscous flux).

du/dt + u du/dx = nu d^2u/dx^2
"""
from __future__ import annotations

import numpy as np


def solve_burgers(
    nu: float = 0.01,
    nx: int = 64,
    nt: int = 100,
    L: float = 2 * np.pi,
    T: float = 1.0,
    u0: np.ndarray | None = None,
    seed: int | None = None,
) -> dict:
    """Integrate 1D Burgers with periodic BC via explicit finite differences.

    Returns dict with keys: x, t, u (shape nt+1, nx), nu, meta.
    """
    dx = L / nx
    dt = T / nt
    x = np.linspace(0.0, L, nx, endpoint=False)

    if u0 is None:
        rng = np.random.default_rng(seed)
        # Smooth random Fourier modes as IC
        modes = rng.normal(size=4) + 1j * rng.normal(size=4)
        k = np.arange(1, 5)
        u = np.zeros(nx, dtype=np.float64)
        for m, kk in zip(modes, k):
            u += (m.real * np.cos(kk * x) + m.imag * np.sin(kk * x)) / kk
        u = u / (np.max(np.abs(u)) + 1e-12)
    else:
        u = np.asarray(u0, dtype=np.float64).copy()
        if u.shape != (nx,):
            raise ValueError(f"u0 shape {u.shape} != ({nx},)")

    # CFL-ish safety for viscosity + advection
    cfl_v = nu * dt / (dx**2)
    if cfl_v > 0.4:
        # substep
        n_sub = int(np.ceil(cfl_v / 0.35))
        dt_sub = dt / n_sub
    else:
        n_sub = 1
        dt_sub = dt

    traj = np.zeros((nt + 1, nx), dtype=np.float64)
    traj[0] = u

    for n in range(nt):
        for _ in range(n_sub):
            up = np.roll(u, -1)
            um = np.roll(u, 1)
            # conservative-ish: 0.5 * d(u^2)/dx + viscous
            adv = (up**2 - um**2) / (4.0 * dx)
            visc = nu * (up - 2.0 * u + um) / (dx**2)
            u = u - dt_sub * adv + dt_sub * visc
        traj[n + 1] = u

    t = np.linspace(0.0, T, nt + 1)
    return {
        "x": x,
        "t": t,
        "u": traj,
        "nu": float(nu),
        "meta": {
            "nx": nx,
            "nt": nt,
            "L": float(L),
            "T": float(T),
            "dx": float(dx),
            "dt": float(dt),
            "n_sub": n_sub,
            "equation": "burgers1d",
        },
    }


def generate_dataset(
    n_traj: int = 48,
    nx: int = 64,
    nt: int = 80,
    nu_range: tuple[float, float] = (0.005, 0.05),
    seed: int = 0,
) -> dict:
    """Generate a small Burgers trajectory set for smoke / Week-1 eval."""
    rng = np.random.default_rng(seed)
    nus = rng.uniform(nu_range[0], nu_range[1], size=n_traj)
    us = []
    xs = None
    ts = None
    for i, nu in enumerate(nus):
        out = solve_burgers(nu=float(nu), nx=nx, nt=nt, seed=int(rng.integers(0, 2**31 - 1)))
        us.append(out["u"])
        if xs is None:
            xs, ts = out["x"], out["t"]
    return {
        "u": np.stack(us, axis=0),  # (n_traj, nt+1, nx)
        "x": xs,
        "t": ts,
        "nu": nus.astype(np.float64),
        "meta": {
            "n_traj": n_traj,
            "nx": nx,
            "nt": nt,
            "nu_range": list(nu_range),
            "seed": seed,
            "equation": "burgers1d",
        },
    }
