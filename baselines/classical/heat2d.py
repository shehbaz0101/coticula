"""Classical 2D heat equation solver (Dirichlet, explicit FTCS).

du/dt = alpha (d^2u/dx^2 + d^2u/dy^2)
"""
from __future__ import annotations

import numpy as np


def solve_heat2d(
    alpha: float = 0.1,
    n: int = 32,
    nt: int = 50,
    L: float = 1.0,
    T: float = 0.5,
    u0: np.ndarray | None = None,
    seed: int | None = None,
) -> dict:
    """Integrate 2D heat on a square with zero Dirichlet BC.

    Returns dict with keys: x, y, t, u (nt+1, n, n), alpha, meta.
    Interior unknowns are (n-2)x(n-2); grid includes boundaries.
    """
    dx = L / (n - 1)
    dy = dx
    dt = T / nt
    x = np.linspace(0.0, L, n)
    y = np.linspace(0.0, L, n)

    if u0 is None:
        rng = np.random.default_rng(seed)
        # bump interior
        X, Y = np.meshgrid(x, y, indexing="ij")
        cx, cy = rng.uniform(0.3, 0.7, size=2)
        sx, sy = rng.uniform(0.08, 0.18, size=2)
        u = np.exp(-((X - cx) ** 2 / (2 * sx**2) + (Y - cy) ** 2 / (2 * sy**2)))
        u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0
    else:
        u = np.asarray(u0, dtype=np.float64).copy()
        if u.shape != (n, n):
            raise ValueError(f"u0 shape {u.shape} != ({n}, {n})")
        u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0

    r = alpha * dt / (dx**2)
    if r > 0.24:
        n_sub = int(np.ceil(r / 0.2))
        dt_sub = dt / n_sub
        r = alpha * dt_sub / (dx**2)
    else:
        n_sub = 1
        dt_sub = dt

    traj = np.zeros((nt + 1, n, n), dtype=np.float64)
    traj[0] = u

    for step in range(nt):
        for _ in range(n_sub):
            lap = (
                np.roll(u, 1, axis=0)
                + np.roll(u, -1, axis=0)
                + np.roll(u, 1, axis=1)
                + np.roll(u, -1, axis=1)
                - 4.0 * u
            )
            # Dirichlet: do not update boundaries via roll; zero them after
            u = u + r * lap
            u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0
        traj[step + 1] = u

    t = np.linspace(0.0, T, nt + 1)
    return {
        "x": x,
        "y": y,
        "t": t,
        "u": traj,
        "alpha": float(alpha),
        "meta": {
            "n": n,
            "nt": nt,
            "L": float(L),
            "T": float(T),
            "dx": float(dx),
            "dt": float(dt),
            "n_sub": n_sub,
            "equation": "heat2d",
        },
    }


def generate_dataset(
    n_traj: int = 32,
    n: int = 32,
    nt: int = 40,
    alpha_range: tuple[float, float] = (0.05, 0.2),
    seed: int = 1,
) -> dict:
    """Generate a small heat-2D trajectory set for smoke / Week-1 eval."""
    rng = np.random.default_rng(seed)
    alphas = rng.uniform(alpha_range[0], alpha_range[1], size=n_traj)
    us = []
    xs = ys = ts = None
    for i, alpha in enumerate(alphas):
        out = solve_heat2d(
            alpha=float(alpha),
            n=n,
            nt=nt,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        us.append(out["u"])
        if xs is None:
            xs, ys, ts = out["x"], out["y"], out["t"]
    return {
        "u": np.stack(us, axis=0),  # (n_traj, nt+1, n, n)
        "x": xs,
        "y": ys,
        "t": ts,
        "alpha": alphas.astype(np.float64),
        "meta": {
            "n_traj": n_traj,
            "n": n,
            "nt": nt,
            "alpha_range": list(alpha_range),
            "seed": seed,
            "equation": "heat2d",
        },
    }
