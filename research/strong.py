"""Strong-form residual, effective wavenumber, and the white-noise formulas.

The per-step normalization matches ``metrics.conserve``: the reported residual
is the mean over time of ||R||_2 / ||u_mid||_2. Vectorized helpers are checked
against that module in the unit tests. Heat keeps homogeneous Dirichlet
boundaries; spectral quantities on heat use a sine basis.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import dstn


def burgers_strong_per_traj(
    u: np.ndarray,
    dx: float,
    dt: float,
    nu: float | np.ndarray,
) -> np.ndarray:
    """Match ``metrics.conserve.burgers_residual_l2`` per trajectory."""
    u = np.asarray(u, dtype=np.float64)
    if u.ndim == 2:
        u = u[None, ...]
    u_n = u[:, :-1, :]
    u_np = u[:, 1:, :]
    u_mid = 0.5 * (u_n + u_np)
    ut = (u_np - u_n) / dt
    up = np.roll(u_mid, -1, axis=-1)
    um = np.roll(u_mid, 1, axis=-1)
    adv = (up**2 - um**2) / (4.0 * dx)
    nus = np.asarray(nu, dtype=np.float64).reshape(-1, 1, 1)
    visc = nus * (up - 2.0 * u_mid + um) / (dx**2)
    residual = ut + adv - visc
    num = np.sqrt((residual * residual).sum(axis=-1))
    den = np.sqrt((u_mid * u_mid).sum(axis=-1)) + 1e-12
    return (num / den).mean(axis=-1)


def burgers_residual_field(
    u: np.ndarray,
    dx: float,
    dt: float,
    nu: float,
) -> np.ndarray:
    """Pointwise midpoint residual, shape (nt, nx), periodic Burgers."""
    u = np.asarray(u, dtype=np.float64)
    u_n = u[:-1]
    u_np = u[1:]
    u_mid = 0.5 * (u_n + u_np)
    ut = (u_np - u_n) / dt
    up = np.roll(u_mid, -1, axis=-1)
    um = np.roll(u_mid, 1, axis=-1)
    adv = (up**2 - um**2) / (4.0 * dx)
    visc = float(nu) * (up - 2.0 * u_mid + um) / (dx**2)
    return ut + adv - visc


def heat_strong_per_traj(
    u: np.ndarray,
    dx: float,
    dt: float,
    alpha: float | np.ndarray,
) -> np.ndarray:
    """Match ``metrics.conserve.heat_residual_l2`` per trajectory.

    Boundaries of the residual are dropped. The field is assumed to carry
    explicit zero Dirichlet edges; ``roll`` then reproduces the 5-point stencil
    on interior nodes.
    """
    u = np.asarray(u, dtype=np.float64)
    if u.ndim == 3:
        u = u[None, ...]
    u_n = u[:, :-1]
    u_np = u[:, 1:]
    u_mid = 0.5 * (u_n + u_np)
    ut = (u_np - u_n) / dt
    lap = (
        np.roll(u_mid, 1, axis=-2)
        + np.roll(u_mid, -1, axis=-2)
        + np.roll(u_mid, 1, axis=-1)
        + np.roll(u_mid, -1, axis=-1)
        - 4.0 * u_mid
    ) / (dx**2)
    alphas = np.asarray(alpha, dtype=np.float64).reshape(-1, 1, 1, 1)
    residual = ut - alphas * lap
    interior_r = residual[:, :, 1:-1, 1:-1]
    interior_u = u_mid[:, :, 1:-1, 1:-1]
    num = np.sqrt((interior_r * interior_r).sum(axis=(-1, -2)))
    den = np.sqrt((interior_u * interior_u).sum(axis=(-1, -2))) + 1e-12
    return (num / den).mean(axis=-1)


def heat_residual_field(
    u: np.ndarray,
    dx: float,
    dt: float,
    alpha: float,
) -> np.ndarray:
    """Pointwise midpoint heat residual on the full grid, shape (nt, n, n)."""
    u = np.asarray(u, dtype=np.float64)
    u_n = u[:-1]
    u_np = u[1:]
    u_mid = 0.5 * (u_n + u_np)
    ut = (u_np - u_n) / dt
    lap = (
        np.roll(u_mid, 1, axis=-2)
        + np.roll(u_mid, -1, axis=-2)
        + np.roll(u_mid, 1, axis=-1)
        + np.roll(u_mid, -1, axis=-1)
        - 4.0 * u_mid
    ) / (dx**2)
    return ut - float(alpha) * lap


def heat_white_second_moment(dt: float, h: float, alpha: float, dim: int = 2) -> float:
    """E|R_j|^2 / s^2 for i.i.d. unit-variance noise. Proposition (white-error).

    E|R|^2 = s^2 (2/dt^2 + d(2d+1) alpha^2 / h^4). This returns the factor
    multiplying s^2.
    """
    d = int(dim)
    return (2.0 / dt**2) + (d * (2 * d + 1) * float(alpha) ** 2) / (h**4)


def heat_white_C_h(dt: float, h: float, alpha: float, dim: int = 2) -> float:
    """C_h = dt^{-1} (2 + d(2d+1) r^2)^{1/2}, r = alpha dt / h^2."""
    return float(np.sqrt(heat_white_second_moment(dt, h, alpha, dim)))


def burgers_white_second_moment_field(
    u: np.ndarray,
    dx: float,
    dt: float,
    nu: float,
    s2: float,
) -> np.ndarray:
    """Pointwise E|R[u+e] - tau|^2 from the Burgers white-noise formula.

    ``u`` is the clean trajectory. ``s2`` is the variance of the i.i.d. noise.
    """
    u = np.asarray(u, dtype=np.float64)
    mid = 0.5 * (u[:-1] + u[1:])
    up = np.roll(mid, -1, axis=-1)
    um = np.roll(mid, 1, axis=-1)
    d0 = (up - um) / (2.0 * dx)
    nu = float(nu)
    s2 = float(s2)
    return s2 * (
        2.0 / dt**2
        + 3.0 * nu**2 / dx**4
        + (up**2 + um**2) / (8.0 * dx**2)
        - nu * d0 / dx**2
    ) + (s2**2) / (16.0 * dx**2)


def lambda_symbol_1d(n: int, h: float) -> np.ndarray:
    """λ_h(k) = (4/h^2) sin^2(k h / 2) on the periodic DFT frequencies."""
    m = np.fft.fftfreq(n) * n
    return (4.0 / h**2) * np.sin(np.pi * m / n) ** 2


def keff_periodic(field: np.ndarray, h: float) -> float:
    """Energy-weighted effective wavenumber on a periodic 1D grid.

    keff = (sum λ(k)^2 |ê|^2 / sum |ê|^2)^{1/4}. Leading axes are pooled.
    """
    field = np.asarray(field, dtype=np.float64)
    n = field.shape[-1]
    spec = np.fft.fft(field, axis=-1)
    power = np.abs(spec) ** 2
    lam = lambda_symbol_1d(n, h)
    # Broadcast λ over every axis except the last.
    view = lam.reshape((1,) * (power.ndim - 1) + (n,))
    num = float((view**2 * power).sum())
    den = float(power.sum())
    if den <= 0:
        return float("nan")
    return float((num / den) ** 0.25)


def keff_dirichlet(field: np.ndarray, h: float) -> float:
    """Sine-basis effective wavenumber for a 2D Dirichlet field.

    ``field`` includes the zero boundary. The interior is expanded in DST-I,
    whose modes are sin(π p j / (n-1)) and match the 5-point symbol
    λ = (4/h^2) sin^2(π p / (2(n-1))).
    """
    field = np.asarray(field, dtype=np.float64)
    interior = field[..., 1:-1, 1:-1]
    n_int = interior.shape[-1]
    if n_int < 1:
        return float("nan")
    spec = dstn(interior, type=1, axes=(-2, -1))
    power = np.abs(spec) ** 2
    # DST-I length N corresponds to modes 1..N with denominator N+1 = n-1.
    m_denom = n_int + 1
    p = np.arange(1, n_int + 1, dtype=np.float64)
    lam_1d = (4.0 / h**2) * np.sin(np.pi * p / (2.0 * m_denom)) ** 2
    lam = lam_1d[:, None] + lam_1d[None, :]
    view = lam.reshape((1,) * (power.ndim - 2) + lam.shape)
    num = float((view**2 * power).sum())
    den = float(power.sum())
    if den <= 0:
        return float("nan")
    return float((num / den) ** 0.25)


def mean_std(values) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))
