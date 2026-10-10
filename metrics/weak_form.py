"""Opt-in weak-form residuals for 1D Burgers and 2D heat.

This module is not used by the default Conserve metric or the trust rule.
The definition matches the evaluation residual in the draft of
"Small Errors, Large Residuals":

    rho_w(v) = sqrt(sum_m r_m[v]^2) / ( ||u||_2 * sqrt(tr G) )

where r_m is the PDE tested against phi_m (derivatives on the test function),
u is the reference field, psi_m = L* phi_m, and G_mn = <psi_m, psi_n> under the
same quadrature. For Burgers, psi_m = -d_t phi_m - nu d_xx phi_m (the linear
adjoint used for normalization; the residual itself keeps the convective term).

Test functions are separable: phi(x,t) = chi_b(t) * eta_a(x).
chi_b is a compact polynomial bump. eta_a is a Fourier mode on the periodic
Burgers interval, or a Dirichlet sine mode for heat.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WeakSpec:
    """Frozen test-function family. Fractions are relative to the time interval."""

    p_bump: int = 8
    k_phi: int = 3
    time_centers_frac: tuple[float, ...] = (0.30, 0.70)
    time_radius_frac: float = 0.20

    def as_dict(self) -> dict:
        return {
            "family": "time_polynomial_bump_times_spatial_eigenmode",
            "p_bump": self.p_bump,
            "k_phi": self.k_phi,
            "time_centers_frac": list(self.time_centers_frac),
            "time_radius_frac": self.time_radius_frac,
            "burgers_spatial": "fourier_cos_sin_k_0_to_K_inclusive",
            "heat_spatial": "dirichlet_sine_1_to_K_inclusive_tensor",
            "normalization": (
                "sqrt(sum_m r_m^2) / (||u_ref||_2 * sqrt(tr G)), "
                "G_mn = <psi_m, psi_n>, psi = L* phi"
            ),
            "burgers_psi": "-d_t phi - nu d_xx phi",
            "heat_psi": "-d_t phi - alpha Delta phi",
            "reference_norm": "space-time quadrature L2 of the reference field u, not of the prediction",
        }


def spec_from_protocol(block: dict) -> WeakSpec:
    return WeakSpec(
        p_bump=int(block["p_bump"]),
        k_phi=int(block["k_phi"]),
        time_centers_frac=tuple(float(x) for x in block["time_centers_frac"]),
        time_radius_frac=float(block["time_radius_frac"]),
    )


def trapezoid_weights(n: int, h: float) -> np.ndarray:
    """Composite trapezoid weights on n nodes spaced by h."""
    if n < 2:
        raise ValueError("need at least two nodes")
    w = np.full(n, h, dtype=np.float64)
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def poly_bump(t: np.ndarray, center: float, radius: float, p: int) -> tuple[np.ndarray, np.ndarray]:
    """WSINDy polynomial bump and its analytic time derivative.

    chi(z) = (1 - z^2)^p for |z| < 1, else 0, z = (t - center) / radius.
    chi and chi_t vanish at the edge of the support when p >= 2.
    """
    t = np.asarray(t, dtype=np.float64)
    z = (t - center) / radius
    chi = np.zeros_like(z)
    chi_t = np.zeros_like(z)
    mask = np.abs(z) < 1.0
    if not np.any(mask):
        return chi, chi_t
    zm = z[mask]
    one = 1.0 - zm * zm
    chi[mask] = one**p
    chi_t[mask] = (p * one ** (p - 1) * (-2.0 * zm)) / radius
    return chi, chi_t


def _time_bumps(t: np.ndarray, spec: WeakSpec) -> list[tuple[np.ndarray, np.ndarray]]:
    t = np.asarray(t, dtype=np.float64)
    t0 = float(t[0])
    t1 = float(t[-1])
    span = t1 - t0
    if span <= 0:
        raise ValueError("time interval must have positive length")
    radius = spec.time_radius_frac * span
    bumps = []
    for frac in spec.time_centers_frac:
        center = t0 + frac * span
        chi, chi_t = poly_bump(t, center, radius, spec.p_bump)
        if float(np.max(chi)) <= 0:
            raise ValueError(f"bump at fraction {frac} has empty support on this grid")
        bumps.append((chi, chi_t))
    return bumps


def _burgers_modes(x: np.ndarray, k_phi: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
    """Real Fourier modes on a periodic grid. k = 0..K, cos and sin for k > 0."""
    x = np.asarray(x, dtype=np.float64)
    modes = []
    eta = np.ones_like(x)
    zero = np.zeros_like(x)
    modes.append((eta, zero, zero, "k0"))
    for k in range(1, int(k_phi) + 1):
        kx = k * x
        ck = np.cos(kx)
        sk = np.sin(kx)
        modes.append((ck, -k * sk, -(k**2) * ck, f"cos{k}"))
        modes.append((sk, k * ck, -(k**2) * sk, f"sin{k}"))
    return modes


def _heat_modes(x: np.ndarray, y: np.ndarray, length: float, k_phi: int):
    """Dirichlet sine modes sin(p π x / L) sin(q π y / L), 1 <= p,q <= K."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    modes = []
    for p in range(1, int(k_phi) + 1):
        sx = np.sin(p * np.pi * x / length)
        for q in range(1, int(k_phi) + 1):
            sy = np.sin(q * np.pi * y / length)
            eta = sx[:, None] * sy[None, :]
            mu = (p * np.pi / length) ** 2 + (q * np.pi / length) ** 2
            modes.append((eta, mu, f"sin{p}_{q}"))
    return modes


def burgers_weak_per_traj(
    u: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    nu: float | np.ndarray,
    spec: WeakSpec,
    *,
    u_ref: np.ndarray | None = None,
) -> np.ndarray:
    """Per-trajectory weak residual. u: (nt+1, nx) or (B, nt+1, nx).

    ``u_ref`` is the reference field in the denominator. It defaults to ``u``
    (the right choice for a solver output, and for coefficient-mismatch probes
    that reuse the true field). Model scores pass the labels here.
    """
    u = np.asarray(u, dtype=np.float64)
    single = u.ndim == 2
    if single:
        u = u[None, ...]
    if u.ndim != 3:
        raise ValueError(f"burgers field must be (T, X) or (B, T, X), got {u.shape}")
    ref = u if u_ref is None else np.asarray(u_ref, dtype=np.float64)
    if ref.ndim == 2:
        ref = np.broadcast_to(ref, u.shape).copy()
    if ref.shape != u.shape:
        raise ValueError(f"u_ref shape {ref.shape} != u shape {u.shape}")

    bsz, _nt1, nx = u.shape
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if x.shape != (nx,) or t.shape[0] != u.shape[1]:
        raise ValueError("grid does not match field")
    dx = float(x[1] - x[0])
    dt = float(t[1] - t[0])
    w_t = trapezoid_weights(t.shape[0], dt)
    weight = w_t[:, None] * dx
    nus = np.broadcast_to(np.asarray(nu, dtype=np.float64).reshape(-1), (bsz,)).astype(np.float64)

    r_cols = []
    tr_cols = []
    for chi, chi_t in _time_bumps(t, spec):
        for eta, eta_x, eta_xx, _name in _burgers_modes(x, spec.k_phi):
            dphi_dt = chi_t[:, None] * eta[None, :]
            dphi_dx = chi[:, None] * eta_x[None, :]
            dphi_xx = chi[:, None] * eta_xx[None, :]
            term = (
                -u * dphi_dt[None, ...]
                - 0.5 * (u * u) * dphi_dx[None, ...]
                - nus[:, None, None] * u * dphi_xx[None, ...]
            )
            r_cols.append((term * weight[None, ...]).sum(axis=(-1, -2)))
            a = float((dphi_dt * dphi_dt * weight).sum())
            b = float((dphi_dt * dphi_xx * weight).sum())
            c = float((dphi_xx * dphi_xx * weight).sum())
            tr_cols.append(a + 2.0 * nus * b + (nus**2) * c)

    residual = np.stack(r_cols, axis=1)
    tr_g = np.stack(tr_cols, axis=1).sum(axis=1)
    u_norm = np.sqrt((ref * ref * weight[None, ...]).sum(axis=(-1, -2)))
    num = np.sqrt((residual * residual).sum(axis=1))
    den = u_norm * np.sqrt(np.maximum(tr_g, 0.0)) + 1e-12
    out = num / den
    return out


def heat_weak_per_traj(
    u: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    alpha: float | np.ndarray,
    spec: WeakSpec,
    *,
    u_ref: np.ndarray | None = None,
) -> np.ndarray:
    """Per-trajectory weak residual. u: (nt+1, n, n) or (B, nt+1, n, n).

    Sine modes vanish on the boundary of the unit square, so integration by
    parts is compatible with homogeneous Dirichlet conditions.
    """
    u = np.asarray(u, dtype=np.float64)
    single = u.ndim == 3
    if single:
        u = u[None, ...]
    if u.ndim != 4:
        raise ValueError(f"heat field must be (T, n, n) or (B, T, n, n), got {u.shape}")
    ref = u if u_ref is None else np.asarray(u_ref, dtype=np.float64)
    if ref.ndim == 3:
        ref = np.broadcast_to(ref, u.shape).copy()
    if ref.shape != u.shape:
        raise ValueError(f"u_ref shape {ref.shape} != u shape {u.shape}")

    bsz, _nt1, n, n2 = u.shape
    if n != n2:
        raise ValueError("heat field must be square")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    dt = float(t[1] - t[0])
    if abs(dx - dy) > 1e-12:
        raise ValueError("heat quadrature assumes dx == dy")
    length = float(x[-1] - x[0])
    w_t = trapezoid_weights(t.shape[0], dt)
    w_x = trapezoid_weights(n, dx)
    weight = w_t[:, None, None] * w_x[None, :, None] * w_x[None, None, :]
    alphas = np.broadcast_to(np.asarray(alpha, dtype=np.float64).reshape(-1), (bsz,)).astype(
        np.float64
    )

    r_cols = []
    tr_cols = []
    for chi, chi_t in _time_bumps(t, spec):
        for eta, mu, _name in _heat_modes(x, y, length, spec.k_phi):
            # A = d_t phi, B = mu * phi, psi = -A + alpha * B, Delta phi = -mu phi
            a_field = chi_t[:, None, None] * eta[None, ...]
            b_field = mu * chi[:, None, None] * eta[None, ...]
            # r = <u, psi> = -<u, A> + alpha <u, B>
            ua = (u * a_field[None, ...] * weight[None, ...]).sum(axis=(-1, -2, -3))
            ub = (u * b_field[None, ...] * weight[None, ...]).sum(axis=(-1, -2, -3))
            r_cols.append(-ua + alphas * ub)
            aa = float((a_field * a_field * weight).sum())
            ab = float((a_field * b_field * weight).sum())
            bb = float((b_field * b_field * weight).sum())
            tr_cols.append(aa - 2.0 * alphas * ab + (alphas**2) * bb)

    residual = np.stack(r_cols, axis=1)
    tr_g = np.stack(tr_cols, axis=1).sum(axis=1)
    u_norm = np.sqrt((ref * ref * weight[None, ...]).sum(axis=(-1, -2, -3)))
    num = np.sqrt((residual * residual).sum(axis=1))
    den = u_norm * np.sqrt(np.maximum(tr_g, 0.0)) + 1e-12
    return num / den


def burgers_weak_components(
    u: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    spec: WeakSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (r_m, ||psi_m||^2) for one trajectory, for least-squares viscosity fits."""
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 2:
        raise ValueError("expected a single Burgers trajectory")
    dx = float(x[1] - x[0])
    dt = float(t[1] - t[0])
    weight = trapezoid_weights(t.shape[0], dt)[:, None] * dx
    a_terms = []
    b_terms = []
    for chi, chi_t in _time_bumps(t, spec):
        for eta, eta_x, eta_xx, _name in _burgers_modes(x, spec.k_phi):
            dphi_dt = chi_t[:, None] * eta[None, :]
            dphi_dx = chi[:, None] * eta_x[None, :]
            dphi_xx = chi[:, None] * eta_xx[None, :]
            # r = A + nu * B, with B = -<u, d_xx phi>, A = -<u, d_t phi> - <u^2/2, d_x phi>
            a = float(((-u * dphi_dt - 0.5 * (u * u) * dphi_dx) * weight).sum())
            b = float(((-u * dphi_xx) * weight).sum())
            a_terms.append(a)
            b_terms.append(b)
    return np.asarray(a_terms), np.asarray(b_terms)


def discrete_cs_bound_heat(
    e: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    alpha: float,
    spec: WeakSpec,
) -> dict:
    """Cauchy-Schwarz check: sqrt(sum <e, psi>^2) / sqrt(tr G) <= ||e||.

    Uses the reference-free numerator of rho_w. The field ``e`` is tested
    directly, so this does not require a PDE solution.
    """
    e = np.asarray(e, dtype=np.float64)
    if e.ndim != 3:
        raise ValueError("expected one heat trajectory")
    # Reuse the residual with u_ref = e, then multiply back by ||e||.
    rho = float(heat_weak_per_traj(e, x, y, t, alpha, spec)[0])
    # rho = num / ||e||, and num / sqrt(tr G) = rho * ||e|| <= ||e||, so rho <= 1
    # when r_m = <e, psi_m> exactly, which holds for heat.
    return {"rho_over_rel": rho, "bound": 1.0}
