"""OOD / transfer probe helpers for Burgers and heat2d.

Probes (each isolates one shift from v0 train support):

- **param_shift** — viscosity / diffusivity outside the label ``*_range``
- **resolution_shift** — spatial grid ≠ train nx / n (FNO spectral conv is
  resolution-agnostic in space; time is still locked as output channels)
- **ic_family_shift** — IC recipe ≠ the generator used for labels

Ground truth is always a fresh classical FD solve. Learned scores are vs that
solve, reported separately from IID exams 1–3.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

from baselines.classical.burgers1d import solve_burgers
from baselines.classical.heat2d import solve_heat2d
from metrics.conserve import audit_burgers, audit_heat
from metrics.predict import batch_relative_l2
from metrics.trust import (
    BURGERS_TRAIN_SUPPORT,
    HEAT_TRAIN_SUPPORT,
    attach_trust,
)

# Isolated OOD knobs (outside train ranges in burgers_v0_meta / heat2d_v0_meta).
BURGERS_OOD_NU = {"below": 0.001, "above": 0.15}
HEAT_OOD_ALPHA = {"below": 0.01, "above": 0.50}
BURGERS_OOD_NX = (32, 128)
HEAT_OOD_N = (16, 48)
BURGERS_OOD_IC = ("gaussian_pulse", "tanh_front")
HEAT_OOD_IC = ("two_bump", "sinusoid")


def make_burgers_ic(
    nx: int,
    L: float,
    family: str,
    seed: int,
) -> np.ndarray:
    """Initial condition on a periodic 1D grid. ``fourier_modes`` matches labels."""
    rng = np.random.default_rng(int(seed))
    x = np.linspace(0.0, L, nx, endpoint=False)
    if family == "fourier_modes":
        modes = rng.normal(size=4) + 1j * rng.normal(size=4)
        k = np.arange(1, 5)
        u = np.zeros(nx, dtype=np.float64)
        for m, kk in zip(modes, k):
            u += (m.real * np.cos(kk * x) + m.imag * np.sin(kk * x)) / kk
        return u / (np.max(np.abs(u)) + 1e-12)
    if family == "gaussian_pulse":
        c = float(rng.uniform(0.0, L))
        s = 0.12 * L
        d = np.minimum(np.abs(x - c), L - np.abs(x - c))
        u = np.exp(-0.5 * (d / s) ** 2)
        return u / (np.max(np.abs(u)) + 1e-12)
    if family == "tanh_front":
        c = float(rng.uniform(0.3 * L, 0.7 * L))
        w = 0.08 * L
        return np.tanh((x - c) / w)
    raise ValueError(f"unknown Burgers IC family {family!r}")


def make_heat_ic(n: int, L: float, family: str, seed: int) -> np.ndarray:
    """Initial condition on a Dirichlet square. ``gaussian_bump`` matches labels."""
    rng = np.random.default_rng(int(seed))
    x = np.linspace(0.0, L, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    if family == "gaussian_bump":
        cx, cy = rng.uniform(0.3, 0.7, size=2)
        sx, sy = rng.uniform(0.08, 0.18, size=2)
        u = np.exp(-((X - cx) ** 2 / (2 * sx**2) + (Y - cy) ** 2 / (2 * sy**2)))
    elif family == "two_bump":
        u = np.zeros((n, n), dtype=np.float64)
        for scale in (1.0, 0.7):
            cx, cy = rng.uniform(0.2, 0.8, size=2)
            s = float(rng.uniform(0.06, 0.12))
            u += scale * np.exp(
                -((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s**2)
            )
    elif family == "sinusoid":
        kx = int(rng.integers(2, 5))
        ky = int(rng.integers(2, 5))
        u = np.sin(kx * np.pi * X / L) * np.sin(ky * np.pi * Y / L)
    else:
        raise ValueError(f"unknown heat IC family {family!r}")
    u = np.asarray(u, dtype=np.float64)
    u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0
    return u


def resample_1d_periodic(u: np.ndarray, nx_new: int) -> np.ndarray:
    """Fourier resample of a periodic 1D field (resolution transfer)."""
    u = np.asarray(u, dtype=np.float64)
    nx = int(u.shape[0])
    nx_new = int(nx_new)
    if nx_new == nx:
        return u.copy()
    ft = np.fft.rfft(u)
    n_out = nx_new // 2 + 1
    ft_new = np.zeros(n_out, dtype=np.complex128)
    n_copy = min(ft.shape[0], n_out)
    ft_new[:n_copy] = ft[:n_copy]
    if nx_new < nx and n_out > 0:
        # drop Nyquist leftover from downsampling odd/even mismatch
        pass
    out = np.fft.irfft(ft_new, n=nx_new)
    out *= nx_new / nx
    return np.asarray(out, dtype=np.float64)


def resample_2d_dirichlet(u: np.ndarray, n_new: int) -> np.ndarray:
    """Bilinear zoom of a Dirichlet field; walls forced to 0."""
    u = np.asarray(u, dtype=np.float64)
    n = int(u.shape[0])
    n_new = int(n_new)
    if n_new == n:
        out = u.copy()
    else:
        out = zoom(u, n_new / n, order=1)
        if out.shape[0] != n_new or out.shape[1] != n_new:
            # fallback: pad or crop to exact n_new
            canvas = np.zeros((n_new, n_new), dtype=np.float64)
            h = min(n_new, out.shape[0])
            w = min(n_new, out.shape[1])
            canvas[:h, :w] = out[:h, :w]
            out = canvas
    out[0, :] = out[-1, :] = out[:, 0] = out[:, -1] = 0.0
    return np.asarray(out, dtype=np.float64)


def _score_burgers(
    pred: np.ndarray,
    truth: np.ndarray,
    nu: np.ndarray,
    dx: float,
    dt: float,
    *,
    support: dict,
    param: float,
    resolution: int,
    ic_family: str,
    role: str,
) -> dict:
    pred_m = batch_relative_l2(pred, truth)
    metrics = {
        "n": int(pred.shape[0]),
        "predict": {
            "rel_l2_mean": pred_m["mean"],
            "rel_l2_std": pred_m["std"],
            "nmse": pred_m["nmse"],
        },
        "conserve": audit_burgers(pred, dx=dx, dt=dt, nu=nu),
    }
    return attach_trust(
        metrics,
        support=support,
        param=param,
        resolution=resolution,
        ic_family=ic_family,
        role=role,
    )


def _score_heat(
    pred: np.ndarray,
    truth: np.ndarray,
    alpha: np.ndarray,
    dx: float,
    dt: float,
    *,
    support: dict,
    param: float,
    resolution: int,
    ic_family: str,
    role: str,
) -> dict:
    pred_m = batch_relative_l2(pred, truth)
    metrics = {
        "n": int(pred.shape[0]),
        "predict": {
            "rel_l2_mean": pred_m["mean"],
            "rel_l2_std": pred_m["std"],
            "nmse": pred_m["nmse"],
        },
        "conserve": audit_heat(pred, dx=dx, dt=dt, alpha=alpha),
    }
    return attach_trust(
        metrics,
        support=support,
        param=param,
        resolution=resolution,
        ic_family=ic_family,
        role=role,
    )


def generate_burgers_cases(
    *,
    n: int,
    nu: float,
    nx: int,
    family: str,
    seed: int,
    nt: int | None = None,
    L: float | None = None,
    T: float | None = None,
    resample_from: int | None = None,
) -> dict:
    """Classical Burgers trajectories for one isolated OOD (or IID) bucket."""
    support = BURGERS_TRAIN_SUPPORT
    nt = int(nt if nt is not None else support["nt"])
    L = float(L if L is not None else support["L"])
    T = float(T if T is not None else support["T"])
    us, u0s = [], []
    for i in range(int(n)):
        if resample_from is not None and int(resample_from) != int(nx):
            u0_src = make_burgers_ic(int(resample_from), L, family, seed + i)
            u0 = resample_1d_periodic(u0_src, nx)
        else:
            u0 = make_burgers_ic(nx, L, family, seed + i)
        out = solve_burgers(nu=float(nu), nx=int(nx), nt=nt, L=L, T=T, u0=u0)
        us.append(out["u"])
        u0s.append(out["u"][0])
        x, t = out["x"], out["t"]
    u = np.stack(us, axis=0)
    return {
        "u": u,
        "u0": np.stack(u0s, axis=0),
        "nu": np.full(int(n), float(nu), dtype=np.float64),
        "x": x,
        "t": t,
        "dx": float(x[1] - x[0]),
        "dt": float(t[1] - t[0]),
        "nx": int(nx),
        "nt": nt,
        "L": L,
        "T": T,
        "ic_family": family,
        "param": float(nu),
        "pde": "burgers1d",
    }


def generate_heat_cases(
    *,
    n_cases: int,
    alpha: float,
    n: int,
    family: str,
    seed: int,
    nt: int | None = None,
    L: float | None = None,
    T: float | None = None,
    resample_from: int | None = None,
) -> dict:
    """Classical heat-2D trajectories for one isolated OOD (or IID) bucket."""
    support = HEAT_TRAIN_SUPPORT
    nt = int(nt if nt is not None else support["nt"])
    L = float(L if L is not None else support["L"])
    T = float(T if T is not None else support["T"])
    us, u0s = [], []
    for i in range(int(n_cases)):
        if resample_from is not None and int(resample_from) != int(n):
            u0_src = make_heat_ic(int(resample_from), L, family, seed + i)
            u0 = resample_2d_dirichlet(u0_src, n)
        else:
            u0 = make_heat_ic(n, L, family, seed + i)
        out = solve_heat2d(alpha=float(alpha), n=int(n), nt=nt, L=L, T=T, u0=u0)
        us.append(out["u"])
        u0s.append(out["u"][0])
        x, t = out["x"], out["t"]
    u = np.stack(us, axis=0)
    return {
        "u": u,
        "u0": np.stack(u0s, axis=0),
        "alpha": np.full(int(n_cases), float(alpha), dtype=np.float64),
        "x": x,
        "t": t,
        "dx": float(x[1] - x[0]),
        "dt": float(t[1] - t[0]),
        "n_grid": int(n),
        "nt": nt,
        "L": L,
        "T": T,
        "ic_family": family,
        "param": float(alpha),
        "pde": "heat2d",
    }


def score_burgers_labeler(cases: dict) -> dict:
    """Self-consistency of the classical solve (labeler residual, not transfer)."""
    return _score_burgers(
        cases["u"],
        cases["u"],
        cases["nu"],
        cases["dx"],
        cases["dt"],
        support=BURGERS_TRAIN_SUPPORT,
        param=float(cases["param"]),
        resolution=int(cases["nx"]),
        ic_family=str(cases["ic_family"]),
        role="labeler",
    )


def score_heat_labeler(cases: dict) -> dict:
    return _score_heat(
        cases["u"],
        cases["u"],
        cases["alpha"],
        cases["dx"],
        cases["dt"],
        support=HEAT_TRAIN_SUPPORT,
        param=float(cases["param"]),
        resolution=int(cases["n_grid"]),
        ic_family=str(cases["ic_family"]),
        role="labeler",
    )


def score_burgers_surrogate(pred: np.ndarray, cases: dict) -> dict:
    return _score_burgers(
        pred,
        cases["u"],
        cases["nu"],
        cases["dx"],
        cases["dt"],
        support=BURGERS_TRAIN_SUPPORT,
        param=float(cases["param"]),
        resolution=int(cases["nx"]),
        ic_family=str(cases["ic_family"]),
        role="surrogate",
    )


def score_heat_surrogate(pred: np.ndarray, cases: dict) -> dict:
    return _score_heat(
        pred,
        cases["u"],
        cases["alpha"],
        cases["dx"],
        cases["dt"],
        support=HEAT_TRAIN_SUPPORT,
        param=float(cases["param"]),
        resolution=int(cases["n_grid"]),
        ic_family=str(cases["ic_family"]),
        role="surrogate",
    )


def default_probe_plan(*, n_param: int = 4, n_res: int = 2, n_ic: int = 2) -> list[dict]:
    """Small CPU-friendly probe list. Each item isolates one shift."""
    b = BURGERS_TRAIN_SUPPORT
    h = HEAT_TRAIN_SUPPORT
    nu_in = float(np.mean(b["param_range"]))
    a_in = float(np.mean(h["param_range"]))
    plan: list[dict] = []
    seed = 10_001
    for name, nu in BURGERS_OOD_NU.items():
        plan.append(
            {
                "id": f"burgers_param_{name}",
                "pde": "burgers",
                "probe": "param_shift",
                "n": n_param,
                "nu": float(nu),
                "nx": int(b["resolution"]),
                "family": b["ic_family"],
                "seed": seed,
            }
        )
        seed += 20
    for name, alpha in HEAT_OOD_ALPHA.items():
        plan.append(
            {
                "id": f"heat2d_param_{name}",
                "pde": "heat2d",
                "probe": "param_shift",
                "n": n_param,
                "alpha": float(alpha),
                "n_grid": int(h["resolution"]),
                "family": h["ic_family"],
                "seed": seed,
            }
        )
        seed += 20
    for nx in BURGERS_OOD_NX:
        plan.append(
            {
                "id": f"burgers_resolution_nx{nx}",
                "pde": "burgers",
                "probe": "resolution_shift",
                "n": n_res,
                "nu": nu_in,
                "nx": int(nx),
                "family": b["ic_family"],
                "resample_from": int(b["resolution"]),
                "seed": seed,
            }
        )
        seed += 20
    for ng in HEAT_OOD_N:
        plan.append(
            {
                "id": f"heat2d_resolution_n{ng}",
                "pde": "heat2d",
                "probe": "resolution_shift",
                "n": n_res,
                "alpha": a_in,
                "n_grid": int(ng),
                "family": h["ic_family"],
                "resample_from": int(h["resolution"]),
                "seed": seed,
            }
        )
        seed += 20
    for fam in BURGERS_OOD_IC:
        plan.append(
            {
                "id": f"burgers_ic_{fam}",
                "pde": "burgers",
                "probe": "ic_family_shift",
                "n": n_ic,
                "nu": nu_in,
                "nx": int(b["resolution"]),
                "family": fam,
                "seed": seed,
            }
        )
        seed += 20
    for fam in HEAT_OOD_IC:
        plan.append(
            {
                "id": f"heat2d_ic_{fam}",
                "pde": "heat2d",
                "probe": "ic_family_shift",
                "n": n_ic,
                "alpha": a_in,
                "n_grid": int(h["resolution"]),
                "family": fam,
                "seed": seed,
            }
        )
        seed += 20
    return plan


def materialize_probe(spec: dict) -> dict:
    """Generate classical cases for one plan item."""
    if spec["pde"] == "burgers":
        cases = generate_burgers_cases(
            n=int(spec["n"]),
            nu=float(spec["nu"]),
            nx=int(spec["nx"]),
            family=str(spec["family"]),
            seed=int(spec["seed"]),
            resample_from=spec.get("resample_from"),
        )
        cases["labeler"] = score_burgers_labeler(cases)
    elif spec["pde"] == "heat2d":
        cases = generate_heat_cases(
            n_cases=int(spec["n"]),
            alpha=float(spec["alpha"]),
            n=int(spec["n_grid"]),
            family=str(spec["family"]),
            seed=int(spec["seed"]),
            resample_from=spec.get("resample_from"),
        )
        cases["labeler"] = score_heat_labeler(cases)
    else:
        raise ValueError(f"unknown pde {spec['pde']!r}")
    cases["spec"] = {
        k: spec[k] for k in spec if k in ("id", "pde", "probe", "n", "family")
    }
    cases["spec"]["param"] = cases["param"]
    return cases
