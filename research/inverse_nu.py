"""Small Burgers viscosity recovery: strong LS, weak LS, solver-in-the-loop.

The FNO-conditioned inverse from the draft is not run. There is no separate
surrogate-training budget for it in this protocol.
"""
from __future__ import annotations

import numpy as np

from baselines.classical.burgers1d import generate_dataset as generate_burgers, solve_burgers
from metrics.predict import relative_l2
from metrics.weak_form import WeakSpec, burgers_weak_components
from research.strong import mean_std


def sensor_indices(nx: int, n_sensors: int) -> np.ndarray:
    idx = np.linspace(0, nx, int(n_sensors), endpoint=False).astype(int)
    if len(np.unique(idx)) != int(n_sensors):
        raise ValueError("sensor layout collapsed; choose n_sensors that divides the grid")
    return idx


def interpolate_periodic(samples: np.ndarray, sensor_idx: np.ndarray, nx: int) -> np.ndarray:
    """Linear interpolation on the circle from evenly or unevenly spaced sensors."""
    full = np.zeros((samples.shape[0], nx), dtype=np.float64)
    xs = sensor_idx.astype(np.float64)
    grid = np.arange(nx, dtype=np.float64)
    xs_ext = np.concatenate([xs - nx, xs, xs + nx])
    for step in range(samples.shape[0]):
        ys = np.concatenate([samples[step], samples[step], samples[step]])
        full[step] = np.interp(grid, xs_ext, ys)
    return full


def strong_least_squares_nu(u: np.ndarray, dx: float, dt: float) -> float:
    """Solve ut + d_x(u^2)/2 = nu u_xx in the least-squares sense on the FD stencil."""
    mid = 0.5 * (u[:-1] + u[1:])
    ut = (u[1:] - u[:-1]) / dt
    up = np.roll(mid, -1, axis=-1)
    um = np.roll(mid, 1, axis=-1)
    adv = (up**2 - um**2) / (4.0 * dx)
    lap = (up - 2.0 * mid + um) / (dx**2)
    rhs = ut + adv
    return float(np.sum(lap * rhs) / (np.sum(lap * lap) + 1e-18))


def weak_least_squares_nu(u: np.ndarray, x: np.ndarray, t: np.ndarray, spec: WeakSpec) -> float:
    """Solve A + nu B = 0 in least squares over the test functions."""
    a_terms, b_terms = burgers_weak_components(u, x, t, spec)
    return float(-np.dot(b_terms, a_terms) / (np.dot(b_terms, b_terms) + 1e-18))


def solver_in_the_loop(
    u0: np.ndarray,
    observed: np.ndarray,
    sensor_idx: np.ndarray,
    grid: np.ndarray,
    *,
    nx: int,
    nt: int,
    length: float,
    horizon: float,
) -> float:
    best_nu = float(grid[0])
    best_err = float("inf")
    denom = float(np.linalg.norm(observed)) + 1e-12
    for nu in grid:
        sol = solve_burgers(
            nu=float(nu), nx=nx, nt=nt, L=length, T=horizon, u0=u0
        )["u"]
        err = float(np.linalg.norm(sol[:, sensor_idx] - observed) / denom)
        if err < best_err:
            best_err = err
            best_nu = float(nu)
    return best_nu


def _summarize(estimates: list[float], truth: np.ndarray, failure: float) -> dict:
    rel = np.abs(np.asarray(estimates) - truth) / np.maximum(np.abs(truth), 1e-12)
    mean, std = mean_std(rel)
    return {
        "rel_err_mean": mean,
        "rel_err_std": std,
        "n": int(rel.size),
        "n_fail": int(np.sum(rel > failure)),
        "failure_threshold": failure,
        "estimates": [float(v) for v in estimates],
    }


def run_inverse(protocol: dict, spec: WeakSpec) -> dict:
    cfg = protocol["inverse"]
    nx = int(cfg["nx"])
    nt = int(cfg["nt"])
    n_sensors = int(cfg["n_sensors"])
    sensors = sensor_indices(nx, n_sensors)
    search = np.linspace(float(cfg["search_min"]), float(cfg["search_max"]), int(cfg["search_points"]))
    failure = float(cfg["failure_rel_err"])
    length = float(cfg["L"])
    horizon = float(cfg["T"])

    def one_split(name: str) -> dict:
        block = cfg[name]
        data = generate_burgers(
            n_traj=int(block["n_traj"]),
            nx=nx,
            nt=nt,
            nu_range=(float(block["nu_range"][0]), float(block["nu_range"][1])),
            seed=int(block["seed"]),
        )
        dx = float(data["x"][1] - data["x"][0])
        dt = float(data["t"][1] - data["t"][0])
        strong, weak, loop = [], [], []
        for i in range(data["u"].shape[0]):
            observed = data["u"][i][:, sensors]
            interp = interpolate_periodic(observed, sensors, nx)
            strong.append(strong_least_squares_nu(interp, dx, dt))
            weak.append(weak_least_squares_nu(interp, data["x"], data["t"], spec))
            loop.append(
                solver_in_the_loop(
                    data["u"][i, 0],
                    observed,
                    sensors,
                    search,
                    nx=nx,
                    nt=nt,
                    length=length,
                    horizon=horizon,
                )
            )
        truth = data["nu"]
        return {
            "strong_form_ls": _summarize(strong, truth, failure),
            "weak_form_ls": _summarize(weak, truth, failure),
            "solver_in_the_loop": _summarize(loop, truth, failure),
            "fno_conditioned": {
                "status": "not_run",
                "reason": "no separate surrogate-training budget in this protocol",
            },
            "nu_true_mean": float(truth.mean()),
        }

    return {
        "sensors": sensors.tolist(),
        "search_grid": search.tolist(),
        "observation_noise": 0.0,
        "id": one_split("id"),
        "hard_ood": one_split("hard_ood"),
    }


def sensor_rel_l2_smoke(nx: int = 16) -> float:
    """Identity check used by tests: interpolating every node is exact."""
    idx = sensor_indices(nx, nx)
    rng = np.random.default_rng(0)
    field = rng.normal(size=(4, nx))
    recon = interpolate_periodic(field[:, idx], idx, nx)
    return relative_l2(recon, field)
