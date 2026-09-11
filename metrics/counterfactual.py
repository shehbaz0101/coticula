"""Counterfactual helper: re-solve with a changed parameter and score delta."""
from __future__ import annotations

import numpy as np

from baselines.classical.burgers1d import solve_burgers
from baselines.classical.heat2d import solve_heat2d
from metrics.predict import nmse, relative_l2


def burgers_counterfactual(
    u0: np.ndarray,
    nu_orig: float,
    nu_cf: float,
    nx: int,
    nt: int,
    L: float = 2 * np.pi,
    T: float = 1.0,
) -> dict:
    """Re-solve Burgers from same IC under nu_cf; compare to orig trajectory."""
    base = solve_burgers(nu=nu_orig, nx=nx, nt=nt, L=L, T=T, u0=u0)
    cf = solve_burgers(nu=nu_cf, nx=nx, nt=nt, L=L, T=T, u0=u0)
    delta = relative_l2(cf["u"], base["u"])
    return {
        "nu_orig": float(nu_orig),
        "nu_cf": float(nu_cf),
        "rel_l2_traj_delta": delta,
        "final_rel_l2": relative_l2(cf["u"][-1], base["u"][-1]),
        "u_base_final_norm": float(np.linalg.norm(base["u"][-1])),
        "u_cf_final_norm": float(np.linalg.norm(cf["u"][-1])),
    }


def grade_burgers_cf_prediction(
    pred_cf: np.ndarray,
    u0: np.ndarray,
    nu_orig: float,
    nu_cf: float,
    nx: int,
    nt: int,
    L: float = 2 * np.pi,
    T: float = 1.0,
) -> dict:
    """Grade a model's ν′ rollout against the classical solver at ν′ (Exam 3)."""
    truth_base = solve_burgers(nu=nu_orig, nx=nx, nt=nt, L=L, T=T, u0=u0)
    truth_cf = solve_burgers(nu=nu_cf, nx=nx, nt=nt, L=L, T=T, u0=u0)
    return {
        "nu_orig": float(nu_orig),
        "nu_cf": float(nu_cf),
        "rel_l2_vs_classical_cf": relative_l2(pred_cf, truth_cf["u"]),
        "nmse_vs_classical_cf": nmse(pred_cf, truth_cf["u"]),
        "rel_l2_traj_delta_classical": relative_l2(truth_cf["u"], truth_base["u"]),
        "final_rel_l2_vs_classical_cf": relative_l2(pred_cf[-1], truth_cf["u"][-1]),
        "u_cf_final_norm_classical": float(np.linalg.norm(truth_cf["u"][-1])),
        "u_cf_final_norm_pred": float(np.linalg.norm(pred_cf[-1])),
    }


def heat_counterfactual(
    u0: np.ndarray,
    alpha_orig: float,
    alpha_cf: float,
    n: int,
    nt: int,
    L: float = 1.0,
    T: float = 0.5,
) -> dict:
    """Re-solve heat from same IC under alpha_cf; compare to orig trajectory."""
    base = solve_heat2d(alpha=alpha_orig, n=n, nt=nt, L=L, T=T, u0=u0)
    cf = solve_heat2d(alpha=alpha_cf, n=n, nt=nt, L=L, T=T, u0=u0)
    delta = relative_l2(cf["u"], base["u"])
    return {
        "alpha_orig": float(alpha_orig),
        "alpha_cf": float(alpha_cf),
        "rel_l2_traj_delta": delta,
        "final_rel_l2": relative_l2(cf["u"][-1], base["u"][-1]),
        "u_base_final_norm": float(np.linalg.norm(base["u"][-1])),
        "u_cf_final_norm": float(np.linalg.norm(cf["u"][-1])),
    }


def grade_heat_cf_prediction(
    pred_cf: np.ndarray,
    u0: np.ndarray,
    alpha_orig: float,
    alpha_cf: float,
    n: int,
    nt: int,
    L: float = 1.0,
    T: float = 0.5,
) -> dict:
    """Grade a model's α′ rollout against the classical solver at α′ (Exam 3)."""
    truth_base = solve_heat2d(alpha=alpha_orig, n=n, nt=nt, L=L, T=T, u0=u0)
    truth_cf = solve_heat2d(alpha=alpha_cf, n=n, nt=nt, L=L, T=T, u0=u0)
    return {
        "alpha_orig": float(alpha_orig),
        "alpha_cf": float(alpha_cf),
        "rel_l2_vs_classical_cf": relative_l2(pred_cf, truth_cf["u"]),
        "nmse_vs_classical_cf": nmse(pred_cf, truth_cf["u"]),
        "rel_l2_traj_delta_classical": relative_l2(truth_cf["u"], truth_base["u"]),
        "final_rel_l2_vs_classical_cf": relative_l2(pred_cf[-1], truth_cf["u"][-1]),
        "u_cf_final_norm_classical": float(np.linalg.norm(truth_cf["u"][-1])),
        "u_cf_final_norm_pred": float(np.linalg.norm(pred_cf[-1])),
    }
