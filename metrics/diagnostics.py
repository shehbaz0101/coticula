"""v0.2 eval diagnostics: residual-vs-L2 scatter and CF sensitivity.

Every number is computed from a field, a report that already holds measured
cells, or a fresh classical solve. Missing models stay ``not_trained``.
"""
from __future__ import annotations

from typing import Callable, Iterable, Mapping

import numpy as np

from metrics.conserve import burgers_residual_l2, heat_residual_l2
from metrics.counterfactual import (
    burgers_counterfactual,
    grade_burgers_cf_prediction,
    grade_heat_cf_prediction,
    heat_counterfactual,
)
from metrics.predict import relative_l2


def pearson_corr(x: Iterable[float], y: Iterable[float]) -> float | None:
    """Pearson r. ``None`` if undefined (n<2 or zero variance)."""
    xa = np.asarray(list(x), dtype=np.float64)
    ya = np.asarray(list(y), dtype=np.float64)
    if xa.size < 2 or ya.size < 2 or xa.size != ya.size:
        return None
    if float(np.std(xa)) < 1e-15 or float(np.std(ya)) < 1e-15:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def spearman_corr(x: Iterable[float], y: Iterable[float]) -> float | None:
    """Spearman rank correlation via midranks + Pearson. ``None`` if undefined."""
    xa = np.asarray(list(x), dtype=np.float64)
    ya = np.asarray(list(y), dtype=np.float64)
    if xa.size < 2 or ya.size != xa.size:
        return None
    return pearson_corr(_ranks(xa), _ranks(ya))


def _ranks(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(a, dtype=np.float64)
    ranks[order] = np.arange(1, a.size + 1, dtype=np.float64)
    # Average ties.
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for val_idx, c in enumerate(counts):
            if c > 1:
                mask = inv == val_idx
                ranks[mask] = float(np.mean(ranks[mask]))
    return ranks


def scatter_summary(points: list[dict]) -> dict:
    """Summarize residual vs Predict L2 points. No invented cells."""
    usable = [
        p
        for p in points
        if p.get("rel_l2") is not None and p.get("residual") is not None
    ]
    l2 = [float(p["rel_l2"]) for p in usable]
    res = [float(p["residual"]) for p in usable]
    trusts = [p.get("trust") for p in usable]
    return {
        "status": "ok" if usable else "empty",
        "n": len(usable),
        "pearson_residual_vs_l2": pearson_corr(res, l2),
        "spearman_residual_vs_l2": spearman_corr(res, l2),
        "rel_l2_median": float(np.median(l2)) if l2 else None,
        "residual_median": float(np.median(res)) if res else None,
        "rel_l2_min": float(np.min(l2)) if l2 else None,
        "rel_l2_max": float(np.max(l2)) if l2 else None,
        "residual_min": float(np.min(res)) if res else None,
        "residual_max": float(np.max(res)) if res else None,
        "n_untrusted": int(sum(1 for t in trusts if t == "untrusted")),
        "n_ood": int(sum(1 for t in trusts if t == "ood")),
        "points": usable,
        "note": (
            "Each point is a measured (Predict rel-L2, residual rel-L2) pair "
            "from IID baselines or OOD probes. Correlation is descriptive, "
            "not a causal claim."
        ),
    }


def collect_scatter_from_report(report: Mapping) -> list[dict]:
    """Extract measured residual vs L2 points from an eval report."""
    points: list[dict] = []
    baselines = report.get("baselines") or {}
    for model in ("fno", "pino"):
        sec = baselines.get(model) or {}
        for pde in ("burgers", "heat2d"):
            sub = sec.get(pde) or {}
            if sub.get("status") != "ok":
                continue
            pred = sub.get("predict") or {}
            cons = sub.get("conserve") or {}
            if "rel_l2_mean" not in pred or "residual_rel_l2" not in cons:
                continue
            points.append(
                {
                    "model": model,
                    "pde": pde,
                    "split": "iid",
                    "probe": "iid",
                    "rel_l2": float(pred["rel_l2_mean"]),
                    "residual": float(cons["residual_rel_l2"]),
                    "trust": (sub.get("trust") or {}).get("status"),
                }
            )
    for row in (report.get("ood") or {}).get("probes") or []:
        meta = row.get("meta") or {}
        for model in ("fno", "pino"):
            sub = row.get(model) or {}
            if sub.get("status") != "ok":
                continue
            pred = sub.get("predict") or {}
            cons = sub.get("conserve") or {}
            if "rel_l2_mean" not in pred or "residual_rel_l2" not in cons:
                continue
            points.append(
                {
                    "model": model,
                    "pde": meta.get("pde") or "unknown",
                    "split": "ood",
                    "probe": meta.get("id"),
                    "rel_l2": float(pred["rel_l2_mean"]),
                    "residual": float(cons["residual_rel_l2"]),
                    "trust": (sub.get("trust") or {}).get("status"),
                }
            )
    return points


def horizon_residual_split(
    u: np.ndarray,
    *,
    dx: float,
    dt: float,
    param: float | np.ndarray,
    pde: str,
) -> dict:
    """Early vs late half residual of one trajectory or a batch.

    This is a time-window diagnostic on the *existing* horizon (FNO time is
    locked as output channels — not a longer-horizon transfer claim).
    """
    u = np.asarray(u, dtype=np.float64)
    if pde == "burgers":
        if u.ndim == 2:
            u = u[None, ...]
        nt = u.shape[1] - 1
        mid = max(1, nt // 2)
        early = burgers_residual_l2(u[:, : mid + 1], dx, dt, param)
        late = burgers_residual_l2(u[:, mid:], dx, dt, param)
        full = burgers_residual_l2(u, dx, dt, param)
    elif pde == "heat2d":
        if u.ndim == 3:
            u = u[None, ...]
        nt = u.shape[1] - 1
        mid = max(1, nt // 2)
        early = heat_residual_l2(u[:, : mid + 1], dx, dt, param)
        late = heat_residual_l2(u[:, mid:], dx, dt, param)
        full = heat_residual_l2(u, dx, dt, param)
    else:
        raise ValueError(f"unknown pde {pde!r}")
    return {
        "status": "ok",
        "pde": pde,
        "n_time_steps": int(nt),
        "split_step": int(mid),
        "residual_early": float(early),
        "residual_late": float(late),
        "residual_full": float(full),
        "late_minus_early": float(late - early),
        "note": (
            "Early/late split of the stored horizon. Not a claim of temporal "
            "resolution transfer."
        ),
    }


def burgers_cf_sensitivity(
    u0: np.ndarray,
    nu: float,
    *,
    scales: Iterable[float] = (0.5, 1.0, 1.5, 2.0, 3.0),
    nx: int,
    nt: int,
    L: float,
    T: float,
    predict_fn: Callable[[np.ndarray, float], np.ndarray] | None = None,
    model_name: str = "classical",
) -> dict:
    """Sweep ν′ = scale * ν from a fixed IC. Classical always; model optional."""
    rows = []
    for scale in scales:
        nu_cf = float(nu) * float(scale)
        classical = burgers_counterfactual(
            u0, nu_orig=float(nu), nu_cf=nu_cf, nx=nx, nt=nt, L=L, T=T
        )
        row = {
            "scale": float(scale),
            "nu_orig": float(nu),
            "nu_cf": nu_cf,
            "classical_traj_delta": classical["rel_l2_traj_delta"],
            "classical_final_rel_l2": classical["final_rel_l2"],
        }
        if predict_fn is not None and abs(float(scale) - 1.0) > 1e-15:
            pred = predict_fn(u0, nu_cf)
            graded = grade_burgers_cf_prediction(
                pred, u0=u0, nu_orig=float(nu), nu_cf=nu_cf, nx=nx, nt=nt, L=L, T=T
            )
            pred_base = predict_fn(u0, float(nu))
            row["model_vs_classical_cf"] = graded["rel_l2_vs_classical_cf"]
            row["model_traj_delta"] = float(
                relative_l2(pred, pred_base)
            )
        elif predict_fn is not None:
            row["model_vs_classical_cf"] = 0.0
            row["model_traj_delta"] = 0.0
        rows.append(row)
    return {
        "status": "ok",
        "pde": "burgers",
        "model": model_name,
        "learned": predict_fn is not None,
        "n_scales": len(rows),
        "rows": rows,
        "note": (
            "Counterfactual sensitivity: trajectory Δ vs coefficient scale. "
            "Learned columns appear only when a predict_fn was supplied."
        ),
    }


def heat_cf_sensitivity(
    u0: np.ndarray,
    alpha: float,
    *,
    scales: Iterable[float] = (0.5, 1.0, 1.5, 2.0),
    n: int,
    nt: int,
    L: float,
    T: float,
    predict_fn: Callable[[np.ndarray, float], np.ndarray] | None = None,
    model_name: str = "classical",
) -> dict:
    """Sweep α′ = scale * α from a fixed IC."""
    rows = []
    for scale in scales:
        a_cf = float(alpha) * float(scale)
        classical = heat_counterfactual(
            u0, alpha_orig=float(alpha), alpha_cf=a_cf, n=n, nt=nt, L=L, T=T
        )
        row = {
            "scale": float(scale),
            "alpha_orig": float(alpha),
            "alpha_cf": a_cf,
            "classical_traj_delta": classical["rel_l2_traj_delta"],
            "classical_final_rel_l2": classical["final_rel_l2"],
        }
        if predict_fn is not None and abs(float(scale) - 1.0) > 1e-15:
            pred = predict_fn(u0, a_cf)
            graded = grade_heat_cf_prediction(
                pred,
                u0=u0,
                alpha_orig=float(alpha),
                alpha_cf=a_cf,
                n=n,
                nt=nt,
                L=L,
                T=T,
            )
            pred_base = predict_fn(u0, float(alpha))
            row["model_vs_classical_cf"] = graded["rel_l2_vs_classical_cf"]
            row["model_traj_delta"] = float(relative_l2(pred, pred_base))
        elif predict_fn is not None:
            row["model_vs_classical_cf"] = 0.0
            row["model_traj_delta"] = 0.0
        rows.append(row)
    return {
        "status": "ok",
        "pde": "heat2d",
        "model": model_name,
        "learned": predict_fn is not None,
        "n_scales": len(rows),
        "rows": rows,
        "note": (
            "Counterfactual sensitivity: trajectory Δ vs diffusivity scale. "
            "Learned columns appear only when a predict_fn was supplied."
        ),
    }


def run_label_diagnostics(
    burgers: Mapping | None = None,
    heat: Mapping | None = None,
) -> dict:
    """Classical-only diagnostics from stored labels (CPU-friendly)."""
    out: dict = {"status": "ok", "burgers": None, "heat2d": None}
    if burgers is not None:
        u = np.asarray(burgers["u"])
        x = np.asarray(burgers["x"])
        t = np.asarray(burgers["t"])
        nu = np.asarray(burgers["nu"])
        dx = float(x[1] - x[0])
        dt = float(t[1] - t[0])
        L = float(x[-1] + (x[1] - x[0]))
        T = float(t[-1])
        i = 0
        out["burgers"] = {
            "horizon_residual": horizon_residual_split(
                u[i], dx=dx, dt=dt, param=float(nu[i]), pde="burgers"
            ),
            "cf_sensitivity": burgers_cf_sensitivity(
                u[i, 0],
                float(nu[i]),
                nx=len(x),
                nt=len(t) - 1,
                L=L,
                T=T,
                model_name="classical_fd",
            ),
        }
    if heat is not None:
        u = np.asarray(heat["u"])
        x = np.asarray(heat["x"])
        t = np.asarray(heat["t"])
        alpha = np.asarray(heat["alpha"])
        dx = float(x[1] - x[0])
        dt = float(t[1] - t[0])
        L = float(x[-1])
        T = float(t[-1])
        i = 0
        out["heat2d"] = {
            "horizon_residual": horizon_residual_split(
                u[i], dx=dx, dt=dt, param=float(alpha[i]), pde="heat2d"
            ),
            "cf_sensitivity": heat_cf_sensitivity(
                u[i, 0],
                float(alpha[i]),
                n=len(x),
                nt=len(t) - 1,
                L=L,
                T=T,
                model_name="classical_fd",
            ),
        }
    if out["burgers"] is None and out["heat2d"] is None:
        out["status"] = "skipped"
        out["reason"] = "labels_missing"
    return out


def build_diagnostics(report: Mapping, labels: Mapping | None = None) -> dict:
    """Assemble the v0.2 diagnostics block from a report (+ optional labels)."""
    scatter = scatter_summary(collect_scatter_from_report(report))
    label_diag = {"status": "skipped", "reason": "labels_not_provided"}
    if labels:
        label_diag = run_label_diagnostics(
            burgers=labels.get("burgers"),
            heat=labels.get("heat"),
        )
    return {
        "description": (
            "v0.2 diagnostics. Residual-vs-L2 scatter re-uses measured IID + "
            "OOD cells. Horizon split and CF sensitivity are fresh classical "
            "solves on stored labels. No fabricated SOTA."
        ),
        "residual_vs_l2": scatter,
        "labels": label_diag,
    }
