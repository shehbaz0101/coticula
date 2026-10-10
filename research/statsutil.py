"""Paired permutation tests, bootstrap intervals, and the 0–100 score.

The zero field is an exact solution of both homogeneous PDEs. Residual scores
exclude it from the trivial-baseline anchor. If the best remaining baseline is
no worse than the reference, the scaled score is undefined.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import kendalltau, linregress


def scaled_score(value: float, baseline: float, reference: float) -> float | None:
    """S = 100 * (M_0 - M) / (M_0 - M_ref). Not clipped.

    Returns None when the denominator is non-positive, i.e. when the best
    trivial baseline does no worse than the reference.
    """
    if not all(math.isfinite(v) for v in (value, baseline, reference)):
        return None
    denom = baseline - reference
    if denom <= 0.0:
        return None
    return float(100.0 * (baseline - value) / denom)


def best_baseline(scores: dict[str, float], *, exclude: tuple[str, ...] = ()) -> tuple[str, float] | None:
    """Lowest score among the trivial baselines, after exclusions."""
    items = [(name, float(val)) for name, val in scores.items() if name not in exclude and math.isfinite(val)]
    if not items:
        return None
    name, val = min(items, key=lambda kv: kv[1])
    return name, val


def paired_permutation(diff: np.ndarray, *, n_perm: int, seed: int) -> dict:
    """Two-sided sign-flip test on the mean of paired differences."""
    diff = np.asarray(diff, dtype=np.float64)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return {"mean_diff": None, "p_value": None, "n": 0}
    observed = float(diff.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(int(n_perm), diff.size))
    null = (signs * diff[None, :]).mean(axis=1)
    p_value = float((np.sum(np.abs(null) >= abs(observed) - 1e-15) + 1) / (n_perm + 1))
    return {"mean_diff": observed, "p_value": p_value, "n": int(diff.size)}


def bootstrap_mean_ci(values: np.ndarray, *, n_boot: int, seed: int, alpha: float = 0.05) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(int(n_boot), values.size))
    means = values[draws].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "mean": float(values.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "n": int(values.size),
    }


def kendall(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return {"tau": None, "p_value": None, "n": int(mask.sum())}
    tau, p_value = kendalltau(x[mask], y[mask])
    return {"tau": float(tau), "p_value": float(p_value), "n": int(mask.sum())}


def linear_fit(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return {"slope": None, "intercept": None, "r_squared": None, "n": int(mask.sum())}
    slope, intercept, r_value, _p, _se = linregress(x[mask], y[mask])
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_value**2),
        "n": int(mask.sum()),
    }


def paired_wilcoxon(diff: np.ndarray) -> dict:
    """Two-sided Wilcoxon signed-rank test on paired differences.

    Effect size is the mean difference and Cohen's dz (mean / sample std).
    """
    from scipy.stats import wilcoxon

    diff = np.asarray(diff, dtype=np.float64)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return {
            "test": "wilcoxon_signed_rank",
            "mean_diff": None,
            "std_diff": None,
            "cohen_dz": None,
            "statistic": None,
            "p_value": None,
            "n": 0,
        }
    mean = float(diff.mean())
    std = float(diff.std(ddof=1)) if diff.size > 1 else 0.0
    dz = float(mean / std) if std > 0 else None
    usable = diff[np.abs(diff) > 0]
    if usable.size < 5:
        return {
            "test": "wilcoxon_signed_rank",
            "mean_diff": mean,
            "std_diff": std,
            "cohen_dz": dz,
            "statistic": None,
            "p_value": None,
            "n": int(diff.size),
            "n_nonzero": int(usable.size),
        }
    result = wilcoxon(usable, alternative="two-sided", zero_method="wilcox")
    return {
        "test": "wilcoxon_signed_rank",
        "mean_diff": mean,
        "std_diff": std,
        "cohen_dz": dz,
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n": int(diff.size),
        "n_nonzero": int(usable.size),
    }


def log_interp(levels: np.ndarray, values: np.ndarray, epsilon: float) -> dict:
    """Log-log linear interpolation. Clips to the covered range and flags it."""
    levels = np.asarray(levels, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(levels)
    levels = levels[order]
    values = values[order]
    clipped = min(max(float(epsilon), float(levels[0])), float(levels[-1]))
    y = np.log(np.maximum(values, 1e-30))
    pred = float(np.exp(np.interp(np.log(clipped), np.log(levels), y)))
    return {
        "value": pred,
        "epsilon_used": clipped,
        "extrapolated": bool(abs(clipped - float(epsilon)) > 1e-15),
    }
