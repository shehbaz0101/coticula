"""Fail-closed trust flags for Vermithor VU-Bench.

A case is never treated as a silent success heatmap:

- ``ood`` — the query leaves declared train support (param, resolution, IC family)
- ``untrusted`` — residual or predict error exceeds documented thresholds

Severity is fail-closed: residual/L2 failure wins over a clean OOD label, so a
pretty field that violates the PDE is ``untrusted`` even if it is also OOD.
"""
from __future__ import annotations

from typing import Iterable

# Declared v0 train support (scripts.generate_labels / dataset meta).
BURGERS_TRAIN_SUPPORT: dict = {
    "pde": "burgers1d",
    "param": "nu",
    "param_range": (0.005, 0.05),
    "resolution": 64,
    "resolution_key": "nx",
    "nt": 80,
    "L": 2 * 3.141592653589793,
    "T": 1.0,
    "ic_family": "fourier_modes",
}

HEAT_TRAIN_SUPPORT: dict = {
    "pde": "heat2d",
    "param": "alpha",
    "param_range": (0.05, 0.2),
    "resolution": 32,
    "resolution_key": "n",
    "nt": 40,
    "L": 1.0,
    "T": 0.5,
    "ic_family": "gaussian_bump",
}

# Documented thresholds — not tuned after seeing scores.
# residual_rel_l2 > 1 means the FD residual is field-scale (Exam 2).
RESIDUAL_UNTRUSTED = 1.0
# predict rel-L2 above one-half of the truth norm.
PREDICT_UNTRUSTED = 0.50

TRUST_POLICY: dict = {
    "name": "vu-bench-v0-fail-closed",
    "residual_untrusted": RESIDUAL_UNTRUSTED,
    "predict_untrusted": PREDICT_UNTRUSTED,
    "statuses": ["trusted", "ood", "untrusted", "reference"],
    "rule": (
        "If residual_rel_l2 > residual_untrusted or predict rel-L2 > "
        "predict_untrusted → untrusted. Else if the case leaves train support "
        "→ ood. Else → trusted. Classical labeler rows use status=reference."
    ),
    "heatmap_rule": (
        "Do not present an OOD or untrusted field as a silent pretty heatmap. "
        "Any plot must carry the trust banner from refuse_silent_heatmap()."
    ),
}


def param_in_range(value: float, lo: float, hi: float) -> bool:
    """Inclusive train-range check used by OOD param-shift probes."""
    return float(lo) <= float(value) <= float(hi)


def support_violations(
    *,
    param: float | None = None,
    param_range: tuple[float, float] | None = None,
    resolution: int | None = None,
    train_resolution: int | None = None,
    ic_family: str | None = None,
    train_ic_family: str | None = None,
) -> list[str]:
    """Return machine-readable reasons the case is outside train support."""
    reasons: list[str] = []
    if param is not None and param_range is not None:
        lo, hi = param_range
        if not param_in_range(param, lo, hi):
            reasons.append("param_shift")
    if (
        resolution is not None
        and train_resolution is not None
        and int(resolution) != int(train_resolution)
    ):
        reasons.append("resolution_shift")
    if (
        ic_family is not None
        and train_ic_family is not None
        and str(ic_family) != str(train_ic_family)
    ):
        reasons.append("ic_family_shift")
    return reasons


def decide_trust(
    *,
    param: float | None = None,
    param_range: tuple[float, float] | None = None,
    residual_rel_l2: float | None = None,
    predict_rel_l2: float | None = None,
    resolution: int | None = None,
    train_resolution: int | None = None,
    ic_family: str | None = None,
    train_ic_family: str | None = None,
    residual_threshold: float = RESIDUAL_UNTRUSTED,
    predict_threshold: float = PREDICT_UNTRUSTED,
    role: str = "surrogate",
) -> dict:
    """Fail-closed trust decision for one (model, case) pair.

    ``role="labeler"`` is the classical FD generator: the *case* may still be
    OOD relative to the train corpus, but the solver is not a transfer claim.
    """
    ood_reasons = support_violations(
        param=param,
        param_range=param_range,
        resolution=resolution,
        train_resolution=train_resolution,
        ic_family=ic_family,
        train_ic_family=train_ic_family,
    )
    ood = bool(ood_reasons)
    quality_reasons: list[str] = []
    if residual_rel_l2 is not None and float(residual_rel_l2) > float(
        residual_threshold
    ):
        quality_reasons.append("residual_too_high")
    if predict_rel_l2 is not None and float(predict_rel_l2) > float(predict_threshold):
        quality_reasons.append("predict_l2_too_high")
    untrusted = bool(quality_reasons)

    if role == "labeler":
        status = "reference"
    elif untrusted:
        status = "untrusted"
    elif ood:
        status = "ood"
    else:
        status = "trusted"

    reasons = list(ood_reasons) + quality_reasons
    return {
        "status": status,
        "ood": ood,
        "untrusted": untrusted,
        "reasons": reasons,
        "role": role,
        "thresholds": {
            "residual_untrusted": float(residual_threshold),
            "predict_untrusted": float(predict_threshold),
        },
        "inputs": {
            "param": None if param is None else float(param),
            "param_range": None if param_range is None else [float(param_range[0]), float(param_range[1])],
            "residual_rel_l2": None if residual_rel_l2 is None else float(residual_rel_l2),
            "predict_rel_l2": None if predict_rel_l2 is None else float(predict_rel_l2),
            "resolution": None if resolution is None else int(resolution),
            "train_resolution": None if train_resolution is None else int(train_resolution),
            "ic_family": ic_family,
            "train_ic_family": train_ic_family,
        },
        "banner": refuse_silent_heatmap(
            {"status": status, "reasons": reasons, "ood": ood, "untrusted": untrusted}
        ),
    }


def refuse_silent_heatmap(decision: dict) -> str:
    """Caption that must accompany any field plot. Fail-closed."""
    status = decision.get("status", "untrusted")
    reasons = ", ".join(decision.get("reasons") or []) or "none"
    if status == "trusted":
        return "trusted: in-support and residual/L2 below thresholds"
    if status == "reference":
        suffix = f" ({reasons})" if decision.get("reasons") else ""
        return (
            f"reference labeler{suffix}: classical FD generated this case; "
            "not a learned-transfer claim"
        )
    if status == "ood":
        return (
            f"OOD ({reasons}): outside declared train support — "
            "do not present as an IID success heatmap"
        )
    return (
        f"UNTRUSTED ({reasons}): residual or predict error exceeds fail-closed "
        "threshold — do not present a silent pretty heatmap"
    )


def attach_trust(
    metrics: dict,
    *,
    support: dict,
    param: float | None,
    resolution: int | None,
    ic_family: str | None,
    role: str = "surrogate",
) -> dict:
    """Copy ``metrics`` and add a ``trust`` block from predict/conserve fields."""
    out = dict(metrics)
    pred = (metrics.get("predict") or {}).get("rel_l2_mean")
    residual = (metrics.get("conserve") or {}).get("residual_rel_l2")
    pr = support.get("param_range")
    param_range = tuple(pr) if pr is not None else None
    out["trust"] = decide_trust(
        param=param,
        param_range=param_range,
        residual_rel_l2=residual,
        predict_rel_l2=pred,
        resolution=resolution,
        train_resolution=support.get("resolution"),
        ic_family=ic_family,
        train_ic_family=support.get("ic_family"),
        role=role,
    )
    return out


def summarize_trust_flags(rows: Iterable[dict]) -> dict:
    """Count trust statuses across probe rows (for the report rollup)."""
    counts = {"trusted": 0, "ood": 0, "untrusted": 0, "reference": 0, "other": 0}
    n = 0
    for row in rows:
        trust = row.get("trust") if isinstance(row, dict) else None
        if not isinstance(trust, dict):
            continue
        n += 1
        st = trust.get("status", "other")
        if st in counts:
            counts[st] += 1
        else:
            counts["other"] += 1
    return {"n_scored": n, "counts": counts}
