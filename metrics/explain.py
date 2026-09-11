"""Explainability stub: keyword rubric over free-text rationales."""
from __future__ import annotations

from typing import Iterable

# Rubric keywords grouped by exam facet (Week-1 stub, not a trained judge).
RUBRIC: dict[str, list[str]] = {
    "pde_terms": [
        "viscosity",
        "diffusion",
        "advection",
        "laplacian",
        "nonlinear",
        "burgers",
        "heat",
        "residual",
    ],
    "numerics": [
        "stability",
        "cfl",
        "discretization",
        "finite difference",
        "boundary",
        "periodic",
        "dirichlet",
    ],
    "physics": [
        "energy",
        "conservation",
        "dissipation",
        "shock",
        "smooth",
        "decay",
        "transport",
    ],
}


def score_explanation(text: str, rubric: dict[str, list[str]] | None = None) -> dict:
    """Keyword hit-rate rubric. Returns per-facet coverage and overall score in [0,1]."""
    rubric = rubric or RUBRIC
    lowered = (text or "").lower()
    facets: dict[str, dict] = {}
    scores = []
    for facet, kws in rubric.items():
        hits = [kw for kw in kws if kw in lowered]
        cov = len(hits) / max(len(kws), 1)
        facets[facet] = {"hits": hits, "coverage": cov, "n_keywords": len(kws)}
        scores.append(cov)
    overall = float(sum(scores) / max(len(scores), 1))
    return {"facets": facets, "overall": overall, "status": "stub_keyword_rubric"}


def batch_score(texts: Iterable[str]) -> dict:
    results = [score_explanation(t) for t in texts]
    overalls = [r["overall"] for r in results]
    return {
        "per_item": results,
        "mean_overall": float(sum(overalls) / max(len(overalls), 1)),
        "status": "stub_keyword_rubric",
    }
