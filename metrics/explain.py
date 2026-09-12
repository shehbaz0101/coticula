"""Exam 4: fixed explanation item set + keyword rubric.

v0 grades a rationale against **expected law keywords** on a pinned item set
(``datasets/exam4/items_v0.json``). The optional LLM judge is env-gated and
stays ``not_trained`` until a real judge is wired — never invent LLM scores.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
EXAM4_PATH = ROOT / "datasets" / "exam4" / "items_v0.json"

# Legacy facet lists (Week-1 stub). Exam 4 items carry their own expected keywords.
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

LLM_JUDGE_ENV = "VU_LLM_JUDGE"
LLM_KEY_ENVS = ("VU_LLM_API_KEY", "OPENAI_API_KEY")


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


def load_exam4(path: Path | None = None) -> dict:
    """Load the pinned Exam 4 item set."""
    path = path or EXAM4_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items") or []
    if len(items) < 15:
        raise ValueError(f"Exam 4 item set too small ({len(items)}); need ≥ 15")
    ids = [it["id"] for it in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Exam 4 item ids must be unique")
    for it in items:
        kws = it.get("expected_keywords") or []
        if not kws:
            raise ValueError(f"{it.get('id')}: expected_keywords required")
        if not it.get("prompt"):
            raise ValueError(f"{it.get('id')}: prompt required")
    return data


def score_item(item: Mapping, text: str) -> dict:
    """Coverage of ``item['expected_keywords']`` in ``text`` (substring, casefold)."""
    lowered = (text or "").casefold()
    expected = [str(kw) for kw in (item.get("expected_keywords") or [])]
    hits = [kw for kw in expected if kw.casefold() in lowered]
    missing = [kw for kw in expected if kw.casefold() not in lowered]
    n = max(len(expected), 1)
    coverage = float(len(hits) / n)
    return {
        "id": item.get("id"),
        "pde": item.get("pde"),
        "facet": item.get("facet"),
        "n_expected": len(expected),
        "hits": hits,
        "missing": missing,
        "coverage": coverage,
        "status": "keyword_rubric",
    }


def grade_exam4(
    answers: Mapping[str, str] | None = None,
    *,
    use_gold: bool = False,
    path: Path | None = None,
) -> dict:
    """Grade the fixed Exam 4 set.

    - ``use_gold=True`` scores authored gold rationales (reference ceiling).
    - ``answers`` maps item id → rationale. Missing ids score 0 coverage.
    - Never calls an LLM. See ``llm_judge_exam4``.
    """
    payload = load_exam4(path)
    items = payload["items"]
    answers = dict(answers or {})
    per_item = []
    source = "gold" if use_gold else ("answers" if answers else "empty")
    for it in items:
        if use_gold:
            text = str(it.get("gold_rationale") or "")
        else:
            text = answers.get(it["id"], "")
        row = score_item(it, text)
        row["answer_source"] = "gold" if use_gold else ("provided" if it["id"] in answers else "missing")
        per_item.append(row)

    coverages = [r["coverage"] for r in per_item]
    mean = float(sum(coverages) / max(len(coverages), 1))
    facets: dict[str, list[float]] = {}
    pdes: dict[str, list[float]] = {}
    for r in per_item:
        facets.setdefault(str(r.get("facet") or "other"), []).append(r["coverage"])
        pdes.setdefault(str(r.get("pde") or "other"), []).append(r["coverage"])

    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / max(len(xs), 1))

    from datasets.pins import sha256_file

    item_path = path or EXAM4_PATH
    try:
        item_set = str(item_path.relative_to(ROOT))
    except ValueError:
        item_set = str(item_path)
    return {
        "status": "keyword_rubric",
        "n_items": len(items),
        "item_set": item_set,
        "item_set_sha256": sha256_file(item_path) if item_path.exists() else None,
        "answer_source": source,
        "mean_coverage": mean,
        "n_perfect": int(sum(1 for c in coverages if c >= 1.0 - 1e-12)),
        "n_zero": int(sum(1 for c in coverages if c <= 1e-12)),
        "by_facet": {k: {"n": len(v), "mean_coverage": _mean(v)} for k, v in sorted(facets.items())},
        "by_pde": {k: {"n": len(v), "mean_coverage": _mean(v)} for k, v in sorted(pdes.items())},
        "per_item": per_item,
        "note": (
            "Keyword coverage vs expected law keywords. Not a trained judge "
            "and not an LLM score."
        ),
        "llm_judge": {"status": "not_trained", "note": "see llm_judge_exam4"},
    }


def llm_judge_exam4(*_args, **_kwargs) -> dict:
    """Optional LLM judge hook — stubbed and env-gated. Never invents a score.

    A real judge would require ``VU_LLM_JUDGE=1`` *and* an implemented client.
    Until that exists, this always returns ``not_trained`` with no numeric score.
    """
    enabled = os.environ.get(LLM_JUDGE_ENV, "").strip() in {"1", "true", "yes"}
    has_key = any(os.environ.get(k) for k in LLM_KEY_ENVS)
    if enabled and has_key:
        return {
            "status": "not_trained",
            "score": None,
            "n_items": None,
            "note": (
                "VU_LLM_JUDGE is set and an API key is present, but no LLM "
                "judge is wired. Refusing to fabricate scores."
            ),
            "env": LLM_JUDGE_ENV,
        }
    if enabled:
        return {
            "status": "not_trained",
            "score": None,
            "note": (
                f"{LLM_JUDGE_ENV} is set but no API key "
                f"({', '.join(LLM_KEY_ENVS)}); keyword rubric only. "
                "No fabricated LLM scores."
            ),
        }
    if has_key:
        return {
            "status": "not_trained",
            "score": None,
            "note": (
                "API key present but LLM judge is not wired "
                f"(set {LLM_JUDGE_ENV}=1 only after a real judge exists). "
                "No fabricated LLM scores."
            ),
        }
    return {
        "status": "not_trained",
        "score": None,
        "note": (
            f"LLM judge off (no {LLM_JUDGE_ENV} / API key). "
            "Keyword rubric only. No fabricated LLM scores."
        ),
    }
