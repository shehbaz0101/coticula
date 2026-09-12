"""Exam 4: fixed explanation item set + keyword rubric.

v0.2 grades a rationale against **expected law keywords** on a pinned item set
(``datasets/exam4/items_v0.json``, 48 items). Optional structured baselines:

- gold reference (authored rationales; rubric ceiling)
- rule-based templates + measured Exam 1–3 observations (not an LLM)
- metric-dump (numbers only; shows the rubric is not a free lunch)

The optional LLM judge is env-gated (``COTICULA_LLM_JUDGE`` / ``VU_LLM_JUDGE``)
and stays ``not_trained`` until a real judge is wired — never invent LLM scores.
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

# Primary name plus the v0.1 alias so existing env files keep working.
LLM_JUDGE_ENV = "COTICULA_LLM_JUDGE"
LLM_JUDGE_ENV_ALIASES = ("COTICULA_LLM_JUDGE", "VU_LLM_JUDGE")
LLM_KEY_ENVS = ("COTICULA_LLM_API_KEY", "VU_LLM_API_KEY", "OPENAI_API_KEY")

# Facet-level templates (not per-item). Item-specific keywords such as
# "cfl", "sinusoid", or "touchstone" are intentionally omitted so the
# rule-based score is a measured ceiling of *structured* knowledge, not gold.
_RULE_TEMPLATES: dict[tuple[str, str], str] = {
    ("burgers1d", "pde_terms"): (
        "Viscous Burgers is nonlinear advection balanced by viscosity-driven "
        "diffusion. A large residual means the predicted Burgers field fails "
        "the discrete PDE discretization."
    ),
    ("burgers1d", "physics"): (
        "Larger viscosity strengthens diffusion and can smooth a forming shock. "
        "Periodic viscous Burgers has no kinetic-energy conservation: viscosity "
        "produces dissipation so energy decays."
    ),
    ("burgers1d", "numerics"): (
        "The classical Burgers solver is periodic in x. An explicit finite "
        "difference step must respect a stability bound on dt relative to dx."
    ),
    ("burgers1d", "counterfactual"): (
        "A counterfactual that scales viscosity increases diffusion, so the "
        "trajectory must differ from the original solve."
    ),
    ("burgers1d", "ood"): (
        "Train ICs are low Fourier modes on a periodic interval. A localized "
        "pulse is an OOD IC-family shift, not an IID Exam 1 draw."
    ),
    ("burgers1d", "trust"): (
        "Predict L2 can look moderate while the residual is field-scale, so "
        "the fail-closed flag is untrusted. A trusted flag is not a claim "
        "that the model understands the PDE."
    ),
    ("heat2d", "pde_terms"): (
        "The heat equation is diffusion through the Laplacian. Exam 2 forms "
        "the residual u_t minus alpha times the discrete Laplacian."
    ),
    ("heat2d", "physics"): (
        "Larger diffusivity speeds diffusion, so peaks decay faster and the "
        "field becomes smooth. Dirichlet heat produces dissipation; energy "
        "decays rather than growing."
    ),
    ("heat2d", "numerics"): (
        "Heat is solved on a square with Dirichlet walls fixed at zero on "
        "every boundary."
    ),
    ("heat2d", "counterfactual"): (
        "A counterfactual that scales alpha must change the trajectory: "
        "stronger diffusion, not a copy of the original field."
    ),
    ("heat2d", "ood"): (
        "Train ICs are a single gaussian bump. Two-bump or sinusoid fields "
        "are IC-family OOD probes. A resolution probe changes the spatial grid."
    ),
    ("harness", "exams"): (
        "Exam 1 scores relative L2 and NMSE against classical labels. "
        "Exam 2 audits the discrete PDE residual and energy drift. "
        "Exam 3 changes a coefficient and grades against the classical solve. "
        "Missing checkpoints stay not_trained; reports do not invent SOTA."
    ),
    ("harness", "trust"): (
        "Untrusted means residual rel-L2 exceeds 1 or Predict rel-L2 exceeds "
        "0.5. OOD means the query left train support while residual and L2 "
        "still sit below those cuts. Trusted requires both. OOD or untrusted "
        "fields must not be shown as a silent pretty heatmap."
    ),
    ("harness", "ood"): (
        "OOD transfer lives in a separate report section and must not be "
        "mixed into IID Exam 1 tables. FNO spectral convolution is "
        "resolution-agnostic in space; time is locked as output channels."
    ),
    ("harness", "explain"): (
        "Exam 4 is a keyword rubric over expected law terms, not a trained "
        "explanation judge and not an LLM score. Rule-based answers are "
        "structured templates plus measured observations."
    ),
    ("harness", "pde_terms"): (
        "PINO adds a residual PDE loss on top of data MSE; it is not a "
        "claim of SOTA."
    ),
    ("harness", "counterfactual"): (
        "A counterfactual sensitivity sweep records how the trajectory "
        "changes as a coefficient is scaled, and grades the model against "
        "the classical solve."
    ),
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
    answer_source: str | None = None,
    status: str | None = None,
    note: str | None = None,
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
    source = answer_source or ("gold" if use_gold else ("answers" if answers else "empty"))
    for it in items:
        if use_gold:
            text = str(it.get("gold_rationale") or "")
        else:
            text = answers.get(it["id"], "")
        row = score_item(it, text)
        if use_gold:
            row["answer_source"] = "gold"
        else:
            row["answer_source"] = "provided" if it["id"] in answers else "missing"
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
        "status": status or "keyword_rubric",
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
        "note": note
        or (
            "Keyword coverage vs expected law keywords. Not a trained judge "
            "and not an LLM score."
        ),
        "llm_judge": {"status": "not_trained", "note": "see llm_judge_exam4"},
    }


def _fmt(value) -> str:
    if value is None:
        return "not_trained"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def observations_from_report(report: Mapping | None) -> dict:
    """Pull measured Exam 1–3 / OOD facts used by the rule-based explainer.

    Only copies numbers that already exist in ``report``. Missing cells stay
    ``None`` / ``not_trained`` — nothing is invented.
    """
    report = dict(report or {})
    exams = report.get("exams") or {}
    pred = exams.get("predict") or {}
    cons = exams.get("conserve") or {}
    cf = exams.get("counterfactual") or {}
    baselines = report.get("baselines") or {}

    def _l2(block: Mapping, pde: str):
        sub = (block or {}).get(pde) or {}
        if sub.get("status") == "ok" and "rel_l2_mean" in sub:
            return float(sub["rel_l2_mean"])
        return None

    def _res(block: Mapping, pde: str):
        sub = (block or {}).get(pde) or {}
        if sub.get("status") == "ok" and "residual_rel_l2" in sub:
            return float(sub["residual_rel_l2"])
        return None

    def _cf_vs(block: Mapping, pde: str):
        sub = (block or {}).get(pde) or {}
        if sub.get("status") == "ok" and "rel_l2_vs_classical_cf" in sub:
            return float(sub["rel_l2_vs_classical_cf"])
        return None

    def _trust(model: str, pde: str):
        sec = (baselines.get(model) or {}).get(pde) or {}
        return (sec.get("trust") or {}).get("status")

    ood = report.get("ood") or {}
    ood_n = (ood.get("trust_summary") or {}).get("n_scored")
    return {
        "source": "report",
        "fno_burgers_rel_l2": _l2(pred.get("fno"), "burgers"),
        "pino_burgers_rel_l2": _l2(pred.get("pino"), "burgers"),
        "fno_burgers_residual": _res(cons.get("fno"), "burgers"),
        "pino_burgers_residual": _res(cons.get("pino"), "burgers"),
        "fno_heat_rel_l2": _l2(pred.get("fno"), "heat2d"),
        "fno_heat_residual": _res(cons.get("fno"), "heat2d"),
        "fno_heat_cf": _cf_vs(cf.get("fno"), "heat2d"),
        "fno_burgers_trust": _trust("fno", "burgers"),
        "pino_burgers_trust": _trust("pino", "burgers"),
        "fno_heat_trust": _trust("fno", "heat2d"),
        "ood_n_scored": ood_n,
        "note": (
            "Copied from a measured report. Empty fields mean the checkpoint "
            "or section was absent — not a placeholder accuracy."
        ),
    }


def measured_snippet(obs: Mapping | None) -> str:
    """One short measured-facts sentence. Numbers only when present."""
    obs = dict(obs or {})
    bits = []
    fno_l2 = obs.get("fno_burgers_rel_l2")
    pino_l2 = obs.get("pino_burgers_rel_l2")
    fno_r = obs.get("fno_burgers_residual")
    pino_r = obs.get("pino_burgers_residual")
    if fno_l2 is not None and pino_l2 is not None and fno_r is not None and pino_r is not None:
        bits.append(
            f"Measured IID Burgers: FNO Predict L2={_fmt(fno_l2)} residual={_fmt(fno_r)}; "
            f"PINO L2={_fmt(pino_l2)} residual={_fmt(pino_r)} "
            f"(FNO trust={obs.get('fno_burgers_trust')}, PINO trust={obs.get('pino_burgers_trust')})."
        )
    heat_cf = obs.get("fno_heat_cf")
    if heat_cf is not None:
        bits.append(f"Measured FNO heat Exam 3 vs classical at alpha': {_fmt(heat_cf)}.")
    if not bits:
        return "No learned metrics were present in this report (not_trained)."
    return " ".join(bits)


def rule_based_answer(item: Mapping, obs: Mapping | None = None) -> str:
    """Structured rationale from a facet template plus measured observations."""
    key = (str(item.get("pde") or ""), str(item.get("facet") or ""))
    text = _RULE_TEMPLATES.get(key) or _RULE_TEMPLATES.get(("harness", key[1]))
    if text is None:
        text = (
            "Coticula grades understanding with Predict L2, residual audits, "
            "and counterfactual coefficient response. Missing models stay not_trained."
        )
    extra = measured_snippet(obs)
    if extra:
        text = f"{text} {extra}"
    return text


def metric_dump_answer(item: Mapping, obs: Mapping | None = None) -> str:
    """Numbers-only dump. Intentionally avoids law keywords."""
    obs = dict(obs or {})
    return (
        f"id={item.get('id')} "
        f"fno_b_l2={_fmt(obs.get('fno_burgers_rel_l2'))} "
        f"pino_b_l2={_fmt(obs.get('pino_burgers_rel_l2'))} "
        f"fno_b_res={_fmt(obs.get('fno_burgers_residual'))} "
        f"pino_b_res={_fmt(obs.get('pino_burgers_residual'))} "
        f"fno_h_cf={_fmt(obs.get('fno_heat_cf'))} "
        f"ood_n={_fmt(obs.get('ood_n_scored'))}"
    )


def build_exam4_answers(
    kind: str,
    obs: Mapping | None = None,
    path: Path | None = None,
) -> dict[str, str]:
    """Map item id → rationale for ``rule_based`` or ``metric_dump``."""
    items = load_exam4(path)["items"]
    if kind == "rule_based":
        return {it["id"]: rule_based_answer(it, obs) for it in items}
    if kind == "metric_dump":
        return {it["id"]: metric_dump_answer(it, obs) for it in items}
    raise ValueError(f"unknown answer kind {kind!r}")


def grade_rule_based_exam4(obs: Mapping | None = None, path: Path | None = None) -> dict:
    """Grade facet-level structured templates. Not gold. Not an LLM."""
    answers = build_exam4_answers("rule_based", obs, path=path)
    graded = grade_exam4(
        answers,
        path=path,
        answer_source="rule_based",
        status="keyword_rubric_rule_based",
        note=(
            "Structured rule-based templates plus measured Exam 1–3 observations. "
            "Facet-level (not per-item) wording, so coverage is below the gold "
            "ceiling. Not an LLM score."
        ),
    )
    graded["observations_used"] = {
        k: obs.get(k) if obs else None
        for k in (
            "fno_burgers_rel_l2",
            "pino_burgers_rel_l2",
            "fno_burgers_residual",
            "pino_burgers_residual",
            "fno_heat_cf",
        )
    }
    return graded


def grade_metric_dump_exam4(obs: Mapping | None = None, path: Path | None = None) -> dict:
    """Grade a numbers-only dump. Expected to miss most law keywords."""
    answers = build_exam4_answers("metric_dump", obs, path=path)
    return grade_exam4(
        answers,
        path=path,
        answer_source="metric_dump",
        status="keyword_rubric_metric_dump",
        note=(
            "Numbers-only dump of measured report fields. Included to show the "
            "keyword rubric is not a free lunch. Not an LLM score."
        ),
    )


def _llm_flag_enabled() -> tuple[bool, str | None]:
    for name in LLM_JUDGE_ENV_ALIASES:
        if os.environ.get(name, "").strip() in {"1", "true", "yes"}:
            return True, name
    return False, None


def llm_judge_exam4(*_args, **_kwargs) -> dict:
    """Optional LLM judge hook — stubbed and env-gated. Never invents a score.

    A real judge would require ``COTICULA_LLM_JUDGE=1`` (or the v0.1 alias
    ``VU_LLM_JUDGE=1``) *and* an implemented client. Until that exists, this
    always returns ``not_trained`` with no numeric score.
    """
    enabled, env_used = _llm_flag_enabled()
    has_key = any(os.environ.get(k) for k in LLM_KEY_ENVS)
    names = " / ".join(LLM_JUDGE_ENV_ALIASES)
    if enabled and has_key:
        return {
            "status": "not_trained",
            "score": None,
            "n_items": None,
            "note": (
                f"{env_used} is set and an API key is present, but no LLM "
                "judge is wired. Refusing to fabricate scores."
            ),
            "env": env_used,
        }
    if enabled:
        return {
            "status": "not_trained",
            "score": None,
            "note": (
                f"{env_used} is set but no API key "
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
            f"LLM judge off (no {names} / API key). "
            "Keyword rubric only. No fabricated LLM scores."
        ),
    }
