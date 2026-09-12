"""CPU tests for the Exam 4 fixed explanation item set."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.pins import EXAM4_ITEMS_PATH, sha256_file
from metrics.explain import (
    grade_exam4,
    llm_judge_exam4,
    load_exam4,
    score_explanation,
    score_item,
)


def test_exam4_item_set_is_real_and_sized():
    payload = load_exam4()
    items = payload["items"]
    assert 15 <= len(items) <= 50
    assert payload["n_items"] == len(items)
    assert payload["llm_judge"] == "not_wired"
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids))
    for it in items:
        assert it["prompt"].strip()
        assert len(it["expected_keywords"]) >= 2
        assert it.get("gold_rationale")


def test_gold_rationales_hit_expected_keywords():
    graded = grade_exam4(use_gold=True)
    assert graded["status"] == "keyword_rubric"
    assert graded["n_items"] >= 15
    assert graded["mean_coverage"] == pytest.approx(1.0, abs=1e-12)
    assert graded["n_perfect"] == graded["n_items"]
    assert graded["n_zero"] == 0
    assert graded["item_set_sha256"] == sha256_file(EXAM4_ITEMS_PATH)
    assert graded["llm_judge"]["status"] == "not_trained"


def test_empty_answers_score_zero():
    graded = grade_exam4(answers={})
    assert graded["mean_coverage"] == pytest.approx(0.0, abs=1e-12)
    assert graded["n_zero"] == graded["n_items"]
    assert all(r["answer_source"] == "missing" for r in graded["per_item"])


def test_junk_and_partial_answers():
    payload = load_exam4()
    first = payload["items"][0]
    junk = grade_exam4(answers={first["id"]: "asdf qwerty no physics here"})
    assert junk["per_item"][0]["coverage"] == 0.0
    one_kw = first["expected_keywords"][0]
    partial = score_item(first, one_kw)
    assert 0.0 < partial["coverage"] < 1.0
    assert one_kw in partial["hits"]


def test_score_item_is_casefold():
    item = {
        "id": "t",
        "pde": "burgers1d",
        "facet": "physics",
        "expected_keywords": ["Viscosity", "Shock"],
    }
    r = score_item(item, "VISCOSITY smooths a SHOCK")
    assert r["coverage"] == pytest.approx(1.0)
    assert r["status"] == "keyword_rubric"


def test_llm_judge_never_invents_a_score(monkeypatch):
    monkeypatch.delenv("VU_LLM_JUDGE", raising=False)
    monkeypatch.delenv("VU_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    off = llm_judge_exam4()
    assert off["status"] == "not_trained"
    assert off["score"] is None

    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key")
    keyed = llm_judge_exam4()
    assert keyed["status"] == "not_trained"
    assert keyed["score"] is None
    assert "fabricat" in keyed["note"].lower()

    monkeypatch.setenv("VU_LLM_JUDGE", "1")
    gated = llm_judge_exam4()
    assert gated["status"] == "not_trained"
    assert gated["score"] is None
    assert gated.get("n_items") in (None, 0)
    # Must not look like a measured LLM grade.
    assert "mean_coverage" not in gated


def test_legacy_score_explanation_still_stub():
    r = score_explanation("viscosity diffusion energy conservation residual")
    assert r["status"] == "stub_keyword_rubric"
    assert 0.0 < r["overall"] <= 1.0


def test_run_eval_explain_schema_when_report_present():
    report = ROOT / "reports" / "latest.json"
    if not report.exists():
        pytest.skip("reports/latest.json not generated yet")
    data = json.loads(report.read_text(encoding="utf-8"))
    expl = data["exams"]["explain"]
    assert expl["llm"]["status"] == "not_trained"
    assert expl["llm"].get("score") in (None, 0, 0.0) or "score" not in expl["llm"]
    if "gold_reference" in expl:
        gold = expl["gold_reference"]
        assert gold["status"] == "keyword_rubric"
        assert gold["n_items"] >= 15
        assert isinstance(gold["mean_coverage"], float)
        assert expl["llm"].get("score") is None
    else:
        assert expl["classical_keyword_stub"]["status"] == "stub_keyword_rubric"
