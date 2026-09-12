"""Smoke tests for Coticula metrics, solvers, stubs, and report shape."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines import fno, llm, pino
from baselines.classical.burgers1d import generate_dataset as gen_burgers
from baselines.classical.burgers1d import solve_burgers
from baselines.classical.heat2d import generate_dataset as gen_heat
from baselines.classical.heat2d import solve_heat2d
from metrics.conserve import audit_burgers, audit_heat
from metrics.counterfactual import (
    burgers_counterfactual,
    grade_burgers_cf_prediction,
    heat_counterfactual,
)
from metrics.explain import score_explanation
from metrics.predict import batch_relative_l2, nmse, relative_l2


def test_relative_l2_identity():
    a = np.ones((4, 8))
    assert relative_l2(a, a) == pytest.approx(0.0, abs=1e-15)


def test_nmse_identity():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(3, 5, 5))
    assert nmse(a, a) == pytest.approx(0.0, abs=1e-15)


def test_burgers_solver_shapes():
    out = solve_burgers(nu=0.02, nx=32, nt=20, seed=3)
    assert out["u"].shape == (21, 32)
    assert out["x"].shape == (32,)
    assert np.isfinite(out["u"]).all()


def test_heat_solver_shapes():
    out = solve_heat2d(alpha=0.1, n=16, nt=10, seed=4)
    assert out["u"].shape == (11, 16, 16)
    assert np.allclose(out["u"][:, 0, :], 0.0)
    assert np.isfinite(out["u"]).all()


def test_burgers_self_consistency_small():
    out = solve_burgers(nu=0.03, nx=32, nt=15, seed=7)
    again = solve_burgers(nu=0.03, nx=32, nt=15, u0=out["u"][0])
    assert relative_l2(again["u"], out["u"]) == pytest.approx(0.0, abs=1e-12)


def test_batch_relative_l2():
    truth = gen_burgers(n_traj=4, nx=32, nt=10, seed=2)["u"]
    m = batch_relative_l2(truth, truth)
    assert m["mean"] == pytest.approx(0.0, abs=1e-15)


def test_conserve_burgers():
    data = gen_burgers(n_traj=2, nx=32, nt=20, seed=5)
    dx = float(data["x"][1] - data["x"][0])
    dt = float(data["t"][1] - data["t"][0])
    aud = audit_burgers(data["u"], dx=dx, dt=dt, nu=data["nu"])
    assert np.isfinite(aud["residual_rel_l2"])
    assert aud["residual_rel_l2"] >= 0.0


def test_conserve_heat():
    data = gen_heat(n_traj=2, n=16, nt=10, seed=6)
    dx = float(data["x"][1] - data["x"][0])
    dt = float(data["t"][1] - data["t"][0])
    aud = audit_heat(data["u"], dx=dx, dt=dt, alpha=data["alpha"])
    assert np.isfinite(aud["residual_rel_l2"])


def test_counterfactual_burgers_changes():
    out = solve_burgers(nu=0.02, nx=32, nt=20, seed=9)
    cf = burgers_counterfactual(out["u"][0], 0.02, 0.08, nx=32, nt=20)
    assert cf["rel_l2_traj_delta"] > 1e-6


def test_counterfactual_heat_changes():
    out = solve_heat2d(alpha=0.1, n=16, nt=10, seed=8)
    cf = heat_counterfactual(out["u"][0], 0.1, 0.2, n=16, nt=10)
    assert cf["rel_l2_traj_delta"] > 1e-6


def test_grade_burgers_cf_against_classical():
    out = solve_burgers(nu=0.02, nx=32, nt=20, seed=9)
    graded = grade_burgers_cf_prediction(out["u"], out["u"][0], 0.02, 0.08, nx=32, nt=20)
    assert graded["rel_l2_vs_classical_cf"] > 1e-6
    assert np.isfinite(graded["nmse_vs_classical_cf"])


def test_explain_keyword_rubric():
    r = score_explanation("viscosity diffusion energy conservation residual")
    assert 0.0 < r["overall"] <= 1.0
    assert r["status"] == "stub_keyword_rubric"


def test_learned_baselines_status_and_no_invented_weights():
    """FNO/PINO architectures are in-repo; missing ckpts stay not_trained. LLM stays stub."""
    assert fno.STATUS == "implemented"
    assert pino.STATUS == "implemented"
    assert llm.STATUS == "not_trained"
    missing = fno.load_model(checkpoint=ROOT / "checkpoints" / "definitely_missing.pt")
    assert missing["status"] == "not_trained"
    assert missing["model"] is None
    missing_pino = pino.load_model(checkpoint=ROOT / "checkpoints" / "definitely_missing.pt")
    assert missing_pino["status"] == "not_trained"


def test_report_exists_after_pipeline():
    """If labels+eval already run, assert report schema; else skip lightly."""
    report = ROOT / "reports" / "latest.json"
    if not report.exists():
        pytest.skip("reports/latest.json not generated yet")
    data = json.loads(report.read_text(encoding="utf-8"))
    for key in ("predict", "conserve", "counterfactual", "explain"):
        assert key in data["exams"]
    assert data["baselines"]["classical"]["status"] == "ok"
    bp = data["exams"]["predict"]["classical_burgers"]
    assert isinstance(bp["rel_l2_mean"], float)
    assert np.isfinite(bp["rel_l2_mean"])
    # FNO/PINO: either measured finite numbers or an honest not_trained — never a fake SOTA.
    for name in ("fno", "pino"):
        block = data["exams"]["predict"][name]
        assert block["status"] in ("ok", "not_trained")
        if block["status"] == "ok":
            for pde in ("burgers", "heat2d"):
                sub = block.get(pde) or {}
                if sub.get("status") == "ok":
                    assert np.isfinite(sub["rel_l2_mean"])
    assert data["exams"]["explain"]["llm"]["status"] == "not_trained"
    # Keyword rubric is not a fabricated LLM score.
    stub = data["exams"]["explain"].get("classical_keyword_stub") or {}
    if stub:
        assert stub["status"] == "stub_keyword_rubric"
    gold = data["exams"]["explain"].get("gold_reference")
    if gold:
        assert gold["status"] == "keyword_rubric"
        assert gold["n_items"] >= 15
        assert data["exams"]["explain"]["llm"].get("score") is None
    if "project" in data:
        assert data["project"] == "Coticula"
    if data.get("bench"):
        assert "coticula" in str(data["bench"]).lower() or "vu-bench" in str(data["bench"])
    # Week 3: OOD is a distinct block, never mixed into Exam 1 IID tables.
    if data.get("ood"):
        assert "probes" in data["ood"]
        assert "predict" in data["exams"]
    if data.get("trust_policy"):
        assert "residual_untrusted" in data["trust_policy"]
