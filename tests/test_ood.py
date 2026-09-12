"""CPU tests for OOD helpers and fail-closed trust flags."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics.ood import (
    BURGERS_OOD_NU,
    generate_burgers_cases,
    generate_heat_cases,
    make_burgers_ic,
    make_heat_ic,
    materialize_probe,
    resample_1d_periodic,
    resample_2d_dirichlet,
    score_burgers_labeler,
    score_burgers_surrogate,
    score_heat_labeler,
)
from metrics.trust import (
    BURGERS_TRAIN_SUPPORT,
    HEAT_TRAIN_SUPPORT,
    RESIDUAL_UNTRUSTED,
    decide_trust,
    param_in_range,
    refuse_silent_heatmap,
)


def test_param_in_range_inclusive():
    lo, hi = BURGERS_TRAIN_SUPPORT["param_range"]
    assert param_in_range(lo, lo, hi)
    assert param_in_range(hi, lo, hi)
    assert not param_in_range(BURGERS_OOD_NU["below"], lo, hi)
    assert not param_in_range(BURGERS_OOD_NU["above"], lo, hi)


def test_decide_trust_iid_low_residual_is_trusted():
    d = decide_trust(
        param=0.02,
        param_range=BURGERS_TRAIN_SUPPORT["param_range"],
        residual_rel_l2=0.02,
        predict_rel_l2=0.01,
        resolution=64,
        train_resolution=64,
        ic_family="fourier_modes",
        train_ic_family="fourier_modes",
    )
    assert d["status"] == "trusted"
    assert d["ood"] is False
    assert d["untrusted"] is False


def test_decide_trust_param_shift_is_ood():
    d = decide_trust(
        param=0.001,
        param_range=BURGERS_TRAIN_SUPPORT["param_range"],
        residual_rel_l2=0.02,
        predict_rel_l2=0.05,
        resolution=64,
        train_resolution=64,
        ic_family="fourier_modes",
        train_ic_family="fourier_modes",
    )
    assert d["status"] == "ood"
    assert d["ood"] is True
    assert "param_shift" in d["reasons"]
    assert "OOD" in d["banner"]


def test_decide_trust_high_residual_never_trusted():
    d = decide_trust(
        param=0.02,
        param_range=BURGERS_TRAIN_SUPPORT["param_range"],
        residual_rel_l2=RESIDUAL_UNTRUSTED + 3.0,
        predict_rel_l2=0.16,
        resolution=64,
        train_resolution=64,
        ic_family="fourier_modes",
        train_ic_family="fourier_modes",
    )
    assert d["status"] == "untrusted"
    assert d["untrusted"] is True
    assert "residual_too_high" in d["reasons"]
    banner = refuse_silent_heatmap(d)
    assert "UNTRUSTED" in banner
    assert "heatmap" in banner.lower()


def test_decide_trust_ood_plus_residual_is_untrusted():
    """Fail-closed: quality failure wins so a pretty OOD field is not trusted."""
    d = decide_trust(
        param=0.15,
        param_range=BURGERS_TRAIN_SUPPORT["param_range"],
        residual_rel_l2=4.0,
        predict_rel_l2=0.2,
        resolution=64,
        train_resolution=64,
        ic_family="fourier_modes",
        train_ic_family="fourier_modes",
    )
    assert d["status"] == "untrusted"
    assert d["ood"] is True
    assert "param_shift" in d["reasons"]
    assert "residual_too_high" in d["reasons"]


def test_decide_trust_resolution_and_ic_family():
    res = decide_trust(
        resolution=128,
        train_resolution=64,
        residual_rel_l2=0.01,
        predict_rel_l2=0.01,
    )
    assert res["status"] == "ood"
    assert "resolution_shift" in res["reasons"]
    ic = decide_trust(
        ic_family="tanh_front",
        train_ic_family="fourier_modes",
        residual_rel_l2=0.01,
        predict_rel_l2=0.01,
    )
    assert ic["status"] == "ood"
    assert "ic_family_shift" in ic["reasons"]


def test_labeler_role_is_reference():
    d = decide_trust(
        param=0.001,
        param_range=BURGERS_TRAIN_SUPPORT["param_range"],
        residual_rel_l2=0.02,
        predict_rel_l2=0.0,
        role="labeler",
    )
    assert d["status"] == "reference"
    assert d["ood"] is True


def test_burgers_ic_families_shapes():
    for fam in ("fourier_modes", "gaussian_pulse", "tanh_front"):
        u = make_burgers_ic(32, 2 * np.pi, fam, seed=3)
        assert u.shape == (32,)
        assert np.isfinite(u).all()


def test_heat_ic_families_dirichlet():
    for fam in ("gaussian_bump", "two_bump", "sinusoid"):
        u = make_heat_ic(16, 1.0, fam, seed=4)
        assert u.shape == (16, 16)
        assert np.allclose(u[0, :], 0.0)
        assert np.allclose(u[:, 0], 0.0)


def test_resample_1d_periodic_roundtrip_mean():
    rng = np.random.default_rng(0)
    u = rng.normal(size=64)
    up = resample_1d_periodic(u, 128)
    down = resample_1d_periodic(up, 64)
    assert up.shape == (128,)
    # Fourier upsample/downsample should be close for a generic field
    assert np.linalg.norm(down - u) / (np.linalg.norm(u) + 1e-12) < 0.25


def test_resample_2d_dirichlet_walls():
    u = make_heat_ic(16, 1.0, "gaussian_bump", seed=1)
    hi = resample_2d_dirichlet(u, 24)
    assert hi.shape == (24, 24)
    assert np.allclose(hi[0, :], 0.0)
    assert np.allclose(hi[:, -1], 0.0)


def test_generate_burgers_param_shift_is_ood_not_iid():
    cases = generate_burgers_cases(
        n=2, nu=0.001, nx=32, family="fourier_modes", seed=11, nt=12
    )
    scored = score_burgers_labeler(cases)
    assert scored["predict"]["rel_l2_mean"] == pytest.approx(0.0, abs=1e-12)
    assert scored["trust"]["ood"] is True
    assert scored["trust"]["status"] == "reference"
    assert "param_shift" in scored["trust"]["reasons"]


def test_generate_heat_ic_shift_flags_family():
    cases = generate_heat_cases(
        n_cases=1, alpha=0.1, n=12, family="sinusoid", seed=8, nt=6
    )
    scored = score_heat_labeler(cases)
    assert "ic_family_shift" in scored["trust"]["reasons"]
    assert np.isfinite(scored["conserve"]["residual_rel_l2"])


def test_surrogate_noise_is_untrusted():
    cases = generate_burgers_cases(
        n=2, nu=0.02, nx=32, family="fourier_modes", seed=2, nt=10
    )
    noise = cases["u"] + 2.0
    scored = score_burgers_surrogate(noise, cases)
    assert scored["trust"]["status"] == "untrusted"
    assert scored["predict"]["rel_l2_mean"] > 0.5


def test_materialize_probe_tiny_param_shift():
    spec = {
        "id": "burgers_param_below",
        "pde": "burgers",
        "probe": "param_shift",
        "n": 1,
        "nu": 0.001,
        "nx": 32,
        "family": "fourier_modes",
        "seed": 99,
    }
    # override nt via generate — materialize uses default nt=80 which is fine but slower
    # Use generate_* directly for speed; just check spec wiring here with nx=32 nt default.
    # Default nt=80 is acceptable for one trajectory on CPU.
    cases = materialize_probe(spec)
    assert cases["param"] == 0.001
    assert cases["labeler"]["trust"]["ood"] is True


def test_report_ood_section_when_present():
    report = ROOT / "reports" / "latest.json"
    if not report.exists():
        pytest.skip("reports/latest.json not generated yet")
    data = json.loads(report.read_text(encoding="utf-8"))
    if "ood" not in data or data["ood"] is None:
        pytest.skip("OOD section not in this report")
    ood = data["ood"]
    assert "probes" in ood
    assert "trust_policy" in ood or "trust_policy" in data
    ids = {row["meta"]["id"] for row in ood["probes"]}
    assert any(i.startswith("burgers_param_") for i in ids)
    assert any(i.startswith("heat2d_param_") for i in ids)
    for row in ood["probes"]:
        assert row["classical"]["status"] == "ok"
        fno = row["fno"]
        if fno.get("status") == "ok":
            assert np.isfinite(fno["predict"]["rel_l2_mean"])
            assert fno["trust"]["status"] in ("ood", "untrusted", "trusted")
    if data.get("failure_analysis"):
        assert "notes" in data["failure_analysis"]


def test_heat_train_support_alpha_ood():
    lo, hi = HEAT_TRAIN_SUPPORT["param_range"]
    assert not param_in_range(0.01, lo, hi)
    assert not param_in_range(0.50, lo, hi)
    assert param_in_range(0.10, lo, hi)
