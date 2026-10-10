"""Checks for the v2 calibrated ratio. They do not score the paper test split."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.amendment import calibrate
from research.protocol import verify_amendment
from research.statsutil import paired_wilcoxon


def test_v2_digest_cites_frozen_v1():
    amendment, digest, _protocol, v1 = verify_amendment()
    assert amendment["amends_sha256"] == v1
    assert amendment["version"] == "2"
    assert amendment["scaled_score_v2"]["applies_to"] == "rel_l2 only"
    assert amendment["scaled_score_v2"]["residual_metrics"] == "not scored with S"
    assert "median_smooth" in amendment["calibrated_ratio"]["formula"]
    assert amendment["verdict_v2"]["rule"].startswith("The legacy untrusted flag survives")
    assert amendment["paired_tests"]["test"] == "wilcoxon_signed_rank"
    assert amendment["mismatch"]["matched_direction_scale"] == 2.0
    assert digest


def test_calibrated_ratio_uses_smooth_median_and_band():
    levels = np.array([0.01, 0.1])
    median = np.array([0.2, 0.4])
    p05 = np.array([0.1, 0.2])
    p95 = np.array([0.3, 0.6])
    above = calibrate(0.5, 0.01, levels, median, p05, p95)
    assert above["C"] == pytest.approx(2.5)
    assert above["above_p95"] is True
    inside = calibrate(0.2, 0.01, levels, median, p05, p95)
    assert inside["C"] == pytest.approx(1.0)
    assert inside["above_p95"] is False
    below = calibrate(0.2, 1e-4, levels, median, p05, p95)
    assert below["extrapolated"] is True
    assert below["above_p95"] is None
    assert below["C"] == pytest.approx(0.2 / 0.2)
    missing = calibrate(0.2, 0.0, levels, median, p05, p95)
    assert missing["C"] is None
    assert missing["above_p95"] is None


def test_wilcoxon_reports_effect_size():
    rng = np.random.default_rng(0)
    diff = 0.4 + 0.05 * rng.normal(size=20)
    out = paired_wilcoxon(diff)
    assert out["test"] == "wilcoxon_signed_rank"
    assert out["n"] == 20
    assert out["mean_diff"] > 0
    assert out["cohen_dz"] > 1
    assert out["p_value"] < 0.01
