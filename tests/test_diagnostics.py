"""CPU tests for v0.2 residual-vs-L2 and CF-sensitivity diagnostics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.classical.burgers1d import solve_burgers
from baselines.classical.heat2d import solve_heat2d
from metrics.diagnostics import (
    burgers_cf_sensitivity,
    collect_scatter_from_report,
    heat_cf_sensitivity,
    horizon_residual_split,
    pearson_corr,
    scatter_summary,
    spearman_corr,
)


def test_pearson_and_spearman_identity():
    x = [1.0, 2.0, 3.0, 4.0]
    assert pearson_corr(x, x) == pytest.approx(1.0)
    assert spearman_corr(x, x) == pytest.approx(1.0)
    assert pearson_corr([1.0], [2.0]) is None
    assert pearson_corr([1.0, 1.0], [0.0, 1.0]) is None


def test_scatter_summary_from_toy_report():
    report = {
        "baselines": {
            "fno": {
                "burgers": {
                    "status": "ok",
                    "predict": {"rel_l2_mean": 0.16},
                    "conserve": {"residual_rel_l2": 4.8},
                    "trust": {"status": "untrusted"},
                }
            },
            "pino": {
                "burgers": {
                    "status": "ok",
                    "predict": {"rel_l2_mean": 0.19},
                    "conserve": {"residual_rel_l2": 1.7},
                    "trust": {"status": "untrusted"},
                }
            },
        },
        "ood": {
            "probes": [
                {
                    "meta": {"id": "burgers_param_below", "pde": "burgers"},
                    "fno": {
                        "status": "ok",
                        "predict": {"rel_l2_mean": 0.27},
                        "conserve": {"residual_rel_l2": 5.3},
                        "trust": {"status": "untrusted"},
                    },
                    "pino": {"status": "not_trained"},
                }
            ]
        },
    }
    pts = collect_scatter_from_report(report)
    assert len(pts) == 3
    summary = scatter_summary(pts)
    assert summary["n"] == 3
    assert summary["n_untrusted"] == 3
    assert summary["residual_max"] == pytest.approx(5.3)
    assert isinstance(summary["pearson_residual_vs_l2"], float)


def test_burgers_cf_sensitivity_monotone_enough():
    out = solve_burgers(nu=0.03, nx=32, nt=16, seed=3)
    sweep = burgers_cf_sensitivity(
        out["u"][0],
        0.03,
        scales=(1.0, 2.0, 3.0),
        nx=32,
        nt=16,
        L=2 * np.pi,
        T=1.0,
    )
    assert sweep["status"] == "ok"
    assert sweep["n_scales"] == 3
    rows = {r["scale"]: r for r in sweep["rows"]}
    assert rows[1.0]["classical_traj_delta"] == pytest.approx(0.0, abs=1e-12)
    assert rows[3.0]["classical_traj_delta"] > rows[2.0]["classical_traj_delta"]
    assert "model_traj_delta" not in rows[2.0]


def test_heat_cf_sensitivity_changes():
    out = solve_heat2d(alpha=0.1, n=12, nt=8, seed=2)
    sweep = heat_cf_sensitivity(
        out["u"][0],
        0.1,
        scales=(1.0, 1.5),
        n=12,
        nt=8,
        L=1.0,
        T=0.5,
    )
    assert sweep["rows"][0]["classical_traj_delta"] == pytest.approx(0.0, abs=1e-12)
    assert sweep["rows"][1]["classical_traj_delta"] > 1e-4


def test_horizon_residual_split_finite():
    out = solve_burgers(nu=0.02, nx=32, nt=20, seed=5)
    dx = float(out["x"][1] - out["x"][0])
    dt = float(out["t"][1] - out["t"][0])
    hz = horizon_residual_split(out["u"], dx=dx, dt=dt, param=0.02, pde="burgers")
    assert hz["status"] == "ok"
    assert np.isfinite(hz["residual_early"])
    assert np.isfinite(hz["residual_late"])
    assert hz["n_time_steps"] == 20
