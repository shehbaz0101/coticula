"""Fast checks for the opt-in weak residual, noise scaling, and protocol freeze.

These tests use manufactured or tiny fields. They do not score the paper test split.
"""
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
from metrics.conserve import burgers_residual_l2, heat_residual_l2
from metrics.predict import relative_l2
from metrics.weak_form import (
    WeakSpec,
    burgers_weak_per_traj,
    discrete_cs_bound_heat,
    heat_weak_per_traj,
    poly_bump,
)
from research.inverse_nu import sensor_rel_l2_smoke
from research.models import build_model
from research.noise import realized_rel, smooth_error_burgers, smooth_error_heat, white_error
from research.protocol import HASH_PATH, PROTOCOL_PATH, sha256_file, verify_frozen
from research.statsutil import best_baseline, scaled_score
from research.strong import (
    burgers_residual_field,
    burgers_strong_per_traj,
    burgers_white_second_moment_field,
    heat_residual_field,
    heat_strong_per_traj,
    heat_white_second_moment,
    keff_dirichlet,
)


SPEC = WeakSpec()


def test_protocol_hash_matches_frozen_bytes():
    digest = sha256_file(PROTOCOL_PATH)
    frozen = HASH_PATH.read_text(encoding="utf-8").strip()
    assert digest == frozen
    loaded, checked = verify_frozen()
    assert checked == digest
    assert loaded["training"]["architectures"] == ["fno", "unet", "deeponet"]
    assert loaded["noise"]["smooth_time"] == "time_varying"
    assert loaded["noise"]["levels"] == [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
    assert loaded["datasets"]["heat"]["boundary"] == "dirichlet_zero"
    assert "zeros" in loaded["zero_field_rule"]["excluded_from_residual_anchor"]
    assert "tr G" in loaded["weak_form"]["normalization"]
    assert loaded["symbol_check"]["formula"].startswith("E|R|^2")


def test_bump_derivative_matches_finite_difference():
    center, radius, power = 0.5, 0.25, 8
    step = 1e-6
    for t0 in (0.35, 0.5, 0.62):
        plus, _ = poly_bump(np.array([t0 + step]), center, radius, power)
        minus, _ = poly_bump(np.array([t0 - step]), center, radius, power)
        _, analytic = poly_bump(np.array([t0]), center, radius, power)
        fd = (plus[0] - minus[0]) / (2.0 * step)
        assert analytic[0] == pytest.approx(fd, abs=1e-6)
    edge, edge_t = poly_bump(np.array([0.0, 1.0]), center, radius, power)
    assert edge[0] == 0.0 and edge[-1] == 0.0
    assert edge_t[0] == 0.0 and edge_t[-1] == 0.0


def test_strong_form_matches_conserve():
    burgers = solve_burgers(nu=0.02, nx=32, nt=16, seed=1)
    dx = float(burgers["x"][1] - burgers["x"][0])
    dt = float(burgers["t"][1] - burgers["t"][0])
    got = float(burgers_strong_per_traj(burgers["u"], dx, dt, 0.02)[0])
    expect = burgers_residual_l2(burgers["u"], dx, dt, 0.02)
    assert got == pytest.approx(expect, rel=0, abs=1e-12)

    heat = solve_heat2d(alpha=0.1, n=16, nt=10, seed=2)
    h = float(heat["x"][1] - heat["x"][0])
    ht = float(heat["t"][1] - heat["t"][0])
    got_h = float(heat_strong_per_traj(heat["u"], h, ht, 0.1)[0])
    expect_h = heat_residual_l2(heat["u"], h, ht, 0.1)
    assert got_h == pytest.approx(expect_h, rel=0, abs=1e-12)
    assert np.allclose(heat["u"][:, 0, :], 0.0)
    assert np.allclose(heat["u"][:, :, 0], 0.0)


def test_manufactured_heat_weak_residual_is_near_floor():
    n, nt = 17, 16
    alpha = 0.12
    x = np.linspace(0.0, 1.0, n)
    t = np.linspace(0.0, 0.4, nt + 1)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    mode = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    u = mode[None, ...] * np.exp(-alpha * 2.0 * np.pi**2 * t)[:, None, None]
    true = float(heat_weak_per_traj(u, x, x, t, alpha, SPEC)[0])
    wrong = float(heat_weak_per_traj(u, x, x, t, 0.0, SPEC)[0])
    assert true < 2e-5
    assert wrong > 50.0 * true
    assert np.allclose(u[:, 0, :], 0.0)


def test_zero_field_has_zero_residual_and_is_excluded_from_anchor():
    burgers = solve_burgers(nu=0.03, nx=24, nt=12, seed=4)
    dx = float(burgers["x"][1] - burgers["x"][0])
    dt = float(burgers["t"][1] - burgers["t"][0])
    zeros = np.zeros_like(burgers["u"])
    assert float(burgers_strong_per_traj(zeros, dx, dt, 0.03)[0]) == pytest.approx(0.0, abs=1e-12)
    weak = float(burgers_weak_per_traj(zeros, burgers["x"], burgers["t"], 0.03, SPEC, u_ref=burgers["u"])[0])
    assert weak == pytest.approx(0.0, abs=1e-12)
    anchor = best_baseline(
        {"zeros": 0.0, "training_mean": 0.4, "frozen_initial": 0.2},
        exclude=("zeros",),
    )
    assert anchor == ("frozen_initial", 0.2)
    assert scaled_score(0.1, 0.2, 0.0) == pytest.approx(50.0)
    assert scaled_score(0.3, 0.2, 0.0) == pytest.approx(-50.0)
    assert scaled_score(0.0, 0.0, 0.0) is None
    assert scaled_score(0.1, 0.05, 0.05) is None


def test_heat_cauchy_bound_and_sine_symbol():
    heat = solve_heat2d(alpha=0.08, n=16, nt=8, seed=5)
    rng = np.random.default_rng(0)
    err = rng.normal(size=heat["u"].shape)
    err[:, 0, :] = err[:, -1, :] = err[:, :, 0] = err[:, :, -1] = 0.0
    bound = discrete_cs_bound_heat(err, heat["x"], heat["y"], heat["t"], 0.08, SPEC)
    assert bound["rho_over_rel"] <= 1.0 + 1e-8

    n = 17
    h = 1.0 / (n - 1)
    p = q = 2
    j = np.arange(n)
    mode = np.sin(np.pi * p * j / (n - 1))
    field = mode[:, None] * mode[None, :]
    field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = 0.0
    keff = keff_dirichlet(field, h)
    lam = (4.0 / h**2) * (
        np.sin(np.pi * p / (2.0 * (n - 1))) ** 2 + np.sin(np.pi * q / (2.0 * (n - 1))) ** 2
    )
    assert keff**2 == pytest.approx(lam, rel=1e-10)


def test_white_second_moment_matches_formula():
    dt, h, alpha = 0.04, 0.05, 0.1
    factor = heat_white_second_moment(dt, h, alpha, dim=2)
    assert factor == pytest.approx(2.0 / dt**2 + 10.0 * alpha**2 / h**4)
    rng = np.random.default_rng(1)
    s2 = 1e-4
    acc = []
    for _ in range(24):
        field = rng.normal(scale=np.sqrt(s2), size=(9, 12, 12))
        residual = heat_residual_field(field, h, dt, alpha)
        acc.append(float((residual**2).mean()))
    assert float(np.mean(acc)) == pytest.approx(s2 * factor, rel=0.08)

    burgers = solve_burgers(nu=0.02, nx=24, nt=10, seed=6)
    dx = float(burgers["x"][1] - burgers["x"][0])
    bdt = float(burgers["t"][1] - burgers["t"][0])
    tau = burgers_residual_field(burgers["u"], dx, bdt, 0.02)
    pred = burgers_white_second_moment_field(burgers["u"], dx, bdt, 0.02, s2)
    draws = []
    for _ in range(30):
        noisy = burgers["u"] + rng.normal(scale=np.sqrt(s2), size=burgers["u"].shape)
        residual = burgers_residual_field(noisy, dx, bdt, 0.02)
        draws.append(float(((residual - tau) ** 2).mean()))
    assert float(np.mean(draws)) == pytest.approx(float(pred.mean()), rel=0.12)


def test_noise_matches_rel_l2_and_respects_dirichlet():
    burgers = solve_burgers(nu=0.02, nx=24, nt=8, seed=8)
    heat = solve_heat2d(alpha=0.1, n=12, nt=6, seed=9)
    rng = np.random.default_rng(3)
    for eps in (0.001, 0.01, 0.1):
        err = white_error(burgers["u"], eps, rng, dirichlet=False)
        assert realized_rel(err, burgers["u"]) == pytest.approx(eps, rel=1e-10)
        smooth = smooth_error_burgers(
            burgers["u"], burgers["x"], burgers["t"], eps, rng, k_smooth=4, n_time_modes=2
        )
        assert realized_rel(smooth, burgers["u"]) == pytest.approx(eps, rel=1e-10)
        # Time-varying: the spatial profile is not constant across time.
        assert np.std(smooth, axis=0).mean() > 0.0
        herr = white_error(heat["u"], eps, rng, dirichlet=True)
        assert realized_rel(herr, heat["u"]) == pytest.approx(eps, rel=1e-10)
        assert np.allclose(herr[:, 0, :], 0.0)
        assert np.allclose(herr[:, :, 0], 0.0)
        hs = smooth_error_heat(
            heat["u"], heat["x"], heat["y"], heat["t"], eps, rng, k_smooth=3, n_time_modes=2
        )
        assert realized_rel(hs, heat["u"]) == pytest.approx(eps, rel=1e-10)
        assert np.allclose(hs[:, 0, :], 0.0)
        assert np.allclose(hs[:, :, -1], 0.0)


def test_model_forward_shapes_lock_boundaries():
    torch = pytest.importorskip("torch")
    burgers = build_model("deeponet", "burgers", {"nt": 8, "nx": 16}, {"rank": 8, "hidden": 16})
    u0 = torch.zeros(2, 16)
    u0[:, 1:-1] = 0.2
    out = burgers(u0, torch.tensor([0.02, 0.03]))
    assert tuple(out.shape) == (2, 9, 16)
    assert torch.allclose(out[:, 0], u0)

    heat = build_model("unet", "heat", {"nt": 6, "n": 12}, {"width": 4})
    h0 = torch.zeros(2, 12, 12)
    h0[:, 1:-1, 1:-1] = 0.3
    hout = heat(h0, torch.tensor([0.1, 0.12]))
    assert tuple(hout.shape) == (2, 7, 12, 12)
    assert torch.allclose(hout[:, 0], h0)
    assert float(hout[:, :, 0, :].detach().abs().max()) == pytest.approx(0.0)
    assert float(hout[:, :, :, -1].detach().abs().max()) == pytest.approx(0.0)


def test_sensor_interpolation_of_full_grid_is_exact():
    assert sensor_rel_l2_smoke(12) == pytest.approx(0.0, abs=1e-12)


def test_solver_weak_residual_below_wrong_coefficient():
    burgers = solve_burgers(nu=0.025, nx=32, nt=16, seed=11)
    true = float(burgers_weak_per_traj(burgers["u"], burgers["x"], burgers["t"], 0.025, SPEC)[0])
    wrong = float(burgers_weak_per_traj(burgers["u"], burgers["x"], burgers["t"], 0.0, SPEC)[0])
    assert true < wrong
    assert relative_l2(burgers["u"], burgers["u"]) == 0.0
