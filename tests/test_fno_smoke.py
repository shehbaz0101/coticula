"""CPU smoke tests for FNO / PINO (skip if torch is not installed)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

from baselines.classical.burgers1d import generate_dataset as gen_burgers
from baselines.classical.burgers1d import solve_burgers
from baselines.classical.heat2d import generate_dataset as gen_heat
from baselines.fno.eval_exams import eval_burgers_operator
from baselines.fno.io import load_checkpoint, save_checkpoint
from baselines.fno.models import BurgersFNO, HeatFNO, count_parameters
from baselines.fno.train import train_burgers, train_heat
from baselines.pino.losses import LOSS_TERMS_DOC, pino_loss
from metrics.counterfactual import grade_burgers_cf_prediction
from metrics.predict import relative_l2


def test_burgers_fno_forward_cpu():
    model = BurgersFNO(modes=2, width=4, n_layers=2, nt=8, nx=16, fc_dim=8)
    u0 = torch.randn(2, 16)
    nu = torch.tensor([0.01, 0.02])
    out = model(u0, nu)
    assert out.shape == (2, 9, 16)
    assert torch.allclose(out[:, 0], u0)
    assert count_parameters(model) > 0


def test_heat_fno_forward_cpu_dirichlet():
    model = HeatFNO(modes=2, width=4, n_layers=2, nt=6, n=12, fc_dim=8)
    u0 = torch.rand(1, 12, 12)
    u0[:, 0, :] = u0[:, -1, :] = u0[:, :, 0] = u0[:, :, -1] = 0.0
    alpha = torch.tensor([0.1])
    out = model(u0, alpha)
    assert out.shape == (1, 7, 12, 12)
    assert torch.allclose(out[:, 0], u0)
    assert torch.allclose(out[:, :, 0, :], torch.zeros(1, 7, 12))
    assert torch.allclose(out[:, :, :, 0], torch.zeros(1, 7, 12))


def test_pino_loss_terms_finite_and_documented():
    for key in ("L", "L_data", "L_pde_burgers", "L_pde_heat", "L_ic", "citation"):
        assert key in LOSS_TERMS_DOC
    u = torch.randn(2, 9, 16)
    truth = u + 0.01 * torch.randn_like(u)
    nu = torch.tensor([0.02, 0.03])
    terms = pino_loss(u, truth, nu, pde="burgers1d", dx=2 * np.pi / 16, dt=1.0 / 8)
    assert terms["total"].ndim == 0
    assert torch.isfinite(terms["data"])
    assert torch.isfinite(terms["pde"])
    # FNO-only: λ_pde=0 ⇒ total == data (+ 0*ic)
    data_only = pino_loss(
        u, truth, nu, pde="burgers1d", dx=0.1, dt=0.1, lambda_pde=0.0, lambda_ic=0.0
    )
    assert torch.allclose(data_only["total"], data_only["data"])


def test_train_few_steps_writes_checkpoint(tmp_path):
    data = gen_burgers(n_traj=6, nx=16, nt=8, seed=0)
    out = tmp_path / "fno_burgers_smoke.pt"
    result = train_burgers(
        data=data,
        epochs=2,
        width=4,
        modes=2,
        n_layers=2,
        batch_size=4,
        eval_n=2,
        device="cpu",
        out_path=out,
        log_every=1,
    )
    assert out.exists()
    loaded = load_checkpoint(out, device="cpu")
    assert loaded["status"] == "ok"
    assert loaded["model"] is not None
    pred = loaded["model"](
        torch.from_numpy(data["u"][:2, 0].astype(np.float32)),
        torch.from_numpy(data["nu"][:2].astype(np.float32)),
    )
    assert pred.shape == (2, 9, 16)
    assert result["training"]["objective"] == "fno"


def test_train_pino_few_steps(tmp_path):
    data = gen_burgers(n_traj=6, nx=16, nt=8, seed=1)
    out = tmp_path / "pino_burgers_smoke.pt"
    result = train_burgers(
        data=data,
        epochs=2,
        width=4,
        modes=2,
        n_layers=2,
        batch_size=4,
        eval_n=2,
        pino=True,
        lambda_pde=0.1,
        device="cpu",
        out_path=out,
        log_every=1,
    )
    assert out.exists()
    assert result["training"]["lambda_pde"] == 0.1
    assert result["training"]["objective"] == "pino"


def test_train_heat_few_steps(tmp_path):
    data = gen_heat(n_traj=4, n=12, nt=6, seed=2)
    out = tmp_path / "fno_heat_smoke.pt"
    train_heat(
        data=data,
        epochs=2,
        width=4,
        modes=2,
        n_layers=2,
        batch_size=2,
        eval_n=2,
        device="cpu",
        out_path=out,
        log_every=1,
    )
    loaded = load_checkpoint(out, device="cpu")
    assert loaded["status"] == "ok"


def test_missing_checkpoint_is_not_trained(tmp_path):
    loaded = load_checkpoint(tmp_path / "nope.pt")
    assert loaded["status"] == "not_trained"
    assert loaded["model"] is None


def test_exams_1_to_3_on_tiny_fno():
    data = gen_burgers(n_traj=4, nx=16, nt=8, seed=3)
    result = train_burgers(
        data=data,
        epochs=3,
        width=8,
        modes=4,
        n_layers=2,
        batch_size=4,
        eval_n=2,
        device="cpu",
        out_path=None,
        log_every=1,
    )
    report = eval_burgers_operator(
        result["model"],
        data,
        idx=np.array([2, 3]),
        device="cpu",
        model_name="fno",
    )
    assert report["status"] == "ok"
    assert np.isfinite(report["predict"]["rel_l2_mean"])
    assert np.isfinite(report["conserve"]["residual_rel_l2"])
    assert np.isfinite(report["counterfactual"]["rel_l2_vs_classical_cf"])


def test_grade_cf_vs_classical():
    out = solve_burgers(nu=0.02, nx=16, nt=8, seed=4)
    # Identity prediction at ν′ should have nonzero error vs classical ν′ solve
    # unless ν′==ν; use the original traj as a dummy pred at 2ν.
    graded = grade_burgers_cf_prediction(
        out["u"], out["u"][0], 0.02, 0.08, nx=16, nt=8
    )
    assert graded["rel_l2_vs_classical_cf"] > 1e-6
    assert graded["rel_l2_traj_delta_classical"] > 1e-6


def test_save_roundtrip_config(tmp_path):
    model = BurgersFNO(modes=2, width=4, n_layers=2, nt=4, nx=8, fc_dim=8)
    path = tmp_path / "roundtrip.pt"
    save_checkpoint(
        path,
        model,
        objective="fno",
        training={"epochs": 0},
        extra={"split": {"eval_idx": [0]}},
    )
    loaded = load_checkpoint(path)
    assert loaded["config"]["nx"] == 8
    assert loaded["extra"]["split"]["eval_idx"] == [0]


def test_untrained_fno_error_is_real_not_placeholder():
    """Random weights yield a finite, typically large error — not a fake 0.0 SOTA."""
    data = gen_burgers(n_traj=2, nx=16, nt=8, seed=5)
    model = BurgersFNO(modes=2, width=4, n_layers=2, nt=8, nx=16, fc_dim=8)
    model.eval()
    with torch.no_grad():
        pred = model(
            torch.from_numpy(data["u"][:, 0].astype(np.float32)),
            torch.from_numpy(data["nu"].astype(np.float32)),
        ).numpy()
    err = relative_l2(pred, data["u"])
    assert np.isfinite(err)
    assert err > 0.0
