"""Adapter protocol and a torch-free grade of the toy persist model."""
from __future__ import annotations

import numpy as np
import pytest

from baselines.classical.burgers1d import generate_dataset
from coticula.adapter import Surrogate, load_adapter_spec
from coticula.grading import _grid_burgers, grade_adapter, reading_lines
from metrics.trust import PREDICT_UNTRUSTED


def test_persist_ic_batch_shape_and_protocol():
    model = load_adapter_spec("examples/persist_ic.py:PersistIC")
    assert isinstance(model, Surrogate)
    assert model.name == "persist_ic"
    u0 = np.zeros((2, 16))
    u0[:, 0] = 1.0
    pred = model.predict(u0, np.array([0.01, 0.02]), n_steps=8)
    assert pred.shape == (2, 9, 16)
    assert np.allclose(pred[:, 0], u0)
    assert np.allclose(pred[:, -1], u0)


def test_persist_ic_rejects_unbatched_vector():
    model = load_adapter_spec("examples/persist_ic.py:PersistIC")
    with pytest.raises(ValueError):
        model.predict(np.zeros(16), np.array([0.01]), n_steps=4)


def test_persist_shape_on_a_short_grid():
    model = load_adapter_spec("examples/persist_ic.py:PersistIC")
    data = generate_dataset(n_traj=2, nx=32, nt=20, seed=1)
    grid = _grid_burgers(data)
    row, sample = grade_adapter(
        model, data, np.arange(2), pde="burgers", grid=grid, role="surrogate"
    )
    assert row["status"] == "ok"
    assert row["trust"]["status"] in {"trusted", "ood", "untrusted"}
    assert sample["pred"].shape == sample["truth"].shape
    # nx=32 is outside the declared train grid, so a quality pass is ood, not trusted.
    if row["predict"]["rel_l2_mean"] <= PREDICT_UNTRUSTED and row["conserve"]["residual_rel_l2"] <= 1.0:
        assert row["trust"]["status"] == "ood"
        assert "resolution_shift" in row["trust"]["reasons"]


def test_reading_states_predict_pass_and_conserve_fail():
    card = {
        "smoke": False,
        "pdes": {
            "burgers": {
                "models": [
                    {
                        "name": "fno",
                        "status": "ok",
                        "role": "surrogate",
                        "predict": {"rel_l2_mean": 0.15},
                        "conserve": {"residual_rel_l2": 4.0},
                        "trust": {
                            "status": "untrusted",
                            "reasons": ["residual_too_high"],
                        },
                        "counterfactual": {},
                    }
                ],
                "ood": [],
            }
        },
        "explain": {},
    }
    text = " ".join(reading_lines(card))
    assert "under the 0.5 cut" in text
    assert "over the 1 cut" in text
    assert "untrusted" in text
    assert "residual_too_high" in text
    assert "passes" not in text.lower()
    assert "fails" not in text.lower()
    assert "accurate" not in text.lower()
