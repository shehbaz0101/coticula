"""CLI smoke: files, Exam 4, and the shipped Burgers trust pattern."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from coticula.cli import main
from coticula.grading import build_scorecard


def test_demo_smoke_writes_scorecard(tmp_path: Path):
    pytest.importorskip("torch")
    code = main(["demo", "--smoke", "--out", str(tmp_path)])
    assert code == 0
    payload = json.loads((tmp_path / "scorecard.json").read_text(encoding="utf-8"))
    assert payload["smoke"] is True
    assert payload["explain"]["n_items"] == 48
    assert payload["explain"]["gold"]["mean_coverage"] == pytest.approx(1.0)
    assert payload["explain"]["llm"]["status"] == "not_trained"
    names = [row["name"] for row in payload["pdes"]["burgers"]["models"]]
    assert "classical_fd" in names
    assert "fno" in names
    assert "persist_ic" in names
    assert (tmp_path / "scorecard.md").is_file()
    assert (tmp_path / "report.html").is_file()
    pngs = list(tmp_path.glob("*.png"))
    assert pngs, "expected at least one PNG plot"
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "Fail-closed" in html
    assert "persist_ic" in html


def test_grade_adapter_without_torch(tmp_path: Path):
    code = main(
        [
            "grade",
            "--adapter",
            "examples/persist_ic.py:PersistIC",
            "--pde",
            "burgers",
            "--smoke",
            "--no-ood",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 0
    payload = json.loads((tmp_path / "scorecard.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in payload["pdes"]["burgers"]["models"]]
    assert names == ["classical_fd", "persist_ic"]
    persist = payload["pdes"]["burgers"]["models"][1]
    assert persist["name"] == "persist_ic"
    assert persist["status"] == "ok"
    # Tiny grid is outside declared nx=64, and this toy stays under both cuts.
    assert persist["trust"]["status"] == "ood"
    assert "resolution_shift" in persist["trust"]["reasons"]


def test_shipped_fno_burgers_fails_closed_on_conserve():
    torch = pytest.importorskip("torch")
    del torch
    ckpt = Path("checkpoints/fno_burgers.pt")
    if not ckpt.is_file():
        pytest.skip("shipped burgers checkpoint is not in this checkout")
    card, _arrays = build_scorecard(
        pdes=["burgers"],
        model_names=["fno", "persist_ic"],
        include_ood=True,
        command="coticula grade --model fno --pde burgers",
    )
    rows = {row["name"]: row for row in card["pdes"]["burgers"]["models"]}
    row = rows["fno"]
    assert row["status"] == "ok"
    assert row["predict"]["rel_l2_mean"] <= 0.5
    assert row["conserve"]["residual_rel_l2"] > 1.0
    assert row["trust"]["status"] == "untrusted"
    assert "residual_too_high" in row["trust"]["reasons"]
    toy = rows["persist_ic"]
    assert toy["status"] == "ok"
    assert toy["predict"]["rel_l2_mean"] > row["predict"]["rel_l2_mean"]
    assert toy["trust"]["status"] == "trusted"
    assert toy["counterfactual"]["rel_l2_traj_delta_model"] == pytest.approx(0.0)
    ood = card["pdes"]["burgers"]["ood"][0]
    assert ood["id"] == "burgers_param_below"
    ood_by = {item["name"]: item for item in ood["models"]}
    ood_row = ood_by["fno"]
    assert ood_row["trust"]["status"] == "untrusted"
    assert "param_shift" in ood_row["trust"]["reasons"]
    text = " ".join(card["reading"])
    assert "inside the Predict cut" in text
    assert "accurate" not in text.lower()
