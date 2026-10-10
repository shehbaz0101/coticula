"""CLI smoke: files, Exam 4, and the shipped Burgers trust pattern."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from coticula.cli import main
from coticula.grading import _fallback_idx, _formula_holdout_idx, build_scorecard, holdout_idx
from coticula.report import markdown_table, terminal_scorecard, verdict_line, write_plots


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
    section = card["pdes"]["burgers"]
    assert section["split"] == "holdout"
    assert section["n_eval"] == 12
    assert section["n_total"] == 48
    assert section["eval_idx"] == list(range(36, 48))
    assert "checkpoint holdout" in card["eval_set"]
    rows = {row["name"]: row for row in section["models"]}
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
    plain = terminal_scorecard(card, color=False)
    assert "fno: passes Predict, fails Conserve, so untrusted" in plain
    assert "persist_ic: passes Predict, passes Conserve, so trusted" in plain
    assert "\x1b" not in plain
    colored = terminal_scorecard(card, color=True)
    assert "\x1b[31m" in colored
    assert "\x1b[32m" in colored


def test_verdict_lines_match_the_trust_flag():
    fno = {
        "name": "fno",
        "status": "ok",
        "role": "surrogate",
        "predict": {"rel_l2_mean": 1.573143e-01},
        "conserve": {"residual_rel_l2": 4.848762},
        "trust": {"status": "untrusted", "reasons": ["residual_too_high"]},
    }
    assert verdict_line(fno) == "fno: passes Predict, fails Conserve, so untrusted"
    ood = {
        "name": "persist_ic",
        "status": "ok",
        "role": "surrogate",
        "predict": {"rel_l2_mean": 0.2},
        "conserve": {"residual_rel_l2": 0.4},
        "trust": {"status": "ood", "reasons": ["resolution_shift"]},
    }
    assert verdict_line(ood) == (
        "persist_ic: passes Predict, passes Conserve, "
        "outside train support (resolution_shift), so ood"
    )
    both = {
        "name": "fno",
        "status": "ok",
        "role": "surrogate",
        "predict": {"rel_l2_mean": 0.57},
        "conserve": {"residual_rel_l2": 6.35},
        "trust": {
            "status": "untrusted",
            "reasons": ["param_shift", "residual_too_high", "predict_l2_too_high"],
        },
    }
    assert verdict_line(both) == (
        "fno: fails Predict, fails Conserve, "
        "also outside train support (param_shift), so untrusted"
    )
    assert verdict_line(
        {"name": "pino", "status": "not_trained", "reason": "missing_checkpoint"}
    ) == "pino: no checkpoint shipped, so not_trained"
    assert verdict_line(
        {
            "name": "classical_fd",
            "status": "ok",
            "role": "labeler",
            "predict": {"rel_l2_mean": 0.0},
            "conserve": {"residual_rel_l2": 0.013},
            "trust": {"status": "reference", "reasons": []},
        }
    ) == "classical_fd: reference labeler, not a learned trust score"


def test_terminal_uses_short_numbers_and_markdown_keeps_full_precision():
    card = {
        "version": "0.3.1",
        "command": "coticula grade --model fno --pde burgers",
        "command_kind": "grade",
        "smoke": False,
        "eval_set": (
            "Eval set: burgers: checkpoint holdout, 12 of 48 trajectories, indices 36-47. "
            "Same checkpoint holdout as coticula demo."
        ),
        "trust_policy": {"predict_untrusted": 0.5, "residual_untrusted": 1.0, "rule": "fail closed"},
        "pdes": {
            "burgers": {
                "label_source": "labels",
                "split": "holdout",
                "split_label": "checkpoint holdout",
                "n_total": 48,
                "n_eval": 12,
                "eval_idx": list(range(36, 48)),
                "models": [
                    {
                        "name": "fno",
                        "status": "ok",
                        "role": "surrogate",
                        "predict": {"rel_l2_mean": 1.573143e-01},
                        "conserve": {"residual_rel_l2": 4.848762e00},
                        "trust": {"status": "untrusted", "reasons": ["residual_too_high"]},
                    }
                ],
                "ood": [],
            }
        },
        "explain": {
            "n_items": 48,
            "gold": {"mean_coverage": 1.0, "status": "ok"},
            "rule_based": {"mean_coverage": 0.878472, "status": "ok"},
            "metric_dump": {"mean_coverage": 0.090278, "status": "ok"},
            "llm": {"status": "not_trained"},
        },
        "reading": [],
    }
    plain = terminal_scorecard(card, color=False)
    assert "0.157" in plain
    assert "4.849" in plain
    assert "1.573143e-01" not in plain
    assert "fno: passes Predict, fails Conserve, so untrusted" in plain
    assert "checkpoint holdout" in plain
    assert "\x1b" not in plain
    assert "\x1b" not in terminal_scorecard(card)
    colored = terminal_scorecard(card, color=True)
    assert "\x1b[31muntrusted\x1b[0m" in colored
    md = markdown_table(card)
    assert "1.573143e-01" in md
    assert "4.848762e+00" in md
    assert "checkpoint holdout" in md


def test_grade_split_holdout_is_smaller_than_the_smoke_default(tmp_path: Path):
    default = tmp_path / "default"
    holdout = tmp_path / "holdout"
    argv = [
        "grade",
        "--adapter",
        "examples/persist_ic.py:PersistIC",
        "--pde",
        "burgers",
        "--smoke",
        "--no-ood",
        "--out",
    ]
    assert main([*argv, str(default)]) == 0
    assert main([*argv, str(holdout), "--split", "holdout"]) == 0
    default_card = json.loads((default / "scorecard.json").read_text(encoding="utf-8"))
    hold_card = json.loads((holdout / "scorecard.json").read_text(encoding="utf-8"))
    assert default_card["pdes"]["burgers"]["split"] == "tail"
    assert default_card["pdes"]["burgers"]["n_eval"] == 2
    assert hold_card["pdes"]["burgers"]["split"] == "holdout"
    assert hold_card["pdes"]["burgers"]["n_eval"] == 1
    assert "Smoke grid" in hold_card["eval_set"]
    pred = hold_card["pdes"]["burgers"]["models"][1]["predict"]["rel_l2_mean"]
    md = (holdout / "scorecard.md").read_text(encoding="utf-8")
    assert f"{pred:.6e}" in md
    assert f"{pred:.6e}" not in terminal_scorecard(hold_card, color=False)


def test_published_split_sizes():
    assert list(_formula_holdout_idx("burgers", 48)) == list(range(36, 48))
    assert list(_formula_holdout_idx("heat2d", 32)) == list(range(24, 32))
    assert list(_fallback_idx(48, None)) == list(range(48))
    assert list(_fallback_idx(32, None)) == list(range(32))
    pytest.importorskip("torch")
    if Path("checkpoints/fno_burgers.pt").is_file():
        assert list(holdout_idx("burgers", 48)) == list(range(36, 48))
    if Path("checkpoints/fno_heat2d.pt").is_file():
        assert list(holdout_idx("heat2d", 32)) == list(range(24, 32))


def test_generated_plots_keep_text_inside_the_figure(tmp_path: Path):
    pytest.importorskip("matplotlib")
    from metrics.trust import refuse_silent_heatmap

    banner = refuse_silent_heatmap(
        {"status": "untrusted", "reasons": ["residual_too_high"], "untrusted": True}
    )
    assert "do not present a silent pretty heatmap" in banner
    card = {
        "trust_policy": {"predict_untrusted": 0.5, "residual_untrusted": 1.0},
        "pdes": {
            "burgers": {
                "models": [
                    {
                        "name": "classical_fd",
                        "status": "ok",
                        "predict": {"rel_l2_mean": 0.0},
                        "conserve": {"residual_rel_l2": 0.013},
                    },
                    {
                        "name": "fno",
                        "status": "ok",
                        "predict": {"rel_l2_mean": 0.157},
                        "conserve": {"residual_rel_l2": 4.849},
                    },
                    {
                        "name": "pino",
                        "status": "ok",
                        "predict": {"rel_l2_mean": 0.188},
                        "conserve": {"residual_rel_l2": 1.733},
                    },
                    {
                        "name": "persist_ic",
                        "status": "ok",
                        "predict": {"rel_l2_mean": 0.377},
                        "conserve": {"residual_rel_l2": 0.809},
                    },
                ]
            },
            "heat2d": {
                "models": [
                    {
                        "name": "fno",
                        "status": "ok",
                        "predict": {"rel_l2_mean": 0.322},
                        "conserve": {"residual_rel_l2": 13.034},
                    }
                ]
            },
        },
    }
    arrays = {
        ("burgers", "fno"): {
            "truth": np.linspace(0, 1, 81 * 64).reshape(81, 64),
            "pred": np.zeros((81, 64)),
            "banner": banner,
            "name": "fno",
        },
        ("heat2d", "fno"): {
            "truth": np.linspace(0, 1, 41 * 32 * 32).reshape(41, 32, 32),
            "pred": np.zeros((41, 32, 32)),
            "banner": banner,
            "name": "fno",
        },
    }
    names = write_plots(card, arrays, tmp_path)
    assert "field_burgers_fno.png" in names
    assert "field_heat2d_fno.png" in names
    assert "bars_burgers.png" in names
    assert "bars_heat2d.png" in names
    # Small residuals used to push a y tick (1.2) past the top of the figure.
    small = {
        "trust_policy": {"predict_untrusted": 0.5, "residual_untrusted": 1.0},
        "pdes": {
            "burgers": {
                "models": [
                    {
                        "name": "classical_fd",
                        "status": "ok",
                        "predict": {"rel_l2_mean": 0.0},
                        "conserve": {"residual_rel_l2": 0.013},
                    },
                    {
                        "name": "persist_ic",
                        "status": "ok",
                        "predict": {"rel_l2_mean": 0.364},
                        "conserve": {"residual_rel_l2": 0.786},
                    },
                ]
            }
        },
    }
    small_dir = tmp_path / "small"
    small_dir.mkdir()
    write_plots(small, {}, small_dir)
