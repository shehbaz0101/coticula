"""Run Week-1 eval: classical baseline on generated labels → reports/latest.*"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines import fno as fno_mod
from baselines import llm as llm_mod
from baselines import pino as pino_mod
from baselines.classical.burgers1d import solve_burgers
from baselines.classical.heat2d import solve_heat2d
from metrics.conserve import audit_burgers, audit_heat
from metrics.counterfactual import burgers_counterfactual, heat_counterfactual
from metrics.explain import score_explanation
from metrics.predict import batch_relative_l2, relative_l2


def _load_burgers() -> dict:
    path = ROOT / "datasets" / "burgers" / "burgers_v0.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run scripts.generate_labels first")
    z = np.load(path)
    return {k: z[k] for k in z.files}


def _load_heat() -> dict:
    path = ROOT / "datasets" / "heat2d" / "heat2d_v0.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run scripts.generate_labels first")
    z = np.load(path)
    return {k: z[k] for k in z.files}


def eval_classical_burgers(data: dict, n_eval: int = 16) -> dict:
    u = data["u"][:n_eval]
    nu = data["nu"][:n_eval]
    x = data["x"]
    t = data["t"]
    nx = len(x)
    nt = len(t) - 1
    L = float(x[-1] + (x[1] - x[0]))  # periodic grid length
    T = float(t[-1])
    dx = float(x[1] - x[0])
    dt = float(t[1] - t[0])

    preds = []
    for i in range(n_eval):
        out = solve_burgers(nu=float(nu[i]), nx=nx, nt=nt, L=L, T=T, u0=u[i, 0])
        preds.append(out["u"])
    pred = np.stack(preds, axis=0)

    pred_metrics = batch_relative_l2(pred, u)
    conserve = audit_burgers(pred, dx=dx, dt=dt, nu=nu)

    # Counterfactual on first traj: double viscosity
    cf = burgers_counterfactual(
        u0=u[0, 0],
        nu_orig=float(nu[0]),
        nu_cf=float(nu[0]) * 2.0,
        nx=nx,
        nt=nt,
        L=L,
        T=T,
    )

    # Self-consistency: classical re-solve vs stored labels should be ~0
    return {
        "model": "classical_fd",
        "status": "ok",
        "n_eval": n_eval,
        "predict": {
            "rel_l2_mean": pred_metrics["mean"],
            "rel_l2_std": pred_metrics["std"],
            "nmse": pred_metrics["nmse"],
        },
        "conserve": conserve,
        "counterfactual": cf,
    }


def eval_classical_heat(data: dict, n_eval: int = 12) -> dict:
    u = data["u"][:n_eval]
    alpha = data["alpha"][:n_eval]
    x = data["x"]
    t = data["t"]
    n = len(x)
    nt = len(t) - 1
    L = float(x[-1])
    T = float(t[-1])
    dx = float(x[1] - x[0])
    dt = float(t[1] - t[0])

    preds = []
    for i in range(n_eval):
        out = solve_heat2d(alpha=float(alpha[i]), n=n, nt=nt, L=L, T=T, u0=u[i, 0])
        preds.append(out["u"])
    pred = np.stack(preds, axis=0)

    pred_metrics = batch_relative_l2(pred, u)
    conserve = audit_heat(pred, dx=dx, dt=dt, alpha=alpha)

    cf = heat_counterfactual(
        u0=u[0, 0],
        alpha_orig=float(alpha[0]),
        alpha_cf=float(alpha[0]) * 1.5,
        n=n,
        nt=nt,
        L=L,
        T=T,
    )

    return {
        "model": "classical_fd",
        "status": "ok",
        "n_eval": n_eval,
        "predict": {
            "rel_l2_mean": pred_metrics["mean"],
            "rel_l2_std": pred_metrics["std"],
            "nmse": pred_metrics["nmse"],
        },
        "conserve": conserve,
        "counterfactual": cf,
    }


def stub_section(name: str, mod) -> dict:
    return {
        "model": name,
        "status": getattr(mod, "STATUS", "not_trained"),
        "predict": None,
        "conserve": None,
        "counterfactual": None,
        "explain": None,
        "note": f"{name} Week-1 stub — not trained; no fabricated metrics.",
    }


def build_report(burgers_ex: dict, heat_ex: dict) -> dict:
    # Explain stub: score a canned classical rationale (not an LLM)
    rationale = (
        "The viscous Burgers equation balances nonlinear advection against "
        "viscosity-driven diffusion; energy dissipates and shocks smooth under "
        "sufficient viscosity. Finite-difference CFL stability and periodic "
        "boundaries matter for residual audits."
    )
    explain = score_explanation(rationale)

    return {
        "bench": "vu-bench-v0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "exams": {
            "predict": {
                "description": "Relative L2 / NMSE vs classical labels",
                "classical_burgers": burgers_ex["predict"],
                "classical_heat2d": heat_ex["predict"],
                "fno": {"status": fno_mod.STATUS},
                "pino": {"status": pino_mod.STATUS},
                "llm": {"status": llm_mod.STATUS},
            },
            "conserve": {
                "description": "Residual / energy-style audit",
                "classical_burgers": burgers_ex["conserve"],
                "classical_heat2d": heat_ex["conserve"],
                "fno": {"status": fno_mod.STATUS},
                "pino": {"status": pino_mod.STATUS},
            },
            "counterfactual": {
                "description": "Re-solve with changed param; traj delta",
                "classical_burgers": burgers_ex["counterfactual"],
                "classical_heat2d": heat_ex["counterfactual"],
                "fno": {"status": fno_mod.STATUS},
                "pino": {"status": pino_mod.STATUS},
            },
            "explain": {
                "description": "Keyword rubric stub over free-text rationale",
                "classical_keyword_stub": explain,
                "llm": {"status": llm_mod.STATUS},
            },
        },
        "baselines": {
            "classical": {"status": "ok", "burgers": burgers_ex, "heat2d": heat_ex},
            "fno": stub_section("fno", fno_mod),
            "pino": stub_section("pino", pino_mod),
            "llm": stub_section("llm", llm_mod),
        },
    }


def render_md(report: dict) -> str:
    ex = report["exams"]
    bp = ex["predict"]["classical_burgers"]
    hp = ex["predict"]["classical_heat2d"]
    bc = ex["conserve"]["classical_burgers"]
    hc = ex["conserve"]["classical_heat2d"]
    bcf = ex["counterfactual"]["classical_burgers"]
    hcf = ex["counterfactual"]["classical_heat2d"]
    expl = ex["explain"]["classical_keyword_stub"]

    lines = [
        "# VU-Bench v0 — latest eval",
        "",
        f"_Generated (UTC): {report['generated_at_utc']}_",
        "",
        "## Exam 1 — Predict",
        "",
        "| Baseline | Burgers rel-L2 mean | Heat2D rel-L2 mean | NMSE (B / H) | Status |",
        "|---|---:|---:|---:|---|",
        f"| classical_fd | {bp['rel_l2_mean']:.6e} | {hp['rel_l2_mean']:.6e} | "
        f"{bp['nmse']:.6e} / {hp['nmse']:.6e} | ok |",
        f"| FNO | — | — | — | {ex['predict']['fno']['status']} |",
        f"| PINO | — | — | — | {ex['predict']['pino']['status']} |",
        f"| LLM | — | — | — | {ex['predict']['llm']['status']} |",
        "",
        "## Exam 2 — Conserve",
        "",
        f"- Burgers residual_rel_l2: `{bc['residual_rel_l2']:.6e}`",
        f"- Burgers energy max |rel drift|: `{bc['energy_max_abs_rel_drift']:.6e}`",
        f"- Heat residual_rel_l2: `{hc['residual_rel_l2']:.6e}`",
        f"- Heat energy max |rel drift|: `{hc['energy_max_abs_rel_drift']:.6e}`",
        f"- FNO/PINO: `{ex['conserve']['fno']['status']}`",
        "",
        "## Exam 3 — Counterfactual",
        "",
        f"- Burgers ν→2ν traj Δ rel-L2: `{bcf['rel_l2_traj_delta']:.6e}` "
        f"(ν={bcf['nu_orig']:.4g}→{bcf['nu_cf']:.4g})",
        f"- Heat α→1.5α traj Δ rel-L2: `{hcf['rel_l2_traj_delta']:.6e}` "
        f"(α={hcf['alpha_orig']:.4g}→{hcf['alpha_cf']:.4g})",
        f"- FNO/PINO: `{ex['counterfactual']['fno']['status']}`",
        "",
        "## Exam 4 — Explain",
        "",
        f"- Keyword rubric overall: `{expl['overall']:.4f}` (status: {expl['status']})",
        f"- LLM: `{ex['explain']['llm']['status']}`",
        "",
        "## Notes",
        "",
        "- Classical numbers are from re-solving stored ICs (self-consistency / label check).",
        "- FNO, PINO, LLM are Week-1 stubs — marked `not_trained`; no fabricated SOTA.",
        "",
    ]
    return "\n".join(lines)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    raise TypeError(type(obj))


def main() -> None:
    burgers = _load_burgers()
    heat = _load_heat()
    print("Evaluating classical Burgers...")
    b_ex = eval_classical_burgers(burgers)
    print("Evaluating classical heat2d...")
    h_ex = eval_classical_heat(heat)
    report = build_report(b_ex, h_ex)

    out_dir = ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "latest.json"
    md_path = out_dir / "latest.md"
    json_path.write_text(
        json.dumps(report, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_md(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(
        "Burgers rel-L2 mean:",
        f"{b_ex['predict']['rel_l2_mean']:.6e}",
        "| Heat rel-L2 mean:",
        f"{h_ex['predict']['rel_l2_mean']:.6e}",
    )


if __name__ == "__main__":
    main()
