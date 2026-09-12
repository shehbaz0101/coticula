"""Run VU-Bench exams: classical FD + trained FNO/PINO (if checkpoints exist).

Writes reports/latest.json and reports/latest.md with **only measured numbers**.
Missing checkpoints stay `not_trained`. Exam 4 is a keyword rubric; the LLM
hook is not scored unless a real judge is wired (never fabricated).

Week 3: OOD / transfer probes (param, resolution, IC family) are a separate
section from IID exams 1–3. Fail-closed trust flags (`ood` / `untrusted`)
travel with the numbers. Use ``--skip-ood`` for IID-only.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines import llm as llm_mod
from baselines.classical.burgers1d import solve_burgers
from baselines.classical.heat2d import solve_heat2d
from baselines.fno.io import (
    DEFAULT_BURGERS,
    DEFAULT_HEAT,
    DEFAULT_PINO_BURGERS,
    DEFAULT_PINO_HEAT,
    load_checkpoint,
)
from metrics.conserve import audit_burgers, audit_heat
from metrics.counterfactual import burgers_counterfactual, heat_counterfactual
from metrics.explain import score_explanation
from metrics.predict import batch_relative_l2
from metrics.trust import (
    BURGERS_TRAIN_SUPPORT,
    HEAT_TRAIN_SUPPORT,
    TRUST_POLICY,
    attach_trust,
)
from scripts.run_ood import build_failure_analysis, evaluate_ood


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

    cf = burgers_counterfactual(
        u0=u[0, 0],
        nu_orig=float(nu[0]),
        nu_cf=float(nu[0]) * 2.0,
        nx=nx,
        nt=nt,
        L=L,
        T=T,
    )

    block = {
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
    return attach_trust(
        block,
        support=BURGERS_TRAIN_SUPPORT,
        param=float(np.mean(nu)),
        resolution=nx,
        ic_family=BURGERS_TRAIN_SUPPORT["ic_family"],
        role="labeler",
    )


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

    block = {
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
    return attach_trust(
        block,
        support=HEAT_TRAIN_SUPPORT,
        param=float(np.mean(alpha)),
        resolution=n,
        ic_family=HEAT_TRAIN_SUPPORT["ic_family"],
        role="labeler",
    )


def _not_trained(name: str, reason: str, path: str | None = None) -> dict:
    return {
        "model": name,
        "status": "not_trained",
        "predict": None,
        "conserve": None,
        "counterfactual": None,
        "reason": reason,
        "checkpoint": path,
        "note": f"{name} checkpoint missing or unloadable; no fabricated metrics.",
    }


def _torch_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _eval_learned(
    name: str,
    burgers_ckpt: Path,
    heat_ckpt: Path,
    burgers: dict,
    heat: dict,
    n_eval_b: int,
    n_eval_h: int,
) -> dict:
    try:
        import torch  # noqa: F401
    except ImportError:
        return _not_trained(name, "torch_not_installed")

    from baselines.fno.eval_exams import eval_burgers_operator, eval_heat_operator

    device = _torch_device()
    loaded_b = load_checkpoint(burgers_ckpt, device=device)
    loaded_h = load_checkpoint(heat_ckpt, device=device)

    out: dict = {
        "model": name,
        "status": "not_trained",
        "burgers": None,
        "heat2d": None,
        "checkpoints": {
            "burgers": loaded_b.get("path"),
            "heat2d": loaded_h.get("path"),
        },
    }
    if loaded_b["status"] == "ok":
        split = (loaded_b.get("extra") or {}).get("split") or {}
        idx = split.get("eval_idx")
        print(f"Evaluating {name} Burgers ({burgers_ckpt})...")
        out["burgers"] = eval_burgers_operator(
            loaded_b["model"],
            burgers,
            idx=idx,
            n_eval=n_eval_b,
            device=device,
            model_name=name,
        )
        nu_eval = burgers["nu"][np.asarray(out["burgers"]["eval_idx"], dtype=int)]
        out["burgers"] = attach_trust(
            out["burgers"],
            support=BURGERS_TRAIN_SUPPORT,
            param=float(np.mean(nu_eval)),
            resolution=int(len(burgers["x"])),
            ic_family=BURGERS_TRAIN_SUPPORT["ic_family"],
            role="surrogate",
        )
        out["burgers"]["training"] = {
            k: loaded_b["training"].get(k)
            for k in (
                "objective",
                "epochs",
                "lambda_pde",
                "lambda_ic",
                "final_val_rel_l2",
            )
            if k in loaded_b["training"]
        }
    else:
        out["burgers"] = {
            "status": "not_trained",
            "reason": loaded_b.get("reason"),
            "checkpoint": loaded_b.get("path"),
        }

    if loaded_h["status"] == "ok":
        split = (loaded_h.get("extra") or {}).get("split") or {}
        idx = split.get("eval_idx")
        print(f"Evaluating {name} heat2d ({heat_ckpt})...")
        out["heat2d"] = eval_heat_operator(
            loaded_h["model"],
            heat,
            idx=idx,
            n_eval=n_eval_h,
            device=device,
            model_name=name,
        )
        a_eval = heat["alpha"][np.asarray(out["heat2d"]["eval_idx"], dtype=int)]
        out["heat2d"] = attach_trust(
            out["heat2d"],
            support=HEAT_TRAIN_SUPPORT,
            param=float(np.mean(a_eval)),
            resolution=int(len(heat["x"])),
            ic_family=HEAT_TRAIN_SUPPORT["ic_family"],
            role="surrogate",
        )
        out["heat2d"]["training"] = {
            k: loaded_h["training"].get(k)
            for k in (
                "objective",
                "epochs",
                "lambda_pde",
                "lambda_ic",
                "final_val_rel_l2",
            )
            if k in loaded_h["training"]
        }
    else:
        out["heat2d"] = {
            "status": "not_trained",
            "reason": loaded_h.get("reason"),
            "checkpoint": loaded_h.get("path"),
        }

    if (
        out["burgers"].get("status") == "ok"
        or out["heat2d"].get("status") == "ok"
    ):
        out["status"] = "ok"
    return out


def llm_explain_section() -> dict:
    """Optional LLM hook — stubbed. Never invents a judge score."""
    key = os.environ.get("VU_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        return {
            "status": getattr(llm_mod, "STATUS", "not_trained"),
            "note": "No API key (VU_LLM_API_KEY / OPENAI_API_KEY); keyword rubric only. No fabricated LLM scores.",
        }
    return {
        "status": "not_trained",
        "note": "API key present but LLM judge is not wired; refusing to fabricate scores.",
    }


def _exam_predict_block(section: dict) -> dict:
    if section.get("status") != "ok":
        return {"status": section.get("status", "not_trained")}
    block: dict = {"status": "ok"}
    for pde in ("burgers", "heat2d"):
        sub = section.get(pde) or {}
        if sub.get("status") == "ok":
            block[pde] = {"status": "ok", **sub["predict"]}
        else:
            block[pde] = {"status": sub.get("status", "not_trained")}
    return block


def _exam_conserve_block(section: dict) -> dict:
    if section.get("status") != "ok":
        return {"status": section.get("status", "not_trained")}
    block: dict = {"status": "ok"}
    for pde in ("burgers", "heat2d"):
        sub = section.get(pde) or {}
        if sub.get("status") == "ok":
            block[pde] = {"status": "ok", **sub["conserve"]}
        else:
            block[pde] = {"status": sub.get("status", "not_trained")}
    return block


def _exam_cf_block(section: dict) -> dict:
    if section.get("status") != "ok":
        return {"status": section.get("status", "not_trained")}
    block: dict = {"status": "ok"}
    for pde in ("burgers", "heat2d"):
        sub = section.get(pde) or {}
        if sub.get("status") == "ok":
            block[pde] = {"status": "ok", **sub["counterfactual"]}
        else:
            block[pde] = {"status": sub.get("status", "not_trained")}
    return block


def _trust_status(section: dict | None) -> str:
    if not section or section.get("status") == "not_trained":
        return "not_trained"
    trust = section.get("trust") or {}
    return str(trust.get("status") or section.get("status") or "not_trained")


def build_report(
    burgers_ex: dict,
    heat_ex: dict,
    fno_ex: dict,
    pino_ex: dict,
    ood: dict | None = None,
    failure_analysis: dict | None = None,
) -> dict:
    rationale = (
        "The viscous Burgers equation balances nonlinear advection against "
        "viscosity-driven diffusion; energy dissipates and shocks smooth under "
        "sufficient viscosity. Finite-difference CFL stability and periodic "
        "boundaries matter for residual audits."
    )
    explain = score_explanation(rationale)
    llm_block = llm_explain_section()

    return {
        "bench": "vu-bench-v0",
        "project": "Vermithor",
        "vu": "Vermithor Understanding Bench",
        "repo": "https://github.com/shehbaz0101/vermithor",
        "week": 3,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "trust_policy": TRUST_POLICY,
        "exams": {
            "predict": {
                "description": "Relative L2 / NMSE vs classical labels",
                "classical_burgers": burgers_ex["predict"],
                "classical_heat2d": heat_ex["predict"],
                "fno": _exam_predict_block(fno_ex),
                "pino": _exam_predict_block(pino_ex),
                "llm": {"status": llm_block["status"]},
            },
            "conserve": {
                "description": "Residual / energy-style audit",
                "classical_burgers": burgers_ex["conserve"],
                "classical_heat2d": heat_ex["conserve"],
                "fno": _exam_conserve_block(fno_ex),
                "pino": _exam_conserve_block(pino_ex),
            },
            "counterfactual": {
                "description": "Changed coefficient; FNO/PINO graded vs classical solver",
                "classical_burgers": burgers_ex["counterfactual"],
                "classical_heat2d": heat_ex["counterfactual"],
                "fno": _exam_cf_block(fno_ex),
                "pino": _exam_cf_block(pino_ex),
            },
            "explain": {
                "description": "Keyword rubric stub over free-text rationale",
                "classical_keyword_stub": explain,
                "llm": llm_block,
            },
        },
        "baselines": {
            "classical": {"status": "ok", "burgers": burgers_ex, "heat2d": heat_ex},
            "fno": fno_ex,
            "pino": pino_ex,
            "llm": llm_block,
        },
        "ood": ood,
        "failure_analysis": failure_analysis,
    }


def _fmt_pred(block: dict | None, pde: str) -> str:
    if not block or block.get("status") != "ok":
        return "—"
    sub = block.get(pde) or {}
    if sub.get("status") != "ok" or "rel_l2_mean" not in sub:
        return "—"
    return f"{sub['rel_l2_mean']:.6e}"


def _fmt_nmse(block: dict | None, pde: str) -> str:
    if not block or block.get("status") != "ok":
        return "—"
    sub = block.get(pde) or {}
    if sub.get("status") != "ok" or "nmse" not in sub:
        return "—"
    return f"{sub['nmse']:.6e}"


def _status_cell(block: dict | None) -> str:
    if not block:
        return "not_trained"
    if block.get("status") == "ok":
        parts = []
        for pde in ("burgers", "heat2d"):
            sub = block.get(pde) or {}
            parts.append(f"{pde}={sub.get('status', 'not_trained')}")
        return "ok (" + ", ".join(parts) + ")"
    return str(block.get("status", "not_trained"))


def render_md(report: dict) -> str:
    ex = report["exams"]
    bp = ex["predict"]["classical_burgers"]
    hp = ex["predict"]["classical_heat2d"]
    bc = ex["conserve"]["classical_burgers"]
    hc = ex["conserve"]["classical_heat2d"]
    bcf = ex["counterfactual"]["classical_burgers"]
    hcf = ex["counterfactual"]["classical_heat2d"]
    expl = ex["explain"]["classical_keyword_stub"]
    fno_p = ex["predict"]["fno"]
    pino_p = ex["predict"]["pino"]
    fno_c = ex["conserve"]["fno"]
    pino_c = ex["conserve"]["pino"]
    fno_cf = ex["counterfactual"]["fno"]
    pino_cf = ex["counterfactual"]["pino"]

    def _cons_line(block: dict, pde: str, label: str) -> str:
        sub = (block or {}).get(pde) or {}
        if sub.get("status") != "ok" or "residual_rel_l2" not in sub:
            return f"- {label}: `not_trained`"
        return (
            f"- {label} residual_rel_l2: `{sub['residual_rel_l2']:.6e}`; "
            f"energy max |rel drift|: `{sub['energy_max_abs_rel_drift']:.6e}`"
        )

    def _cf_line(block: dict, pde: str, label: str) -> str:
        sub = (block or {}).get(pde) or {}
        if sub.get("status") != "ok" or "rel_l2_vs_classical_cf" not in sub:
            return f"- {label}: `not_trained`"
        return (
            f"- {label} vs classical at θ′ rel-L2: `{sub['rel_l2_vs_classical_cf']:.6e}`"
        )

    lines = [
        "# Vermithor VU-Bench v0 — latest eval (Week 3)",
        "",
        f"_Generated (UTC): {report['generated_at_utc']}_",
        "",
        "**VU = Vermithor Understanding Bench** (not the `uv` Python packager).",
        "Project: **Vermithor**. Repo: https://github.com/shehbaz0101/vermithor",
        "",
        "## Exam 1 — Predict",
        "",
        "| Baseline | Burgers rel-L2 mean | Heat2D rel-L2 mean | NMSE (B / H) | Status |",
        "|---|---:|---:|---:|---|",
        f"| classical_fd | {bp['rel_l2_mean']:.6e} | {hp['rel_l2_mean']:.6e} | "
        f"{bp['nmse']:.6e} / {hp['nmse']:.6e} | ok |",
        f"| FNO | {_fmt_pred(fno_p, 'burgers')} | {_fmt_pred(fno_p, 'heat2d')} | "
        f"{_fmt_nmse(fno_p, 'burgers')} / {_fmt_nmse(fno_p, 'heat2d')} | {_status_cell(fno_p)} |",
        f"| PINO | {_fmt_pred(pino_p, 'burgers')} | {_fmt_pred(pino_p, 'heat2d')} | "
        f"{_fmt_nmse(pino_p, 'burgers')} / {_fmt_nmse(pino_p, 'heat2d')} | {_status_cell(pino_p)} |",
        f"| LLM | — | — | — | {ex['predict']['llm']['status']} |",
        "",
        "## Exam 2 — Conserve",
        "",
        f"- classical Burgers residual_rel_l2: `{bc['residual_rel_l2']:.6e}`",
        f"- classical Burgers energy max |rel drift|: `{bc['energy_max_abs_rel_drift']:.6e}`",
        f"- classical Heat residual_rel_l2: `{hc['residual_rel_l2']:.6e}`",
        f"- classical Heat energy max |rel drift|: `{hc['energy_max_abs_rel_drift']:.6e}`",
        _cons_line(fno_c, "burgers", "FNO Burgers"),
        _cons_line(fno_c, "heat2d", "FNO Heat"),
        _cons_line(pino_c, "burgers", "PINO Burgers"),
        _cons_line(pino_c, "heat2d", "PINO Heat"),
        "",
        "## Exam 3 — Counterfactual",
        "",
        f"- classical Burgers ν→2ν traj Δ rel-L2: `{bcf['rel_l2_traj_delta']:.6e}` "
        f"(ν={bcf['nu_orig']:.4g}→{bcf['nu_cf']:.4g})",
        f"- classical Heat α→1.5α traj Δ rel-L2: `{hcf['rel_l2_traj_delta']:.6e}` "
        f"(α={hcf['alpha_orig']:.4g}→{hcf['alpha_cf']:.4g})",
        _cf_line(fno_cf, "burgers", "FNO Burgers"),
        _cf_line(fno_cf, "heat2d", "FNO Heat"),
        _cf_line(pino_cf, "burgers", "PINO Burgers"),
        _cf_line(pino_cf, "heat2d", "PINO Heat"),
        "",
        "## Trust (fail-closed)",
        "",
        f"- Policy: `{TRUST_POLICY['name']}` — residual > {TRUST_POLICY['residual_untrusted']} or "
        f"Predict rel-L2 > {TRUST_POLICY['predict_untrusted']} → `untrusted`; "
        "outside train support → `ood`.",
        f"- classical Burgers: `{_trust_status(report['baselines']['classical']['burgers'])}`",
        f"- classical Heat: `{_trust_status(report['baselines']['classical']['heat2d'])}`",
        f"- FNO Burgers: `{_trust_status((report['baselines']['fno'] or {}).get('burgers'))}`",
        f"- FNO Heat: `{_trust_status((report['baselines']['fno'] or {}).get('heat2d'))}`",
        f"- PINO Burgers: `{_trust_status((report['baselines']['pino'] or {}).get('burgers'))}`",
        f"- PINO Heat: `{_trust_status((report['baselines']['pino'] or {}).get('heat2d'))}`",
        "- OOD / untrusted fields must not be shown as silent pretty heatmaps "
        "(see `metrics.trust.refuse_silent_heatmap`).",
        "",
        "## Exam 4 — Explain",
        "",
        f"- Keyword rubric overall: `{expl['overall']:.4f}` (status: {expl['status']})",
        f"- LLM: `{ex['explain']['llm']['status']}` — {ex['explain']['llm'].get('note', '')}",
        "",
    ]
    lines.extend(_render_ood_md(report.get("ood")))
    lines.extend(_render_failure_md(report.get("failure_analysis")))
    lines.extend(
        [
            "## Notes",
            "",
            "- Classical IID numbers are from re-solving stored ICs (self-consistency / label check).",
            "- OOD numbers are a separate section: fresh classical solves outside train support.",
            "- FNO / PINO numbers appear only when a checkpoint loads; otherwise `not_trained`.",
            "- No fabricated SOTA. Exam 4 is a keyword stub, not a trained judge.",
            "- VU = Vermithor Understanding Bench. Non-goals: no chip cooling, no AU-scale FM.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt_trust_summary(summary: dict | None) -> str:
    if not summary:
        return "—"
    counts = summary.get("counts") or {}
    n = summary.get("n_scored", 0)
    parts = [f"{k}={v}" for k, v in counts.items() if v]
    return f"n={n} ({', '.join(parts)})"


def _ood_cell(block: dict | None) -> str:
    if not block:
        return "—"
    if block.get("status") != "ok" or "predict" not in block:
        return str(block.get("status", "not_trained"))
    trust = (block.get("trust") or {}).get("status", "?")
    l2 = block["predict"]["rel_l2_mean"]
    res = block["conserve"]["residual_rel_l2"]
    return f"{l2:.3e} / {res:.3e} [{trust}]"


def _render_ood_md(ood: dict | None) -> list[str]:
    if not ood:
        return [
            "## OOD / transfer (distinct from IID)",
            "",
            "- skipped (`--skip-ood` or OOD runner not called)",
            "",
        ]
    lines = [
        "## OOD / transfer (distinct from IID)",
        "",
        ood.get("description", ""),
        "",
        f"- Resolution note: {ood.get('resolution_note', '')}",
        f"- Trust rollup: `{_fmt_trust_summary(ood.get('trust_summary'))}`",
        "",
        "| Probe | param / grid / IC | classical residual | FNO L2 / residual [trust] | PINO L2 / residual [trust] |",
        "|---|---|---:|---|---|",
    ]
    for row in ood.get("probes") or []:
        meta = row.get("meta") or {}
        if meta.get("pde") == "burgers":
            ident = f"ν={meta.get('nu'):.4g}, nx={meta.get('nx')}, IC={meta.get('ic_family')}"
        else:
            ident = (
                f"α={meta.get('alpha'):.4g}, n={meta.get('n_grid')}, "
                f"IC={meta.get('ic_family')}"
            )
        cl = row.get("classical") or {}
        cl_res = "—"
        if cl.get("status") == "ok":
            cl_res = f"{cl['conserve']['residual_rel_l2']:.3e}"
        lines.append(
            f"| {meta.get('id')} | {ident} | {cl_res} | "
            f"{_ood_cell(row.get('fno'))} | {_ood_cell(row.get('pino'))} |"
        )
    lines.append("")
    return lines


def _render_failure_md(fa: dict | None) -> list[str]:
    if not fa or not fa.get("notes"):
        return [
            "## Failure analysis",
            "",
            "- No measured comparison available this run (missing checkpoint or OOD skip).",
            "",
        ]
    lines = [
        "## Failure analysis",
        "",
        fa.get("description", ""),
        "",
    ]
    for note in fa["notes"]:
        lines.append(f"### {note['title']}")
        lines.append("")
        lines.append(note["observation"])
        lines.append("")
        meas = note.get("measured") or {}
        for k, v in meas.items():
            if isinstance(v, float):
                lines.append(f"- `{k}`: `{v:.6e}`")
            else:
                lines.append(f"- `{k}`: `{v}`")
        lines.append("")
    return lines


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    raise TypeError(type(obj))


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="VU-Bench IID exams + optional OOD probes")
    p.add_argument(
        "--skip-ood",
        action="store_true",
        help="IID exams only (no param/resolution/IC transfer probes)",
    )
    args = p.parse_args(argv)

    burgers = _load_burgers()
    heat = _load_heat()
    print("Evaluating classical Burgers...")
    b_ex = eval_classical_burgers(burgers)
    print("Evaluating classical heat2d...")
    h_ex = eval_classical_heat(heat)

    fno_ex = _eval_learned(
        "fno",
        DEFAULT_BURGERS,
        DEFAULT_HEAT,
        burgers,
        heat,
        n_eval_b=16,
        n_eval_h=12,
    )
    pino_ex = _eval_learned(
        "pino",
        DEFAULT_PINO_BURGERS,
        DEFAULT_PINO_HEAT,
        burgers,
        heat,
        n_eval_b=16,
        n_eval_h=12,
    )
    ood = None
    if not args.skip_ood:
        print("Evaluating OOD / transfer probes...")
        ood = evaluate_ood()
    fa = build_failure_analysis(b_ex, h_ex, fno_ex, pino_ex, ood)
    report = build_report(b_ex, h_ex, fno_ex, pino_ex, ood=ood, failure_analysis=fa)

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
    if fno_ex.get("burgers", {}).get("status") == "ok":
        print(
            "FNO Burgers rel-L2 mean:",
            f"{fno_ex['burgers']['predict']['rel_l2_mean']:.6e}",
        )
    else:
        print("FNO Burgers: not_trained")


if __name__ == "__main__":
    main()
