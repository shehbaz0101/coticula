"""Refresh reports/latest.* for v0.2 without retraining FNO/PINO.

Keeps measured Exam 1–3 / OOD cells from the existing JSON. Re-grades Exam 4
(gold / rule-based / metric-dump), rebuilds diagnostics, and rewrites branding.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics.diagnostics import build_diagnostics
from metrics.explain import (
    grade_exam4,
    grade_metric_dump_exam4,
    grade_rule_based_exam4,
    llm_judge_exam4,
    observations_from_report,
    score_explanation,
)
from metrics.trust import TRUST_POLICY
from scripts.run_eval import _json_default, render_md


def _load_labels() -> dict | None:
    b = ROOT / "datasets" / "burgers" / "burgers_v0.npz"
    h = ROOT / "datasets" / "heat2d" / "heat2d_v0.npz"
    if not b.exists() or not h.exists():
        return None
    import numpy as np

    zb = np.load(b)
    zh = np.load(h)
    return {
        "burgers": {k: zb[k] for k in zb.files},
        "heat": {k: zh[k] for k in zh.files},
    }


def refresh(path: Path | None = None) -> dict:
    path = path or (ROOT / "reports" / "latest.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    report["bench"] = "coticula-v0"
    report["project"] = "Coticula"
    report["brand"] = "Coticula"
    report["formerly"] = "Vermithor / VU-Bench"
    report["repo"] = "https://github.com/shehbaz0101/coticula"
    report["version"] = "0.2.0"
    WEEK3_IID_OOD_UTC = "2026-09-12T00:18:43.378079+00:00"
    if "iid_ood_measured_at_utc" not in report:
        prev = report.get("generated_at_utc") or ""
        report["iid_ood_measured_at_utc"] = (
            prev if prev.startswith("2026-09-12T00:18") else WEEK3_IID_OOD_UTC
        )
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["refresh_note"] = (
        "Exam 4 and diagnostics were re-measured on this revision. "
        "IID Exam 1–3 and OOD cells are copied from the prior measured run "
        "(see iid_ood_measured_at_utc). No FNO/PINO numbers were invented."
    )
    report["trust_policy"] = TRUST_POLICY
    report.pop("vu", None)
    report.pop("week", None)

    rationale = (
        "The viscous Burgers equation balances nonlinear advection against "
        "viscosity-driven diffusion; energy dissipates and shocks smooth under "
        "sufficient viscosity. Finite-difference CFL stability and periodic "
        "boundaries matter for residual audits."
    )
    exam4 = grade_exam4(use_gold=True)
    expl = report.setdefault("exams", {}).setdefault("explain", {})
    expl["description"] = (
        "Exam 4: fixed item set graded by expected law-keyword coverage. "
        "Gold = authored ceiling. rule_based = facet templates + measured "
        "observations. metric_dump = numbers only. LLM stays not_trained "
        "unless a real judge is wired."
    )
    expl["n_items"] = exam4["n_items"]
    expl["item_set"] = exam4["item_set"]
    expl["item_set_sha256"] = exam4["item_set_sha256"]
    expl["gold_reference"] = exam4
    expl["classical_keyword_stub"] = score_explanation(rationale)
    expl["llm"] = llm_judge_exam4()
    obs = observations_from_report(report)
    expl["observations"] = obs
    expl["rule_based"] = grade_rule_based_exam4(obs)
    expl["metric_dump"] = grade_metric_dump_exam4(obs)

    labels = _load_labels()
    report["diagnostics"] = build_diagnostics(report, labels=labels)

    out_dir = ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "latest.json"
    md_path = out_dir / "latest.md"
    diag_path = out_dir / "diagnostics.json"
    json_path.write_text(
        json.dumps(report, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_md(report), encoding="utf-8")
    diag_path.write_text(
        json.dumps(report["diagnostics"], indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {diag_path}")
    gold = expl["gold_reference"]["mean_coverage"]
    rb = expl["rule_based"]["mean_coverage"]
    dump = expl["metric_dump"]["mean_coverage"]
    print(f"Exam 4 gold={gold:.4f} rule_based={rb:.4f} metric_dump={dump:.4f}")
    return report


def main() -> None:
    refresh()


if __name__ == "__main__":
    main()
