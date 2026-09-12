"""Standalone v0.2 diagnostics (scatter from latest.json + classical label sweeps).

Usage::

    python -m scripts.run_diagnostics
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
from scripts.run_eval import _json_default


def main() -> None:
    report_path = ROOT / "reports" / "latest.json"
    report: dict = {}
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    labels = None
    b = ROOT / "datasets" / "burgers" / "burgers_v0.npz"
    h = ROOT / "datasets" / "heat2d" / "heat2d_v0.npz"
    if b.exists() and h.exists():
        import numpy as np

        zb = np.load(b)
        zh = np.load(h)
        labels = {
            "burgers": {k: zb[k] for k in zb.files},
            "heat": {k: zh[k] for k in zh.files},
        }
    diag = build_diagnostics(report, labels=labels)
    out = {
        "bench": "coticula-v0",
        "project": "Coticula",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostics": diag,
    }
    path = ROOT / "reports" / "diagnostics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(f"Wrote {path}")
    scatter = diag.get("residual_vs_l2") or {}
    print("scatter n=", scatter.get("n"), "pearson=", scatter.get("pearson_residual_vs_l2"))


if __name__ == "__main__":
    main()
