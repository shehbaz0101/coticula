"""OOD / transfer probes for Vermithor VU-Bench (distinct from IID exams).

Usage:
    python -m scripts.run_ood
    python -m scripts.run_eval          # includes this module

Reports only measured numbers. Missing checkpoints stay ``not_trained``.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.fno.io import (
    DEFAULT_BURGERS,
    DEFAULT_HEAT,
    DEFAULT_PINO_BURGERS,
    DEFAULT_PINO_HEAT,
    load_checkpoint,
)
from metrics.ood import default_probe_plan, materialize_probe, score_burgers_surrogate, score_heat_surrogate
from metrics.trust import (
    BURGERS_TRAIN_SUPPORT,
    HEAT_TRAIN_SUPPORT,
    TRUST_POLICY,
    summarize_trust_flags,
)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    raise TypeError(type(obj))


def _torch_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _load_learned(name: str, burgers_ckpt: Path, heat_ckpt: Path) -> dict:
    try:
        import torch  # noqa: F401
    except ImportError:
        return {
            "model": name,
            "status": "not_trained",
            "reason": "torch_not_installed",
            "burgers": None,
            "heat2d": None,
        }
    device = _torch_device()
    loaded_b = load_checkpoint(burgers_ckpt, device=device)
    loaded_h = load_checkpoint(heat_ckpt, device=device)
    out = {
        "model": name,
        "device": device,
        "burgers": loaded_b if loaded_b.get("status") == "ok" else None,
        "heat2d": loaded_h if loaded_h.get("status") == "ok" else None,
        "checkpoints": {
            "burgers": loaded_b.get("path"),
            "heat2d": loaded_h.get("path"),
        },
        "reasons": {
            "burgers": None if loaded_b.get("status") == "ok" else loaded_b.get("reason"),
            "heat2d": None if loaded_h.get("status") == "ok" else loaded_h.get("reason"),
        },
    }
    out["status"] = (
        "ok"
        if (out["burgers"] is not None or out["heat2d"] is not None)
        else "not_trained"
    )
    return out


def _predict_burgers(model, u0: np.ndarray, nu: np.ndarray, device) -> np.ndarray:
    import torch

    with torch.no_grad():
        u0_t = torch.from_numpy(np.asarray(u0, dtype=np.float32)).to(device)
        nu_t = torch.from_numpy(np.asarray(nu, dtype=np.float32)).to(device)
        return model(u0_t, nu_t).cpu().numpy()


def _predict_heat(model, u0: np.ndarray, alpha: np.ndarray, device) -> np.ndarray:
    import torch

    with torch.no_grad():
        u0_t = torch.from_numpy(np.asarray(u0, dtype=np.float32)).to(device)
        a_t = torch.from_numpy(np.asarray(alpha, dtype=np.float32)).to(device)
        return model(u0_t, a_t).cpu().numpy()


def _score_learned(learned: dict, cases: dict) -> dict:
    pde = cases["pde"]
    slot = "burgers" if pde == "burgers1d" else "heat2d"
    payload = learned.get(slot)
    if learned.get("status") == "not_trained" or payload is None:
        reason = (learned.get("reasons") or {}).get(slot) or learned.get("reason") or "not_trained"
        return {
            "status": "not_trained",
            "reason": reason,
            "model": learned.get("model"),
        }
    model = payload["model"]
    device = learned.get("device") or payload.get("device") or "cpu"
    model.eval()
    try:
        if pde == "burgers1d":
            pred = _predict_burgers(model, cases["u0"], cases["nu"], device)
            scored = score_burgers_surrogate(pred, cases)
        else:
            pred = _predict_heat(model, cases["u0"], cases["alpha"], device)
            scored = score_heat_surrogate(pred, cases)
    except Exception as exc:  # resolution / shape mismatch — fail closed, no fake score
        return {
            "status": "untrusted",
            "reason": f"forward_failed: {type(exc).__name__}: {exc}",
            "model": learned.get("model"),
            "trust": {
                "status": "untrusted",
                "ood": True,
                "untrusted": True,
                "reasons": ["forward_failed"],
                "banner": (
                    "UNTRUSTED (forward_failed): operator could not score this "
                    "OOD grid — do not present a silent pretty heatmap"
                ),
            },
        }
    scored["status"] = "ok"
    scored["model"] = learned.get("model")
    return scored


def _public_case_meta(cases: dict) -> dict:
    spec = cases.get("spec") or {}
    meta = {
        "id": spec.get("id"),
        "pde": spec.get("pde") or ("burgers" if cases["pde"] == "burgers1d" else "heat2d"),
        "probe": spec.get("probe"),
        "n": spec.get("n") or int(cases["u"].shape[0]),
        "ic_family": cases["ic_family"],
        "param": float(cases["param"]),
    }
    if cases["pde"] == "burgers1d":
        meta["nx"] = int(cases["nx"])
        meta["nt"] = int(cases["nt"])
        meta["nu"] = float(cases["param"])
    else:
        meta["n_grid"] = int(cases["n_grid"])
        meta["nt"] = int(cases["nt"])
        meta["alpha"] = float(cases["param"])
    return meta


def evaluate_ood(
    *,
    n_param: int = 4,
    n_res: int = 2,
    n_ic: int = 2,
    fno: dict | None = None,
    pino: dict | None = None,
) -> dict:
    """Run OOD probes. Returns a JSON-serializable report section."""
    if fno is None:
        fno = _load_learned("fno", DEFAULT_BURGERS, DEFAULT_HEAT)
    if pino is None:
        pino = _load_learned("pino", DEFAULT_PINO_BURGERS, DEFAULT_PINO_HEAT)

    plan = default_probe_plan(n_param=n_param, n_res=n_res, n_ic=n_ic)
    probes: list[dict] = []
    trust_rows: list[dict] = []
    for spec in plan:
        print(f"OOD probe {spec['id']}...")
        cases = materialize_probe(spec)
        row = {
            "meta": _public_case_meta(cases),
            "classical": {"status": "ok", "role": "labeler", **cases["labeler"]},
            "fno": _score_learned(fno, cases),
            "pino": _score_learned(pino, cases),
        }
        probes.append(row)
        for key in ("classical", "fno", "pino"):
            trust_rows.append(row[key])

    by_probe: dict[str, list[dict]] = {}
    for row in probes:
        by_probe.setdefault(row["meta"]["probe"], []).append(row)

    return {
        "description": (
            "Transfer probes distinct from IID exams 1–3. Ground truth is a "
            "fresh classical FD solve. Learned rows are fail-closed (ood / untrusted)."
        ),
        "train_support": {
            "burgers": {
                **BURGERS_TRAIN_SUPPORT,
                "param_range": list(BURGERS_TRAIN_SUPPORT["param_range"]),
            },
            "heat2d": {
                **HEAT_TRAIN_SUPPORT,
                "param_range": list(HEAT_TRAIN_SUPPORT["param_range"]),
            },
        },
        "trust_policy": TRUST_POLICY,
        "resolution_note": (
            "FNO spatial conv is resolution-agnostic; time is locked as output "
            "channels (Burgers nt=80, heat nt=40). No temporal transfer is claimed."
        ),
        "baselines": {
            "fno": {
                "status": fno.get("status"),
                "checkpoints": fno.get("checkpoints"),
                "reasons": fno.get("reasons"),
            },
            "pino": {
                "status": pino.get("status"),
                "checkpoints": pino.get("checkpoints"),
                "reasons": pino.get("reasons"),
            },
        },
        "probes": probes,
        "by_probe": {k: [r["meta"]["id"] for r in v] for k, v in by_probe.items()},
        "trust_summary": summarize_trust_flags(trust_rows),
    }


def build_failure_analysis(
    burgers_ex: dict,
    heat_ex: dict,
    fno_ex: dict,
    pino_ex: dict,
    ood: dict | None = None,
) -> dict:
    """Qualitative comparisons from *measured* IID / OOD numbers only."""
    notes: list[dict] = []

    fno_b = (fno_ex or {}).get("burgers") or {}
    pino_b = (pino_ex or {}).get("burgers") or {}
    if fno_b.get("status") == "ok" and pino_b.get("status") == "ok":
        fno_l2 = float(fno_b["predict"]["rel_l2_mean"])
        pino_l2 = float(pino_b["predict"]["rel_l2_mean"])
        fno_r = float(fno_b["conserve"]["residual_rel_l2"])
        pino_r = float(pino_b["conserve"]["residual_rel_l2"])
        if fno_l2 < pino_l2 and fno_r > pino_r:
            observation = (
                "On IID Burgers, FNO has a lower Predict rel-L2 than PINO but a "
                "larger FD residual. A silent heatmap of the FNO field would look "
                "'closer' to the label while Exam 2 and the fail-closed trust flag "
                "show a worse PDE violation. PINO's residual term reduces that "
                "violation at a modest L2 cost — not a SOTA claim, a measured tradeoff."
            )
        else:
            observation = (
                "IID Burgers FNO vs PINO (measured Predict rel-L2 and Exam 2 residual). "
                "Compare residual vs L2 rather than ranking a single number."
            )
        notes.append(
            {
                "title": "FNO vs PINO on Burgers (IID): L2 vs residual",
                "kind": "residual_vs_l2",
                "split": "iid",
                "observation": observation,
                "measured": {
                    "fno_rel_l2": fno_l2,
                    "fno_residual_rel_l2": fno_r,
                    "pino_rel_l2": pino_l2,
                    "pino_residual_rel_l2": pino_r,
                    "fno_trust": (fno_b.get("trust") or {}).get("status"),
                    "pino_trust": (pino_b.get("trust") or {}).get("status"),
                },
            }
        )

    fno_h = (fno_ex or {}).get("heat2d") or {}
    if fno_h.get("status") == "ok":
        cf = fno_h.get("counterfactual") or {}
        model_delta = cf.get("rel_l2_traj_delta_model")
        class_delta = cf.get("rel_l2_traj_delta_classical")
        vs = cf.get("rel_l2_vs_classical_cf")
        if (
            model_delta is not None
            and class_delta is not None
            and vs is not None
            and float(model_delta) < 0.25 * float(class_delta)
        ):
            notes.append(
                {
                    "title": "FNO heat2d counterfactual insensitivity",
                    "kind": "counterfactual",
                    "split": "iid_exam3",
                    "observation": (
                        "Classical heat at α′ differs from α by a large trajectory Δ, "
                        "but the FNO field barely moves when α is scaled. Exam 3 vs "
                        "the classical solve at α′ is therefore large — a failure of "
                        "coefficient response, not just interpolation error."
                    ),
                    "measured": {
                        "alpha_orig": cf.get("alpha_orig"),
                        "alpha_cf": cf.get("alpha_cf"),
                        "rel_l2_traj_delta_classical": float(class_delta),
                        "rel_l2_traj_delta_model": float(model_delta),
                        "rel_l2_vs_classical_cf": float(vs),
                    },
                }
            )

    if ood:
        # First OOD param-shift Burgers row with both FNO and PINO measured.
        for row in ood.get("probes") or []:
            if row.get("meta", {}).get("id") != "burgers_param_below":
                continue
            fno_row = row.get("fno") or {}
            pino_row = row.get("pino") or {}
            if fno_row.get("status") == "ok" and pino_row.get("status") == "ok":
                notes.append(
                    {
                        "title": "OOD Burgers ν below train range",
                        "kind": "ood_param_shift",
                        "split": "ood",
                        "observation": (
                            "ν=0.001 is below the v0 train range [0.005, 0.05]. "
                            "Numbers below are transfer scores vs a fresh classical "
                            "solve — not IID Exam 1. Trust flags are fail-closed."
                        ),
                        "measured": {
                            "nu": row["meta"]["param"],
                            "fno_rel_l2": fno_row["predict"]["rel_l2_mean"],
                            "fno_residual_rel_l2": fno_row["conserve"]["residual_rel_l2"],
                            "fno_trust": (fno_row.get("trust") or {}).get("status"),
                            "pino_rel_l2": pino_row["predict"]["rel_l2_mean"],
                            "pino_residual_rel_l2": pino_row["conserve"]["residual_rel_l2"],
                            "pino_trust": (pino_row.get("trust") or {}).get("status"),
                        },
                    }
                )
            break

    return {
        "description": (
            "Qualitative comparisons using only measured fields from this run. "
            "No fabricated metrics."
        ),
        "notes": notes,
    }


def main() -> None:
    ood = evaluate_ood()
    out = {
        "bench": "vu-bench-v0",
        "project": "Vermithor",
        "vu": "Vermithor Understanding Bench",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ood": ood,
    }
    out_dir = ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ood.json"
    path.write_text(json.dumps(out, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(f"Wrote {path}")
    print("trust_summary:", ood["trust_summary"])


if __name__ == "__main__":
    main()
