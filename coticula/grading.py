"""Score a surrogate on Predict, Conserve, Counterfactual, and one OOD probe.

Exam 4 is graded from the measured numbers of this run. Missing checkpoints
stay ``not_trained``. Nothing here invents a metric.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from coticula.adapter import (
    ClassicalFD,
    TorchOperatorAdapter,
    persist_example,
    repo_root,
)
from metrics.conserve import audit_burgers, audit_heat
from metrics.counterfactual import grade_burgers_cf_prediction, grade_heat_cf_prediction
from metrics.explain import (
    grade_exam4,
    grade_metric_dump_exam4,
    grade_rule_based_exam4,
    llm_judge_exam4,
    observations_from_report,
)
from metrics.ood import (
    BURGERS_OOD_NU,
    HEAT_OOD_ALPHA,
    generate_burgers_cases,
    generate_heat_cases,
    score_burgers_surrogate,
    score_heat_surrogate,
)
from metrics.predict import batch_relative_l2, relative_l2
from metrics.trust import (
    BURGERS_TRAIN_SUPPORT,
    HEAT_TRAIN_SUPPORT,
    PREDICT_UNTRUSTED,
    RESIDUAL_UNTRUSTED,
    TRUST_POLICY,
    attach_trust,
)

ROOT = repo_root()


def _not_trained(name: str, reason: str, checkpoint: str | None = None) -> dict:
    return {
        "name": name,
        "status": "not_trained",
        "reason": reason,
        "checkpoint": checkpoint,
        "role": "surrogate",
        "predict": None,
        "conserve": None,
        "counterfactual": None,
        "trust": None,
    }


def _public_metrics(block: dict) -> dict:
    """Drop arrays so the scorecard JSON stays small."""
    pred = block.get("predict") or {}
    cons = block.get("conserve") or {}
    cf = dict(block.get("counterfactual") or {})
    for key in list(cf):
        if isinstance(cf[key], np.ndarray):
            cf.pop(key)
    trust = block.get("trust")
    if isinstance(trust, dict):
        trust = {
            k: trust[k]
            for k in ("status", "ood", "untrusted", "reasons", "role", "thresholds", "banner")
            if k in trust
        }
    return {
        "predict": {
            "rel_l2_mean": pred.get("rel_l2_mean"),
            "rel_l2_std": pred.get("rel_l2_std"),
            "nmse": pred.get("nmse"),
        }
        if pred
        else None,
        "conserve": {
            "residual_rel_l2": cons.get("residual_rel_l2"),
            "energy_max_abs_rel_drift": cons.get("energy_max_abs_rel_drift"),
        }
        if cons
        else None,
        "counterfactual": cf or None,
        "trust": trust,
    }


def grade_adapter(
    adapter,
    data: dict,
    idx: np.ndarray,
    *,
    pde: str,
    grid: dict,
    role: str = "surrogate",
) -> tuple[dict, dict | None]:
    """Exam 1-3 on ``idx``. Returns ``(row, sample_arrays_or_None)``."""
    name = str(getattr(adapter, "name", "surrogate"))
    u = np.asarray(data["u"])[idx]
    if pde == "burgers":
        coeff = np.asarray(data["nu"], dtype=np.float64)[idx]
        support = BURGERS_TRAIN_SUPPORT
        ic_family = support["ic_family"]
    elif pde == "heat2d":
        coeff = np.asarray(data["alpha"], dtype=np.float64)[idx]
        support = HEAT_TRAIN_SUPPORT
        ic_family = support["ic_family"]
    else:
        raise ValueError(f"unknown pde {pde!r}")

    u0 = u[:, 0]
    nt = int(grid["nt"])
    try:
        pred = np.asarray(adapter.predict(u0, coeff, n_steps=nt), dtype=np.float64)
    except Exception as exc:
        row = _not_trained(name, f"forward_failed: {type(exc).__name__}: {exc}")
        row["status"] = "error"
        row["role"] = role
        return row, None
    if pred.shape != u.shape:
        row = _not_trained(
            name,
            f"forward_failed: shape {pred.shape} != truth {u.shape}",
        )
        row["status"] = "error"
        row["role"] = role
        return row, None

    pred_metrics = batch_relative_l2(pred, u)
    if pde == "burgers":
        conserve = audit_burgers(pred, dx=grid["dx"], dt=grid["dt"], nu=coeff)
        nu_cf = float(coeff[0]) * 2.0
        pred_cf = np.asarray(
            adapter.predict(u0[:1], np.asarray([nu_cf]), n_steps=nt),
            dtype=np.float64,
        )[0]
        cf = grade_burgers_cf_prediction(
            pred_cf,
            u0=u0[0],
            nu_orig=float(coeff[0]),
            nu_cf=nu_cf,
            nx=int(grid["nx"]),
            nt=nt,
            L=float(grid["L"]),
            T=float(grid["T"]),
        )
        resolution = int(grid["nx"])
    else:
        conserve = audit_heat(pred, dx=grid["dx"], dt=grid["dt"], alpha=coeff)
        a_cf = float(coeff[0]) * 1.5
        pred_cf = np.asarray(
            adapter.predict(u0[:1], np.asarray([a_cf]), n_steps=nt),
            dtype=np.float64,
        )[0]
        cf = grade_heat_cf_prediction(
            pred_cf,
            u0=u0[0],
            alpha_orig=float(coeff[0]),
            alpha_cf=a_cf,
            n=int(grid["n"]),
            nt=nt,
            L=float(grid["L"]),
            T=float(grid["T"]),
        )
        resolution = int(grid["n"])
    cf["rel_l2_traj_delta_model"] = relative_l2(pred_cf, pred[0])

    block = {
        "name": name,
        "status": "ok",
        "role": role,
        "checkpoint": getattr(adapter, "checkpoint", None),
        "n_eval": int(len(idx)),
        "eval_idx": np.asarray(idx, dtype=int).tolist(),
        "predict": {
            "rel_l2_mean": pred_metrics["mean"],
            "rel_l2_std": pred_metrics["std"],
            "nmse": pred_metrics["nmse"],
        },
        "conserve": conserve,
        "counterfactual": cf,
    }
    block = attach_trust(
        block,
        support=support,
        param=float(np.mean(coeff)),
        resolution=resolution,
        ic_family=ic_family,
        role=role,
    )
    sample = {
        "truth": u[0],
        "pred": pred[0],
        "x": np.asarray(data["x"]),
        "t": np.asarray(data["t"]),
        "banner": (block.get("trust") or {}).get("banner", ""),
        "name": name,
        "pde": pde,
    }
    public = {
        "name": name,
        "status": "ok",
        "role": role,
        "checkpoint": block.get("checkpoint"),
        "n_eval": block["n_eval"],
        "eval_idx": block["eval_idx"],
        **_public_metrics(block),
    }
    return public, sample


def grade_ood_adapter(adapter, cases: dict, *, pde: str, role: str) -> dict:
    name = str(getattr(adapter, "name", "surrogate"))
    nt = int(cases["nt"])
    try:
        if pde == "burgers":
            pred = np.asarray(
                adapter.predict(cases["u0"], cases["nu"], n_steps=nt),
                dtype=np.float64,
            )
            scored = score_burgers_surrogate(pred, cases) if role != "labeler" else None
        else:
            pred = np.asarray(
                adapter.predict(cases["u0"], cases["alpha"], n_steps=nt),
                dtype=np.float64,
            )
            scored = score_heat_surrogate(pred, cases) if role != "labeler" else None
        if role == "labeler":
            from metrics.ood import score_burgers_labeler, score_heat_labeler

            scored = score_burgers_labeler(cases) if pde == "burgers" else score_heat_labeler(cases)
            # Labeler score uses the classical field already in ``cases``, not ``pred``.
            # Still require the adapter to run so a broken labeler is visible.
            del pred
    except Exception as exc:
        return {
            "name": name,
            "status": "error",
            "reason": f"forward_failed: {type(exc).__name__}: {exc}",
            "trust": {
                "status": "untrusted",
                "reasons": ["forward_failed"],
                "banner": "UNTRUSTED (forward_failed)",
            },
        }
    public = {
        "name": name,
        "status": "ok",
        "role": role,
        **_public_metrics(scored),
    }
    return public


def _grid_burgers(data: dict) -> dict:
    x = np.asarray(data["x"])
    t = np.asarray(data["t"])
    dx = float(x[1] - x[0])
    return {
        "nx": int(len(x)),
        "nt": int(len(t) - 1),
        "dx": dx,
        "dt": float(t[1] - t[0]),
        "L": float(x[-1] + dx),
        "T": float(t[-1]),
    }


def _grid_heat(data: dict) -> dict:
    x = np.asarray(data["x"])
    t = np.asarray(data["t"])
    return {
        "n": int(len(x)),
        "nt": int(len(t) - 1),
        "dx": float(x[1] - x[0]),
        "dt": float(t[1] - t[0]),
        "L": float(x[-1]),
        "T": float(t[-1]),
    }


def load_burgers_labels(root: Path | None = None) -> tuple[dict, str]:
    root = root or ROOT
    path = root / "datasets" / "burgers" / "burgers_v0.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}, "datasets/burgers/burgers_v0.npz"
    from baselines.classical.burgers1d import generate_dataset

    data = generate_dataset(n_traj=48, nx=64, nt=80, seed=0)
    return data, "generated burgers n_traj=48 nx=64 nt=80 seed=0 (same recipe as scripts.generate_labels)"


def load_heat_labels(root: Path | None = None) -> tuple[dict, str]:
    root = root or ROOT
    path = root / "datasets" / "heat2d" / "heat2d_v0.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}, "datasets/heat2d/heat2d_v0.npz"
    from baselines.classical.heat2d import generate_dataset

    data = generate_dataset(n_traj=32, n=32, nt=40, seed=1)
    return data, "generated heat2d n_traj=32 n=32 nt=40 seed=1 (same recipe as scripts.generate_labels)"


def smoke_burgers() -> tuple[dict, str]:
    from baselines.classical.burgers1d import generate_dataset

    data = generate_dataset(n_traj=4, nx=16, nt=8, seed=0)
    return data, "smoke generated burgers n_traj=4 nx=16 nt=8 seed=0"


def _formula_holdout_idx(pde: str, n: int) -> np.ndarray:
    """Match ``baselines.fno.train._split_indices`` for the shipped defaults.

    Burgers asks for 16 eval trajectories and heat asks for 12. Both are
    capped at ``n // 4``, which is 12 of 48 and 8 of 32 on the label sets.
    """
    requested = 16 if pde == "burgers" else 12
    eval_n = min(int(requested), max(1, int(n) // 4))
    train_n = int(n) - eval_n
    if train_n < 1:
        train_n = max(1, int(n) - 1)
    return np.arange(train_n, int(n))


def _read_shipped_eval_idx(pde: str) -> list[int] | None:
    """Eval indices stored on the shipped FNO checkpoint, if it is present."""
    try:
        import torch
        from baselines.fno.io import default_ckpt
    except ImportError:
        return None
    path = default_ckpt(pde, "fno")
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        return None
    idx = ((payload.get("extra") or {}).get("split") or {}).get("eval_idx")
    if not idx:
        return None
    return [int(i) for i in idx]


def holdout_idx(pde: str, n: int, n_eval: int | None = None) -> np.ndarray:
    """Checkpoint holdout for ``pde``, capped to ``n_eval`` when that is set.

    Prefers the indices saved on the shipped FNO checkpoint. The training
    formula is the fallback when that file, or torch, is absent. Indices
    outside ``[0, n)`` are dropped.
    """
    raw = _read_shipped_eval_idx(pde)
    if raw is None:
        arr = _formula_holdout_idx(pde, n)
    else:
        arr = np.asarray(raw, dtype=int)
        arr = arr[(arr >= 0) & (arr < int(n))]
        if len(arr) == 0:
            arr = _formula_holdout_idx(pde, n)
    if n_eval is not None:
        arr = arr[: int(n_eval)]
    return np.asarray(arr, dtype=int)


def _eval_idx_from_checkpoint(loaded: dict, n: int, n_eval: int | None) -> np.ndarray | None:
    extra = loaded.get("extra") or {}
    idx = (extra.get("split") or {}).get("eval_idx")
    if not idx:
        return None
    arr = np.asarray(idx, dtype=int)
    arr = arr[arr < n]
    if n_eval is not None:
        arr = arr[: int(n_eval)]
    return arr if len(arr) else None


def _fallback_idx(n: int, n_eval: int | None) -> np.ndarray:
    take = n if n_eval is None else min(int(n_eval), n)
    return np.arange(n - take, n)


def load_shipped(name: str, pde: str, checkpoint: Path | None = None) -> tuple[object | None, dict]:
    """Load ``fno`` or ``pino``. Missing files return ``(None, not_trained_info)``."""
    try:
        from baselines.fno.io import default_ckpt, load_checkpoint
    except ImportError as exc:
        return None, {"status": "not_trained", "reason": f"torch_not_installed: {exc}", "path": None}
    objective = "pino" if name == "pino" else "fno"
    path = Path(checkpoint) if checkpoint is not None else default_ckpt(pde, objective)
    loaded = load_checkpoint(path, device="cpu")
    if loaded.get("status") != "ok":
        return None, loaded
    adapter = TorchOperatorAdapter(
        loaded["model"],
        name=name,
        checkpoint=str(path),
        device="cpu",
    )
    return adapter, loaded


def train_smoke_fno(data: dict, out_path: Path) -> tuple[TorchOperatorAdapter, dict]:
    from baselines.fno.train import train_burgers

    trained = train_burgers(
        data=data,
        epochs=1,
        width=4,
        modes=2,
        n_layers=2,
        batch_size=2,
        eval_n=2,
        seed=0,
        device="cpu",
        out_path=out_path,
        log_every=1,
    )
    adapter = TorchOperatorAdapter(
        trained["model"],
        name="fno",
        checkpoint=trained.get("path"),
        device="cpu",
    )
    return adapter, trained


def _ood_probe(pde: str, grid: dict, *, n: int) -> dict:
    if pde == "burgers":
        cases = generate_burgers_cases(
            n=n,
            nu=float(BURGERS_OOD_NU["below"]),
            nx=int(grid["nx"]),
            family="fourier_modes",
            seed=10_001,
            nt=int(grid["nt"]),
            L=float(grid["L"]),
            T=float(grid["T"]),
        )
        cases["spec"] = {
            "id": "burgers_param_below",
            "probe": "param_shift",
            "note": "nu below the declared train range; grid matches this run",
        }
        return cases
    cases = generate_heat_cases(
        n_cases=n,
        alpha=float(HEAT_OOD_ALPHA["below"]),
        n=int(grid["n"]),
        family="gaussian_bump",
        seed=10_041,
        nt=int(grid["nt"]),
        L=float(grid["L"]),
        T=float(grid["T"]),
    )
    cases["spec"] = {
        "id": "heat2d_param_below",
        "probe": "param_shift",
        "note": "alpha below the declared train range; grid matches this run",
    }
    return cases


def _exam4_brief(block: dict) -> dict:
    keep = (
        "status",
        "n_items",
        "item_set",
        "item_set_sha256",
        "answer_source",
        "mean_coverage",
        "n_perfect",
        "n_zero",
        "note",
    )
    return {k: block.get(k) for k in keep}


def _observations_report(sections: dict) -> dict:
    """Shape a scorecard into the dict ``observations_from_report`` reads."""
    report: dict = {
        "exams": {"predict": {}, "conserve": {}, "counterfactual": {}},
        "baselines": {},
        "ood": {"trust_summary": {"n_scored": 0}},
    }
    n_ood = 0
    for pde, section in sections.items():
        for row in section.get("models") or []:
            if row.get("name") not in {"fno", "pino"}:
                continue
            model = row["name"]
            report["baselines"].setdefault(model, {})
            report["baselines"][model][pde] = row
            status = row.get("status")
            for exam, key in (
                ("predict", "predict"),
                ("conserve", "conserve"),
                ("counterfactual", "counterfactual"),
            ):
                slot = report["exams"][exam].setdefault(model, {"status": "ok"})
                payload = row.get(key)
                if status == "ok" and isinstance(payload, dict):
                    slot[pde] = {"status": "ok", **payload}
                else:
                    slot[pde] = {"status": status or "not_trained"}
                    slot["status"] = "ok"
        for probe in section.get("ood") or []:
            for row in probe.get("models") or []:
                if row.get("name") in {"fno", "pino"} and row.get("trust"):
                    n_ood += 1
    report["ood"]["trust_summary"]["n_scored"] = n_ood
    return report


def reading_lines(card: dict) -> list[str]:
    """Sentences that only restate measured comparisons. No praise, no guesses."""
    lines: list[str] = []
    pred_cut = float(PREDICT_UNTRUSTED)
    res_cut = float(RESIDUAL_UNTRUSTED)
    if card.get("smoke"):
        lines.append("Smoke run on a tiny Burgers grid. Not the shipped checkpoint.")
    for pde, section in (card.get("pdes") or {}).items():
        rows = {row["name"]: row for row in section.get("models") or []}
        for name, row in rows.items():
            if row.get("status") != "ok":
                lines.append(
                    f"{name} {pde}: {row.get('status')} ({row.get('reason') or 'no metrics'})."
                )
                continue
            pred = float(row["predict"]["rel_l2_mean"])
            res = float(row["conserve"]["residual_rel_l2"])
            trust = (row.get("trust") or {}).get("status")
            reasons = ",".join((row.get("trust") or {}).get("reasons") or []) or "none"
            pred_side = "under" if pred <= pred_cut else "over"
            res_side = "under" if res <= res_cut else "over"
            lines.append(
                f"{name} {pde}: Predict rel-L2 {pred:.6e} is {pred_side} the {pred_cut:g} cut. "
                f"Residual rel-L2 {res:.6e} is {res_side} the {res_cut:g} cut. "
                f"Trust is {trust} ({reasons})."
            )
            if (
                row.get("role") != "labeler"
                and pred <= pred_cut
                and res > res_cut
                and trust == "untrusted"
            ):
                lines.append(
                    f"Fail-closed: {name} on {pde} is inside the Predict cut and outside "
                    f"the Conserve cut, so the flag is untrusted."
                )
            cf = row.get("counterfactual") or {}
            model_delta = cf.get("rel_l2_traj_delta_model")
            class_delta = cf.get("rel_l2_traj_delta_classical")
            vs = cf.get("rel_l2_vs_classical_cf")
            if (
                model_delta is not None
                and class_delta is not None
                and vs is not None
                and row.get("role") != "labeler"
            ):
                lines.append(
                    f"{name} {pde} counterfactual: model traj change {float(model_delta):.6e}, "
                    f"classical traj change {float(class_delta):.6e}, "
                    f"model vs classical at the new coefficient {float(vs):.6e}."
                )
                if float(model_delta) < 1e-8:
                    lines.append(
                        f"{name} on {pde} does not move when the coefficient changes. "
                        f"IID trust is {trust}. The trust flag uses Predict and Conserve only, not Exam 3."
                    )
        trusted_rows = [
            row
            for row in rows.values()
            if row.get("status") == "ok"
            and row.get("role") != "labeler"
            and (row.get("trust") or {}).get("status") == "trusted"
        ]
        untrusted_rows = [
            row
            for row in rows.values()
            if row.get("status") == "ok"
            and (row.get("trust") or {}).get("status") == "untrusted"
        ]
        for good in trusted_rows:
            for bad in untrusted_rows:
                good_l2 = float(good["predict"]["rel_l2_mean"])
                bad_l2 = float(bad["predict"]["rel_l2_mean"])
                if bad_l2 < good_l2:
                    lines.append(
                        f"On {pde}, {bad['name']} has a lower Predict rel-L2 ({bad_l2:.6e}) than "
                        f"{good['name']} ({good_l2:.6e}) and is untrusted, while {good['name']} is trusted. "
                        "A lower Predict rel-L2 is not the trust flag."
                    )
        fno = rows.get("fno")
        pino = rows.get("pino")
        if fno and pino and fno.get("status") == "ok" and pino.get("status") == "ok":
            fno_l2 = float(fno["predict"]["rel_l2_mean"])
            pino_l2 = float(pino["predict"]["rel_l2_mean"])
            fno_r = float(fno["conserve"]["residual_rel_l2"])
            pino_r = float(pino["conserve"]["residual_rel_l2"])
            lower_l2 = "fno" if fno_l2 < pino_l2 else "pino"
            lower_res = "fno" if fno_r < pino_r else "pino"
            lines.append(
                f"On {pde}, {lower_l2} has the lower Predict rel-L2 "
                f"({min(fno_l2, pino_l2):.6e}) and {lower_res} has the lower residual "
                f"({min(fno_r, pino_r):.6e})."
            )
        for probe in section.get("ood") or []:
            for row in probe.get("models") or []:
                if row.get("status") != "ok" or row.get("role") == "labeler":
                    continue
                trust = (row.get("trust") or {}).get("status")
                reason_list = list((row.get("trust") or {}).get("reasons") or [])
                reasons = ",".join(reason_list) or "none"
                pred = row.get("predict") or {}
                cons = row.get("conserve") or {}
                sentence = (
                    f"OOD {probe.get('id')} {row['name']}: Predict rel-L2 "
                    f"{float(pred.get('rel_l2_mean')):.6e}, residual "
                    f"{float(cons.get('residual_rel_l2')):.6e}, trust {trust} ({reasons})."
                )
                ood_reasons = {"param_shift", "resolution_shift", "ic_family_shift"}
                if trust == "untrusted" and ood_reasons.intersection(reason_list):
                    sentence += " untrusted wins over ood when both apply."
                lines.append(sentence)
    explain = card.get("explain") or {}
    gold = (explain.get("gold") or {}).get("mean_coverage")
    rule = (explain.get("rule_based") or {}).get("mean_coverage")
    dump = (explain.get("metric_dump") or {}).get("mean_coverage")
    llm = (explain.get("llm") or {}).get("status")
    if gold is not None:
        lines.append(
            f"Exam 4 ({explain.get('n_items')} items): gold coverage {gold:.6f}, "
            f"rule-based {rule:.6f}, metric-dump {dump:.6f}, llm {llm}. "
            "Gold, rule-based, and metric-dump are keyword scores, not an LLM judge."
        )
    return lines


def _index_span(idx: list[int]) -> str:
    if not idx:
        return "no indices"
    if idx == list(range(idx[0], idx[-1] + 1)):
        return f"indices {idx[0]}-{idx[-1]}"
    shown = ",".join(str(i) for i in idx[:8])
    return f"indices {shown}" + ("..." if len(idx) > 8 else "")


def eval_set_note(card: dict) -> str:
    """One or two sentences naming the trajectories this card actually scored."""
    parts: list[str] = []
    for pde, section in (card.get("pdes") or {}).items():
        idx = [int(i) for i in (section.get("eval_idx") or [])]
        n_total = section.get("n_total")
        total = f" of {n_total}" if n_total is not None else ""
        label = section.get("split_label") or "eval"
        parts.append(
            f"{pde}: {label}, {section.get('n_eval')}{total} trajectories, {_index_span(idx)}"
        )
    sentence = "Eval set: " + "; ".join(parts) + "."
    if card.get("smoke"):
        return sentence + " Smoke grid, not the shipped checkpoint holdout."
    splits = {section.get("split") for section in (card.get("pdes") or {}).values()}
    if card.get("command_kind") == "demo":
        return (
            sentence
            + " demo scores this checkpoint holdout."
            + " grade scores every trajectory when no shipped checkpoint is loaded"
            + " (Burgers 48, heat 32). grade --split holdout matches these indices."
        )
    if splits == {"holdout"}:
        return sentence + " Same checkpoint holdout as coticula demo."
    if splits == {"all"}:
        return (
            sentence
            + " grade default without a shipped checkpoint is every trajectory."
            + " coticula demo uses the checkpoint holdout (Burgers 12, heat 8)."
            + " Pass --split holdout to match demo."
        )
    return sentence


def build_scorecard(
    *,
    pdes: list[str],
    model_names: list[str],
    smoke: bool = False,
    n_eval: int | None = None,
    include_ood: bool = True,
    checkpoint: Path | None = None,
    adapter=None,
    command: str = "coticula",
    out_dir: Path | None = None,
    split: str = "auto",
    command_kind: str = "grade",
) -> tuple[dict, dict]:
    """Grade the requested models. Returns ``(scorecard, plot_arrays)``.

    ``split="auto"`` keeps the historical choice: a loaded checkpoint's
    eval indices, otherwise every trajectory (the last 2 on a smoke grid).
    ``split="holdout"`` is the checkpoint holdout ``coticula demo`` uses.
    ``split="all"`` is every trajectory in the label set.
    """
    requested = (split or "auto").lower()
    if requested not in {"auto", "holdout", "all"}:
        raise ValueError(f"unknown split {split!r}")
    from datetime import datetime, timezone

    from coticula import __version__

    sections: dict = {}
    arrays: dict = {}
    for pde in pdes:
        if smoke:
            if pde != "burgers":
                continue
            data, source = smoke_burgers()
            grid = _grid_burgers(data)
        elif pde == "burgers":
            data, source = load_burgers_labels()
            grid = _grid_burgers(data)
        elif pde == "heat2d":
            data, source = load_heat_labels()
            grid = _grid_heat(data)
        else:
            raise ValueError(f"unknown pde {pde!r}")

        n = int(np.asarray(data["u"]).shape[0])
        shared_idx = None
        prepared: list[tuple[object, str]] = []
        missing_rows: list[dict] = []
        learned_names = [m for m in model_names if m in {"fno", "pino"}]

        if smoke and "fno" in model_names:
            ckpt_path = (out_dir or Path(".")) / "smoke_fno_burgers.pt"
            fno_adapter, trained = train_smoke_fno(data, ckpt_path)
            prepared.append((fno_adapter, "surrogate"))
            shared_idx = _eval_idx_from_checkpoint(
                {"extra": trained.get("extra")}, n, n_eval or 2
            )
        for name in model_names:
            if name == "classical_fd":
                prepared.append((ClassicalFD(pde, L=grid["L"], T=grid["T"]), "labeler"))
            elif name == "persist_ic":
                prepared.append((persist_example(), "surrogate"))
            elif name in {"fno", "pino"}:
                if smoke:
                    if name == "pino":
                        missing_rows.append(_not_trained("pino", "smoke_skips_pino", None))
                    continue
                use_ckpt = checkpoint if learned_names == [name] else None
                loaded_adapter, info = load_shipped(name, pde, use_ckpt)
                if loaded_adapter is None:
                    missing_rows.append(
                        _not_trained(
                            name,
                            str(info.get("reason") or "not_trained"),
                            info.get("path"),
                        )
                    )
                    continue
                if shared_idx is None:
                    shared_idx = _eval_idx_from_checkpoint(info, n, n_eval)
                prepared.append((loaded_adapter, "surrogate"))
            else:
                raise ValueError(f"unknown model {name!r}")

        if adapter is not None:
            prepared.append((adapter, getattr(adapter, "role", "surrogate")))

        from_checkpoint = shared_idx is not None
        if requested == "all":
            shared_idx = _fallback_idx(n, n_eval)
            split_name = "all"
            split_label = "all trajectories"
        elif requested == "holdout":
            if not from_checkpoint:
                shared_idx = holdout_idx(pde, n, n_eval)
            split_name = "holdout"
            split_label = "checkpoint holdout"
        elif shared_idx is None:
            cap = 2 if smoke else n_eval
            shared_idx = _fallback_idx(n, cap)
            if cap is None or int(cap) >= n:
                split_name = "all"
                split_label = "all trajectories"
            else:
                split_name = "tail"
                split_label = f"last {len(shared_idx)} trajectories"
        else:
            split_name = "holdout"
            split_label = "checkpoint holdout"

        rows = []
        for obj, role in prepared:
            row, sample = grade_adapter(
                obj, data, shared_idx, pde=pde, grid=grid, role=role
            )
            rows.append(row)
            if sample is not None and row.get("name") in {"fno", "persist_ic"}:
                arrays[(pde, row["name"])] = sample
        rows.extend(missing_rows)
        order = {name: i for i, name in enumerate(model_names)}
        rows.sort(key=lambda row: order.get(row.get("name"), 50))

        ood_rows = []
        if include_ood:
            # Match the model's time axis. Smoke FNO is nt=8; shipped models match the label grid.
            probe_grid = dict(grid)
            cases = _ood_probe(pde, probe_grid, n=1 if smoke else 4)
            probe_models = []
            for obj, role in prepared:
                probe_models.append(grade_ood_adapter(obj, cases, pde=pde, role=role))
            for row in missing_rows:
                probe_models.append(
                    {
                        "name": row["name"],
                        "status": row["status"],
                        "reason": row.get("reason"),
                        "trust": None,
                    }
                )
            probe_models.sort(key=lambda row: order.get(row.get("name"), 50))
            spec = cases.get("spec") or {}
            ood_rows.append(
                {
                    "id": spec.get("id"),
                    "probe": spec.get("probe"),
                    "note": spec.get("note"),
                    "param": float(cases["param"]),
                    "n": int(cases["u"].shape[0]),
                    "resolution": int(cases["nx"] if pde == "burgers" else cases["n_grid"]),
                    "ic_family": cases["ic_family"],
                    "models": probe_models,
                }
            )

        sections[pde] = {
            "label_source": source,
            "grid": grid,
            "n_total": n,
            "n_eval": int(len(shared_idx)),
            "eval_idx": shared_idx.tolist(),
            "split": split_name,
            "split_label": split_label,
            "models": rows,
            "ood": ood_rows,
        }

    card = {
        "project": "Coticula",
        "version": __version__,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "command_kind": command_kind,
        "smoke": bool(smoke),
        "trust_policy": {
            "name": TRUST_POLICY["name"],
            "residual_untrusted": RESIDUAL_UNTRUSTED,
            "predict_untrusted": PREDICT_UNTRUSTED,
            "rule": TRUST_POLICY["rule"],
        },
        "pdes": sections,
    }
    obs_report = _observations_report(sections)
    obs = observations_from_report(obs_report)
    gold = grade_exam4(use_gold=True)
    card["explain"] = {
        "n_items": gold["n_items"],
        "item_set": gold["item_set"],
        "item_set_sha256": gold["item_set_sha256"],
        "gold": _exam4_brief(gold),
        "rule_based": _exam4_brief(grade_rule_based_exam4(obs)),
        "metric_dump": _exam4_brief(grade_metric_dump_exam4(obs)),
        "llm": llm_judge_exam4(),
        "observations": obs,
    }
    card["reading"] = reading_lines(card)
    card["eval_set"] = eval_set_note(card)
    return card, arrays
