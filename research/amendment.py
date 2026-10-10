"""Protocol v2 scoring. Does not rewrite v1 artifacts or the frozen v1 protocol.

v1 selected hyperparameters are reused. The amendment retrains the five seeds
so paired tests can use every (seed, trajectory), then calibrates residuals
against a smooth-error band.
"""
from __future__ import annotations

import time

import numpy as np

from baselines.classical.burgers1d import solve_burgers
from baselines.classical.heat2d import solve_heat2d
from metrics.predict import relative_l2
from metrics.trust import RESIDUAL_UNTRUSTED
from metrics.weak_form import spec_from_protocol
from research.datautil import load_burgers, load_heat, split_indices
from research.models import build_model, parameter_count
from research.noise import scale_to_rel
from research.pipeline import (
    _baseline_fields_burgers,
    _baseline_fields_heat,
    _inject,
    _score_burgers,
    _score_heat,
)
from research.statsutil import best_baseline, log_interp, paired_wilcoxon, scaled_score
from research.strong import burgers_strong_per_traj, heat_strong_per_traj, mean_std
from research.train_study import predict, train_model
from metrics.weak_form import burgers_weak_per_traj, heat_weak_per_traj


def calibrate(rho: float, rel: float, levels, median, p05, p95) -> dict:
    """C = rho / smooth median at this relative L2, plus the 5–95% band.

    Survival is the strict comparison with the 95% edge. It is not called when
    the relative L2 lies outside the noise grid or the ratio is undefined.
    """
    rel = float(rel)
    rho = float(rho)
    levels = np.asarray(levels, dtype=np.float64)
    if not np.isfinite(rel) or rel <= 0.0 or not np.isfinite(rho):
        return {
            "C": None,
            "median": None,
            "p05": None,
            "p95": None,
            "above_p95": None,
            "extrapolated": True,
        }
    med = log_interp(levels, median, rel)
    lo = log_interp(levels, p05, rel)
    hi = log_interp(levels, p95, rel)
    if med["value"] is None or med["value"] <= 0:
        return {
            "C": None,
            "median": med["value"],
            "p05": lo["value"],
            "p95": hi["value"],
            "above_p95": None,
            "extrapolated": True,
        }
    above = None if med["extrapolated"] else bool(rho > hi["value"])
    return {
        "C": float(rho / med["value"]),
        "median": med["value"],
        "p05": lo["value"],
        "p95": hi["value"],
        "above_p95": above,
        "extrapolated": bool(med["extrapolated"]),
    }


def _quantiles(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"median": float("nan"), "p05": float("nan"), "p95": float("nan"), "n": 0}
    p05, med, p95 = np.quantile(arr, [0.05, 0.5, 0.95])
    return {"median": float(med), "p05": float(p05), "p95": float(p95), "n": int(arr.size)}


def _score_stack(fields, ref, data, coef, spec, pde: str) -> dict:
    """Score a batch that shares one reference trajectory and one coefficient."""
    if pde == "burgers":
        dx = float(data["x"][1] - data["x"][0])
        dt = float(data["t"][1] - data["t"][0])
        strong = burgers_strong_per_traj(fields, dx, dt, coef)
        weak = burgers_weak_per_traj(fields, data["x"], data["t"], coef, spec, u_ref=ref)
    else:
        dx = float(data["x"][1] - data["x"][0])
        dt = float(data["t"][1] - data["t"][0])
        strong = heat_strong_per_traj(fields, dx, dt, coef)
        weak = heat_weak_per_traj(fields, data["x"], data["y"], data["t"], coef, spec, u_ref=ref)
    rels = np.array([relative_l2(fields[i], ref) for i in range(fields.shape[0])])
    return {"rel_l2": rels, "strong": np.asarray(strong), "weak": np.asarray(weak)}


def _noise_curves(v1, spec, data, pde: str, seeds: list[int], salt: int) -> dict:
    cfg = v1["noise"]
    levels = [float(v) for v in cfg["levels"]]
    coef_key = "nu" if pde == "burgers" else "alpha"
    coords = {"x": data["x"], "t": data["t"]}
    if pde == "heat":
        coords["y"] = data["y"]
    buckets = {
        (kind, level): {"strong": [], "weak": []} for kind in ("white", "smooth") for level in levels
    }
    for i in range(data["u"].shape[0]):
        ref = data["u"][i]
        coef = float(data[coef_key][i])
        for level in levels:
            for kind in ("white", "smooth"):
                draws = []
                for seed in seeds:
                    rng = np.random.default_rng(
                        np.random.SeedSequence(
                            [
                                salt,
                                int(seed),
                                i,
                                int(round(level * 1_000_000)),
                                1 if kind == "white" else 2,
                                1 if pde == "burgers" else 2,
                            ]
                        )
                    )
                    draws.append(ref + _inject(ref, kind, level, rng, pde=pde, coords=coords, noise_cfg=cfg))
                scored = _score_stack(np.stack(draws, axis=0), ref, data, coef, spec, pde)
                buckets[(kind, level)]["strong"].extend(scored["strong"].tolist())
                buckets[(kind, level)]["weak"].extend(scored["weak"].tolist())
    curves = {"levels": levels}
    for kind in ("white", "smooth"):
        curves[kind] = {}
        for metric in ("strong", "weak"):
            rows = [_quantiles(buckets[(kind, level)][metric]) for level in levels]
            curves[kind][metric] = {
                "median": [r["median"] for r in rows],
                "p05": [r["p05"] for r in rows],
                "p95": [r["p95"] for r in rows],
                "n": [r["n"] for r in rows],
            }
    return curves


def _scaled_field(pde: str, data, index: int, scale: float, block: dict) -> np.ndarray:
    ref = data["u"][index]
    if pde == "burgers":
        coef = float(data["nu"][index]) * float(scale)
        return solve_burgers(
            nu=coef,
            nx=int(block["nx"]),
            nt=int(block["nt"]),
            L=float(block["L"]),
            T=float(block["T"]),
            u0=ref[0],
        )["u"]
    coef = float(data["alpha"][index]) * float(scale)
    return solve_heat2d(
        alpha=coef,
        n=int(block["n"]),
        nt=int(block["nt"]),
        L=float(block["L"]),
        T=float(block["T"]),
        u0=ref[0],
    )["u"]


def _curve_arrays(curves, kind, metric):
    block = curves[kind][metric]
    return (
        np.asarray(curves["levels"], dtype=np.float64),
        np.asarray(block["median"], dtype=np.float64),
        np.asarray(block["p05"], dtype=np.float64),
        np.asarray(block["p95"], dtype=np.float64),
    )


def _ratio(rho, rel, curves, metric):
    levels, med, p05, p95 = _curve_arrays(curves, "smooth", metric)
    return calibrate(rho, rel, levels, med, p05, p95)


def _finite_mean(values) -> float | None:
    arr = np.asarray([np.nan if v is None else v for v in values], dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(finite.mean())


def _mean_std_optional(values) -> dict:
    arr = np.asarray([np.nan if v is None else v for v in values], dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": None, "std": None, "n": 0}
    mean, std = mean_std(finite.tolist())
    return {"mean": mean, "std": std, "n": int(finite.size)}


def run_mismatch_ladder(v1, v2, spec, burgers, heat, curves) -> dict:
    scales = [float(s) for s in v2["mismatch"]["scales"]]
    direction = float(v2["mismatch"]["matched_direction_scale"])
    levels = [float(v) for v in v1["noise"]["levels"]]
    out = {"natural": {}, "matched": {}}
    datasets = {"burgers": burgers, "heat": heat}
    for pde, data in datasets.items():
        block = v1["datasets"][pde]
        coef_key = "nu" if pde == "burgers" else "alpha"
        solved = {scale: [] for scale in scales}
        for i in range(data["u"].shape[0]):
            for scale in scales:
                field = _scaled_field(pde, data, i, scale, block)
                if not np.isfinite(field).all():
                    solved[scale].append(None)
                else:
                    solved[scale].append(field)
        natural_rows = []
        for scale in scales:
            per = []
            for i, field in enumerate(solved[scale]):
                if field is None:
                    continue
                ref = data["u"][i]
                coef = float(data[coef_key][i])
                if pde == "burgers":
                    scored = _score_burgers(field, ref, data["x"], data["t"], coef, spec)
                else:
                    scored = _score_heat(field, ref, data["x"], data["y"], data["t"], coef, spec)
                c_s = _ratio(scored["strong"], scored["rel_l2"], curves[pde], "strong")
                c_w = _ratio(scored["weak"], scored["rel_l2"], curves[pde], "weak")
                per.append({"rel_l2": scored["rel_l2"], "strong": scored["strong"], "weak": scored["weak"], "C_strong": c_s, "C_weak": c_w})
            rels = [r["rel_l2"] for r in per]
            rel_mean = float(np.mean(rels))
            smooth_s = _ratio(1.0, rel_mean, curves[pde], "strong")
            smooth_w = _ratio(1.0, rel_mean, curves[pde], "weak")
            w_levels, w_med_s, _, _ = _curve_arrays(curves[pde], "white", "strong")
            _, w_med_w, _, _ = _curve_arrays(curves[pde], "white", "weak")
            natural_rows.append(
                {
                    "scale": scale,
                    "n": len(per),
                    "rel_l2_mean": rel_mean,
                    "rel_l2_std": mean_std(rels)[1],
                    "strong_mean": float(np.mean([r["strong"] for r in per])),
                    "weak_mean": float(np.mean([r["weak"] for r in per])),
                    "strong_median": float(np.median([r["strong"] for r in per])),
                    "weak_median": float(np.median([r["weak"] for r in per])),
                    "C_strong": _mean_std_optional([r["C_strong"]["C"] for r in per if not r["C_strong"]["extrapolated"]]),
                    "C_weak": _mean_std_optional([r["C_weak"]["C"] for r in per if not r["C_weak"]["extrapolated"]]),
                    "n_extrapolated": int(sum(1 for r in per if r["C_strong"]["extrapolated"])),
                    "smooth_strong_at_mean_rel": smooth_s["median"],
                    "smooth_weak_at_mean_rel": smooth_w["median"],
                    "white_strong_at_mean_rel": log_interp(w_levels, w_med_s, rel_mean)["value"],
                    "white_weak_at_mean_rel": log_interp(w_levels, w_med_w, rel_mean)["value"],
                    "smooth_extrapolated": bool(smooth_s["extrapolated"]),
                }
            )
        matched_rows = []
        for level in levels:
            strong_m, weak_m = [], []
            for i in range(data["u"].shape[0]):
                ref = data["u"][i]
                shifted = solved[direction][i]
                if shifted is None:
                    continue
                err = scale_to_rel(shifted - ref, ref, level)
                field = ref + err
                coef = float(data[coef_key][i])
                if pde == "burgers":
                    scored = _score_burgers(field, ref, data["x"], data["t"], coef, spec)
                else:
                    scored = _score_heat(field, ref, data["x"], data["y"], data["t"], coef, spec)
                strong_m.append(scored["strong"])
                weak_m.append(scored["weak"])
            sm, ss = mean_std(strong_m)
            wm, ws = mean_std(weak_m)
            s_med = float(np.median(strong_m))
            w_med = float(np.median(weak_m))
            c_s = _ratio(s_med, level, curves[pde], "strong")
            c_w = _ratio(w_med, level, curves[pde], "weak")
            matched_rows.append(
                {
                    "level": level,
                    "direction_scale": direction,
                    "n": len(strong_m),
                    "strong_mean": sm,
                    "strong_std": ss,
                    "weak_mean": wm,
                    "weak_std": ws,
                    "strong_median": s_med,
                    "weak_median": w_med,
                    "C_strong": c_s["C"],
                    "C_weak": c_w["C"],
                }
            )
        out["natural"][pde] = natural_rows
        out["matched"][pde] = matched_rows
    return out


def _score_named_fields(fields, ref, data, coef, spec, pde, curves) -> list[dict]:
    rows = []
    for i in range(ref.shape[0]):
        if pde == "burgers":
            scored = _score_burgers(fields[i], ref[i], data["x"], data["t"], float(coef[i]), spec)
        else:
            scored = _score_heat(fields[i], ref[i], data["x"], data["y"], data["t"], float(coef[i]), spec)
        rows.append(
            {
                **scored,
                "C_strong": _ratio(scored["strong"], scored["rel_l2"], curves, "strong"),
                "C_weak": _ratio(scored["weak"], scored["rel_l2"], curves, "weak"),
            }
        )
    return rows


def _pack_rows(rows: list[dict]) -> dict:
    def col(key):
        return [r[key] for r in rows]

    def cstat(key):
        return _mean_std_optional(
            [r[key]["C"] for r in rows if r[key]["C"] is not None and not r[key]["extrapolated"]]
        )

    rel = _mean_std_optional(col("rel_l2"))
    strong = _mean_std_optional(col("strong"))
    weak = _mean_std_optional(col("weak"))
    return {
        "rel_l2": rel,
        "strong": strong,
        "weak": weak,
        "C_strong": cstat("C_strong"),
        "C_weak": cstat("C_weak"),
        "per_traj": {
            "rel_l2": col("rel_l2"),
            "strong": col("strong"),
            "weak": col("weak"),
            "C_strong": [r["C_strong"]["C"] for r in rows],
            "C_weak": [r["C_weak"]["C"] for r in rows],
        },
    }


def run_models_v2(v1, summary, spec, burgers, heat, curves) -> dict:
    """Retrain the v1 selected configs. Test labels are not used to choose them."""
    train_cfg = v1["training"]
    architectures = ["fno", "unet", "deeponet"]
    results = {"models": {}, "baselines": {}, "paired": [], "retrain_max_abs_rel_diff": {}}
    retained = {}
    for pde, data in (("burgers", burgers), ("heat", heat)):
        block = v1["datasets"][pde]
        splits = split_indices(block)
        coef = data["nu"] if pde == "burgers" else data["alpha"]
        grid = {"nt": int(block["nt"])}
        if pde == "burgers":
            grid["nx"] = int(block["nx"])
            fields = _baseline_fields_burgers(data, splits["train"], splits["test"])
        else:
            grid["n"] = int(block["n"])
            fields = _baseline_fields_heat(data, splits["train"], splits["test"])
        ref = data["u"][splits["test"]]
        coef_test = coef[splits["test"]]
        pde_curves = curves[pde]
        baseline_pack = {}
        for name, batch in fields.items():
            rows = _score_named_fields(batch, ref, data, coef_test, spec, pde, pde_curves)
            baseline_pack[name] = _pack_rows(rows)
        trivials = {name: baseline_pack[name]["rel_l2"]["mean"] for name in ("zeros", "training_mean", "frozen_initial")}
        anchor = best_baseline(trivials)
        for name, pack in baseline_pack.items():
            if anchor is None or name == "reference" and pack["rel_l2"]["mean"] == 0:
                score = scaled_score(0.0, anchor[1], 0.0) if anchor and name == "reference" else None
            else:
                score = scaled_score(pack["rel_l2"]["mean"], anchor[1], 0.0)
            pack["S_rel_l2"] = score
        results["baselines"][pde] = baseline_pack

        model_pde = {}
        traj_store = {}
        max_diff = 0.0
        for arch in architectures:
            config = summary["models"]["hpo"][pde][arch]["selected_config"]
            per_seed = []
            store = {key: [] for key in ("rel_l2", "strong", "weak", "C_strong", "C_weak")}
            t0 = time.perf_counter()
            for seed in train_cfg["final_seeds"]:
                model = build_model(arch, pde, grid, config)
                train_model(
                    model,
                    data["u"][splits["train"]],
                    coef[splits["train"]],
                    data["u"][splits["val"]],
                    coef[splits["val"]],
                    epochs=int(train_cfg["epochs_final"]),
                    lr=float(train_cfg["lr"]),
                    weight_decay=float(train_cfg["weight_decay"]),
                    batch_size=int(train_cfg["batch_size"]),
                    seed=int(seed),
                )
                pred = predict(model, ref[:, 0], coef_test)
                rows = _score_named_fields(pred, ref, data, coef_test, spec, pde, pde_curves)
                pack = _pack_rows(rows)
                published = next(
                    r["rel_l2"]
                    for r in summary["models"]["models"][pde][arch]["per_seed"]
                    if int(r["seed"]) == int(seed)
                )
                max_diff = max(max_diff, abs(pack["rel_l2"]["mean"] - float(published)))
                per_seed.append(
                    {
                        "seed": int(seed),
                        "rel_l2": pack["rel_l2"]["mean"],
                        "strong": pack["strong"]["mean"],
                        "weak": pack["weak"]["mean"],
                        "C_strong": pack["C_strong"]["mean"],
                        "C_weak": pack["C_weak"]["mean"],
                        "S_rel_l2": None
                        if anchor is None
                        else scaled_score(pack["rel_l2"]["mean"], anchor[1], 0.0),
                        "n_params": parameter_count(model),
                    }
                )
                for key in store:
                    store[key].append(
                        [np.nan if v is None else float(v) for v in pack["per_traj"][key]]
                    )
            summary_pack = {
                key: _mean_std_optional([row[key] for row in per_seed])
                for key in ("rel_l2", "strong", "weak", "C_strong", "C_weak", "S_rel_l2")
            }
            model_pde[arch] = {
                "selected_config": config,
                "per_seed": per_seed,
                "summary": summary_pack,
                "seconds": time.perf_counter() - t0,
                "n_params": per_seed[0]["n_params"],
            }
            traj_store[arch] = {key: np.asarray(val, dtype=np.float64) for key, val in store.items()}
            print(
                f"[v2 {pde} {arch}] rel {summary_pack['rel_l2']['mean']:.4f} "
                f"C_s {summary_pack['C_strong']['mean']:.3f} C_w {summary_pack['C_weak']['mean']:.3f}",
                flush=True,
            )
        results["models"][pde] = model_pde
        results["retrain_max_abs_rel_diff"][pde] = max_diff
        retained[pde] = traj_store
        names = list(traj_store)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                for metric in v2_metrics():
                    diff = traj_store[a][metric] - traj_store[b][metric]
                    test = paired_wilcoxon(diff.ravel())
                    results["paired"].append({"pde": pde, "a": a, "b": b, "metric": metric, **test})
    results["anchor_note"] = "S uses relative L2 only. Residual columns are C."
    return results


def v2_metrics():
    return ["rel_l2", "strong", "weak", "C_strong", "C_weak"]


def run_shipped_v2(v1, v2, spec) -> list[dict]:
    from pathlib import Path

    from baselines.fno.io import load_checkpoint

    root = Path(__file__).resolve().parents[1]
    cfg = v1["shipped"]
    if float(v2["verdict_v2"]["legacy_threshold"]) != float(RESIDUAL_UNTRUSTED):
        raise AssertionError("v2 legacy threshold drifted from metrics.trust.RESIDUAL_UNTRUSTED")
    draws = [int(s) for s in v2["calibrated_ratio"]["draw_seeds_shipped"]]
    salt = int(v2["calibrated_ratio"]["rng_salt_shipped"])
    rows = []
    for pde, ckpt_name, coef_key in (
        ("burgers", "fno_burgers.pt", "nu"),
        ("burgers", "pino_burgers.pt", "nu"),
        ("heat", "fno_heat2d.pt", "alpha"),
        ("heat", "pino_heat2d.pt", "alpha"),
    ):
        shipped_block = cfg[pde]
        block = {**v1["datasets"][pde], **shipped_block}
        data = load_burgers(block) if pde == "burgers" else load_heat(block)
        path = root / "checkpoints" / ckpt_name
        loaded = load_checkpoint(path, device="cpu")
        model_name = "fno" if ckpt_name.startswith("fno") else "pino"
        if loaded["status"] != "ok":
            rows.append(
                {
                    "pde": pde,
                    "model": model_name,
                    "checkpoint": ckpt_name,
                    "status": loaded["status"],
                    "reason": loaded.get("reason"),
                }
            )
            continue
        split = (loaded.get("extra") or {}).get("split") or {}
        idx = split.get("eval_idx")
        if not idx:
            n_eval = 12 if pde == "burgers" else 8
            idx = list(range(data["u"].shape[0] - n_eval, data["u"].shape[0]))
        idx = np.asarray(idx, dtype=int)
        subset = {
            "u": data["u"][idx],
            "x": data["x"],
            "t": data["t"],
            coef_key: data[coef_key][idx],
        }
        if pde == "heat":
            subset["y"] = data["y"]
        print(f"[v2 shipped envelope {pde} {model_name}] n_eval={len(idx)}", flush=True)
        curves = _noise_curves(v1, spec, subset, pde, draws, salt)
        coef = subset[coef_key]
        ref = subset["u"]

        def add_row(label, fields, checkpoint):
            scored_rows = _score_named_fields(fields, ref, subset, coef, spec, pde, curves)
            rel = float(np.mean([r["rel_l2"] for r in scored_rows]))
            strong = float(np.mean([r["strong"] for r in scored_rows]))
            weak = float(np.mean([r["weak"] for r in scored_rows]))
            c_s = _ratio(strong, rel, curves, "strong")
            c_w = _ratio(weak, rel, curves, "weak")
            legacy = bool(strong > float(v2["verdict_v2"]["legacy_threshold"]))
            rows.append(
                {
                    "pde": pde,
                    "model": label,
                    "checkpoint": checkpoint,
                    "status": "ok",
                    "n_eval": int(len(idx)),
                    "rel_l2": rel,
                    "strong": strong,
                    "weak": weak,
                    "C_strong": c_s["C"],
                    "C_weak": c_w["C"],
                    "weak_p95": c_w["p95"],
                    "smooth_weak_median": c_w["median"],
                    "extrapolated": c_w["extrapolated"],
                    "legacy_untrusted": legacy,
                    "verdict_survives_weak_form": c_w["above_p95"],
                }
            )

        pred = predict(loaded["model"], ref[:, 0], coef)
        add_row(model_name, pred, ckpt_name)
        if model_name == "fno":
            if pde == "burgers":
                frozen = np.repeat(ref[:, 0:1, :], ref.shape[1], axis=1)
            else:
                frozen = np.repeat(ref[:, 0:1, :, :], ref.shape[1], axis=1)
            add_row("frozen_initial", frozen, None)
            add_row("reference", ref, None)
    return rows


def _story(curves, mismatch) -> dict:
    notes = {}
    for pde, block in curves.items():
        levels = block["levels"]
        strong_ratio, weak_ratio, mis_s, mis_w = [], [], [], []
        matched = {row["level"]: row for row in mismatch["matched"][pde]}
        for i, level in enumerate(levels):
            s_w = block["white"]["strong"]["median"][i]
            s_s = block["smooth"]["strong"]["median"][i]
            w_w = block["white"]["weak"]["median"][i]
            w_s = block["smooth"]["weak"]["median"][i]
            strong_ratio.append(s_w / s_s)
            weak_ratio.append(w_w / w_s)
            mis_s.append(matched[level]["strong_median"] / s_s)
            mis_w.append(matched[level]["weak_median"] / w_s)
        notes[pde] = {
            "strong_white_over_smooth": strong_ratio,
            "weak_white_over_smooth": weak_ratio,
            "strong_mismatch_over_smooth": mis_s,
            "weak_mismatch_over_smooth": mis_w,
        }
    return notes


def run_amendment(v1: dict, v2: dict, summary: dict) -> dict:
    started = time.perf_counter()
    spec = spec_from_protocol(v1["weak_form"])
    print("v2 datasets...", flush=True)
    burgers = load_burgers(v1["datasets"]["burgers"])
    heat = load_heat(v1["datasets"]["heat"])
    draws = [int(s) for s in v2["calibrated_ratio"]["draw_seeds_paper"]]
    salt = int(v2["calibrated_ratio"]["rng_salt_paper"])
    curves = {}
    t0 = time.perf_counter()
    print("v2 smooth/white bands...", flush=True)
    curves["burgers"] = _noise_curves(v1, spec, burgers, "burgers", draws, salt)
    curves["heat"] = _noise_curves(v1, spec, heat, "heat", draws, salt)
    t_noise = time.perf_counter() - t0
    t0 = time.perf_counter()
    print("v2 mismatch ladder...", flush=True)
    mismatch = run_mismatch_ladder(v1, v2, spec, burgers, heat, curves)
    t_mis = time.perf_counter() - t0
    t0 = time.perf_counter()
    print("v2 model retrain...", flush=True)
    models = run_models_v2(v1, summary, spec, burgers, heat, curves)
    t_models = time.perf_counter() - t0
    t0 = time.perf_counter()
    print("v2 shipped...", flush=True)
    shipped = run_shipped_v2(v1, v2, spec)
    t_ship = time.perf_counter() - t0
    timings = {
        "noise_bands": t_noise,
        "mismatch": t_mis,
        "models": t_models,
        "shipped": t_ship,
        "total": time.perf_counter() - started,
    }
    print(f"v2 done in {timings['total']:.1f}s", flush=True)
    return {
        "curves": curves,
        "mismatch": mismatch,
        "models": models,
        "shipped": shipped,
        "story": _story(curves, mismatch),
        "timings_seconds": timings,
    }
