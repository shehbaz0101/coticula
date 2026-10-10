"""Pre-registered noise, model, symbol, and inverse runs.

Reads every knob from the protocol dict. Does not modify the default Conserve
metric or the trust rule.
"""
from __future__ import annotations

import time

import numpy as np

from baselines.classical.burgers1d import solve_burgers
from baselines.classical.heat2d import solve_heat2d
from metrics.predict import relative_l2
from metrics.trust import RESIDUAL_UNTRUSTED
from metrics.weak_form import WeakSpec, burgers_weak_per_traj, heat_weak_per_traj, spec_from_protocol
from research.datautil import load_burgers, load_heat, split_indices
from research.inverse_nu import run_inverse
from research.models import build_model, parameter_count
from research.noise import (
    realized_rel,
    smooth_error_burgers,
    smooth_error_heat,
    white_error,
)
from research.statsutil import (
    best_baseline,
    bootstrap_mean_ci,
    kendall,
    linear_fit,
    log_interp,
    paired_permutation,
    scaled_score,
)
from research.strong import (
    burgers_residual_field,
    burgers_strong_per_traj,
    burgers_white_second_moment_field,
    heat_residual_field,
    heat_strong_per_traj,
    heat_white_C_h,
    heat_white_second_moment,
    keff_dirichlet,
    keff_periodic,
    mean_std,
)
from research.train_study import predict, train_model


def _spec_with_k(base: WeakSpec, k_phi: int) -> WeakSpec:
    return WeakSpec(
        p_bump=base.p_bump,
        k_phi=int(k_phi),
        time_centers_frac=base.time_centers_frac,
        time_radius_frac=base.time_radius_frac,
    )


def _inject(reference, kind, epsilon, rng, *, pde, coords, noise_cfg):
    if kind == "white":
        return white_error(reference, epsilon, rng, dirichlet=(pde == "heat"))
    if pde == "burgers":
        return smooth_error_burgers(
            reference,
            coords["x"],
            coords["t"],
            epsilon,
            rng,
            k_smooth=int(noise_cfg["k_smooth"]),
            n_time_modes=int(noise_cfg["n_time_modes"]),
        )
    return smooth_error_heat(
        reference,
        coords["x"],
        coords["y"],
        coords["t"],
        epsilon,
        rng,
        k_smooth=int(noise_cfg["k_smooth"]),
        n_time_modes=int(noise_cfg["n_time_modes"]),
    )


def _score_burgers(field, ref, x, t, nu, spec: WeakSpec):
    dx = float(x[1] - x[0])
    dt = float(t[1] - t[0])
    rel = relative_l2(field, ref)
    strong = float(burgers_strong_per_traj(field[None], dx, dt, [nu])[0])
    weak = float(burgers_weak_per_traj(field, x, t, nu, spec, u_ref=ref)[0])
    err = field - ref
    return {
        "rel_l2": rel,
        "strong": strong,
        "weak": weak,
        "keff": keff_periodic(err, dx),
        "keff_h": keff_periodic(err, dx) * dx,
    }


def _score_heat(field, ref, x, y, t, alpha, spec: WeakSpec):
    dx = float(x[1] - x[0])
    dt = float(t[1] - t[0])
    rel = relative_l2(field, ref)
    strong = float(heat_strong_per_traj(field[None], dx, dt, [alpha])[0])
    weak = float(heat_weak_per_traj(field, x, y, t, alpha, spec, u_ref=ref)[0])
    keff = keff_dirichlet(field - ref, dx)
    return {
        "rel_l2": rel,
        "strong": strong,
        "weak": weak,
        "keff": keff,
        "keff_h": keff * dx,
    }


def _mean_std_records(rows: list[dict], key: str) -> dict:
    # rows grouped: average trajectories inside each seed, then mean/std over seeds
    by_seed: dict[int, list[float]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(float(row[key]))
    per_seed = [float(np.mean(vals)) for vals in by_seed.values()]
    mean, std = mean_std(per_seed)
    return {"mean": mean, "std": std, "n_seeds": len(per_seed)}


def run_noise(protocol, spec, burgers, heat) -> dict:
    cfg = protocol["noise"]
    levels = [float(v) for v in cfg["levels"]]
    seeds = [int(v) for v in cfg["seeds"]]
    out = {"levels": levels, "seeds": seeds, "pdes": {}}
    paired = {}
    for pde, data, coef_key in (
        ("burgers", burgers, "nu"),
        ("heat", heat, "alpha"),
    ):
        coords = {"x": data["x"], "t": data["t"]}
        if pde == "heat":
            coords["y"] = data["y"]
        rows = []
        clean_strong, clean_weak = [], []
        for i in range(data["u"].shape[0]):
            ref = data["u"][i]
            coef = float(data[coef_key][i])
            if pde == "burgers":
                clean = _score_burgers(ref, ref, data["x"], data["t"], coef, spec)
            else:
                clean = _score_heat(ref, ref, data["x"], data["y"], data["t"], coef, spec)
            clean_strong.append(clean["strong"])
            clean_weak.append(clean["weak"])
            for seed in seeds:
                for level in levels:
                    for kind in ("white", "smooth"):
                        # Distinct stream per (pde, traj, seed, level, kind) so the
                        # paired white/smooth comparison does not share one draw.
                        rng = np.random.default_rng(
                            np.random.SeedSequence(
                                [
                                    seed,
                                    i,
                                    int(round(level * 1_000_000)),
                                    1 if kind == "white" else 2,
                                    1 if pde == "burgers" else 2,
                                ]
                            )
                        )
                        error = _inject(ref, kind, level, rng, pde=pde, coords=coords, noise_cfg=cfg)
                        field = ref + error
                        if pde == "burgers":
                            scored = _score_burgers(field, ref, data["x"], data["t"], coef, spec)
                        else:
                            scored = _score_heat(
                                field, ref, data["x"], data["y"], data["t"], coef, spec
                            )
                        rows.append(
                            {
                                "traj": i,
                                "seed": seed,
                                "level": level,
                                "kind": kind,
                                "rel_l2": scored["rel_l2"],
                                "strong": scored["strong"],
                                "weak": scored["weak"],
                                "keff_h": scored["keff_h"],
                            }
                        )
        table = []
        for level in levels:
            for kind in ("white", "smooth"):
                subset = [r for r in rows if r["kind"] == kind and abs(r["level"] - level) < 1e-12]
                entry = {"level": level, "kind": kind}
                for key in ("rel_l2", "strong", "weak", "keff_h"):
                    entry[key] = _mean_std_records(subset, key)
                table.append(entry)
        pair_rows = []
        primary = float(cfg["primary_level_for_tests"])
        for seed in seeds:
            for i in range(data["u"].shape[0]):
                white = next(
                    r
                    for r in rows
                    if r["seed"] == seed and r["traj"] == i and r["kind"] == "white" and abs(r["level"] - primary) < 1e-12
                )
                smooth = next(
                    r
                    for r in rows
                    if r["seed"] == seed and r["traj"] == i and r["kind"] == "smooth" and abs(r["level"] - primary) < 1e-12
                )
                pair_rows.append(
                    {
                        "strong": white["strong"] - smooth["strong"],
                        "weak": white["weak"] - smooth["weak"],
                    }
                )
        tests = {}
        stats = protocol["statistics"]
        for key in ("strong", "weak"):
            tests[key] = paired_permutation(
                np.array([r[key] for r in pair_rows]),
                n_perm=int(stats["n_perm"]),
                seed=int(stats["seed"]),
            )
        cs, _ = mean_std(clean_strong)
        cw, _ = mean_std(clean_weak)
        out["pdes"][pde] = {
            "table": table,
            "clean": {"strong_mean": cs, "weak_mean": cw, "n_traj": int(data["u"].shape[0])},
            "paired_white_minus_smooth": tests,
        }
        paired[pde] = rows
    out["_rows"] = paired
    return out


def _curve(noise_pde: dict, kind: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    levels, values = [], []
    for row in noise_pde["table"]:
        if row["kind"] == kind:
            levels.append(row["level"])
            values.append(row[key]["mean"])
    return np.asarray(levels), np.asarray(values)


def run_resolution(protocol, spec) -> dict:
    cfg = protocol["resolution_sweep"]
    epsilon = float(cfg["epsilon"])
    seeds = [int(s) for s in cfg["seeds"]]
    n_traj = int(cfg["n_traj"])
    rows = []

    def evaluate(pde: str, data: dict, coef_key: str, sweep: str, grid: dict):
        dx = float(data["x"][1] - data["x"][0])
        dt = float(data["t"][1] - data["t"][0])
        coords = {"x": data["x"], "t": data["t"]}
        if pde == "heat":
            coords["y"] = data["y"]
        bucket = {kind: {"strong": [], "weak": [], "theory": []} for kind in ("white", "smooth")}
        for i in range(data["u"].shape[0]):
            ref = data["u"][i]
            coef = float(data[coef_key][i])
            if pde == "heat":
                theory_white = heat_white_C_h(dt, dx, coef, dim=2) * epsilon
            else:
                s2 = (epsilon * float(np.linalg.norm(ref))) ** 2 / ref.size
                pred = burgers_white_second_moment_field(ref, dx, dt, coef, s2)
                umid = 0.5 * (ref[:-1] + ref[1:])
                step = np.sqrt(pred.sum(axis=-1)) / (np.linalg.norm(umid, axis=-1) + 1e-12)
                theory_white = float(step.mean())
            for seed in seeds:
                for kind in ("white", "smooth"):
                    rng = np.random.default_rng(
                        np.random.SeedSequence(
                            [seed, i, 1 if kind == "white" else 2, int(grid.get("nx", grid.get("n"))), int(grid["nt"])]
                        )
                    )
                    error = _inject(ref, kind, epsilon, rng, pde=pde, coords=coords, noise_cfg=protocol["noise"])
                    field = ref + error
                    if pde == "burgers":
                        scored = _score_burgers(field, ref, data["x"], data["t"], coef, spec)
                    else:
                        scored = _score_heat(field, ref, data["x"], data["y"], data["t"], coef, spec)
                    bucket[kind]["strong"].append(scored["strong"])
                    bucket[kind]["weak"].append(scored["weak"])
                    bucket[kind]["theory"].append(theory_white if kind == "white" else float("nan"))
        h = dx
        # Mesh ratio uses the mean coefficient.
        coef_mean = float(np.mean(data[coef_key]))
        if pde == "heat":
            r_mesh = coef_mean * dt / h**2
            dim = 2
        else:
            r_mesh = coef_mean * dt / h**2
            dim = 1
        row = {
            "pde": pde,
            "sweep": sweep,
            "n": int(grid.get("nx", grid.get("n"))),
            "nt": int(grid["nt"]),
            "h": h,
            "dt": dt,
            "mesh_ratio_r": r_mesh,
            "dim": dim,
            "coef_mean": coef_mean,
            "time_term": 2.0 / dt**2,
            "space_term": (dim * (2 * dim + 1) * coef_mean**2) / h**4 if pde == "heat" else None,
        }
        for kind in ("white", "smooth"):
            sm, ss = mean_std(bucket[kind]["strong"])
            wm, ws = mean_std(bucket[kind]["weak"])
            row[f"strong_{kind}_mean"] = sm
            row[f"strong_{kind}_std"] = ss
            row[f"weak_{kind}_mean"] = wm
            row[f"weak_{kind}_std"] = ws
        tm, ts = mean_std(bucket["white"]["theory"])
        row["theory_white_mean"] = tm
        row["theory_white_std"] = ts
        row["strong_white_over_theory"] = float(row["strong_white_mean"] / tm) if tm else None
        rows.append(row)

    bcfg = protocol["datasets"]["burgers"]
    hcfg = protocol["datasets"]["heat"]
    for sweep, grids in cfg["burgers"].items():
        for grid in grids:
            block = dict(bcfg)
            block.update({"nx": int(grid["nx"]), "nt": int(grid["nt"]), "n_traj": n_traj})
            block["seed"] = int(cfg["data_seed_burgers"]) + int(grid["nx"]) + 1000 * int(grid["nt"])
            evaluate("burgers", load_burgers(block), "nu", sweep, grid)
    for sweep, grids in cfg["heat"].items():
        for grid in grids:
            block = dict(hcfg)
            block.update({"n": int(grid["n"]), "nt": int(grid["nt"]), "n_traj": n_traj})
            block["seed"] = int(cfg["data_seed_heat"]) + int(grid["n"]) + 1000 * int(grid["nt"])
            evaluate("heat", load_heat(block), "alpha", sweep, grid)
    return {"epsilon": epsilon, "rows": rows}


def run_symbol(protocol, burgers, heat) -> dict:
    """Algebraic second-moment checks and the static effective-wavenumber identity."""
    cfg = protocol["symbol_check"]
    reps = int(cfg["synthetic_reps"])
    nt, n, _n = cfg["synthetic_shape"]
    checks = []
    for h, dt, alpha in cfg["heat_cases"]:
        h, dt, alpha = float(h), float(dt), float(alpha)
        theory = heat_white_second_moment(dt, h, alpha, dim=2)
        emp = []
        for rep in range(reps):
            rng = np.random.default_rng(10_000 + rep)
            field = rng.normal(size=(int(nt), int(n), int(n)))
            residual = heat_residual_field(field, h, dt, alpha)
            emp.append(float((residual**2).mean()))
        em, es = mean_std(emp)
        checks.append(
            {
                "h": h,
                "dt": dt,
                "alpha": alpha,
                "theory": theory,
                "empirical_mean": em,
                "empirical_std": es,
                "ratio": em / theory,
            }
        )
    # Burgers formula on the first paper trajectory, known variance, no rel-L2 rescaling.
    ref = burgers["u"][0]
    dx = float(burgers["x"][1] - burgers["x"][0])
    dt = float(burgers["t"][1] - burgers["t"][0])
    nu = float(burgers["nu"][0])
    s2 = float(cfg["burgers_s2"])
    tau = burgers_residual_field(ref, dx, dt, nu)
    pred = burgers_white_second_moment_field(ref, dx, dt, nu, s2)
    emp_b = []
    for rep in range(reps):
        rng = np.random.default_rng(20_000 + rep)
        error = rng.normal(scale=np.sqrt(s2), size=ref.shape)
        residual = burgers_residual_field(ref + error, dx, dt, nu)
        emp_b.append(float(((residual - tau) ** 2).mean()))
    bm, bs = mean_std(emp_b)
    burgers_check = {
        "s2": s2,
        "theory_mean": float(pred.mean()),
        "empirical_mean": bm,
        "empirical_std": bs,
        "ratio": bm / float(pred.mean()),
    }
    # Static heat identity: error constant in time, Dirichlet sine-compatible noise.
    href = heat["u"][0]
    hdx = float(heat["x"][1] - heat["x"][0])
    hdt = float(heat["t"][1] - heat["t"][0])
    alpha = float(heat["alpha"][0])
    rng = np.random.default_rng(int(cfg["static_seed"]))
    spatial = rng.normal(size=href.shape[1:])
    spatial[0, :] = spatial[-1, :] = spatial[:, 0] = spatial[:, -1] = 0.0
    error = np.repeat(spatial[None, ...], href.shape[0], axis=0)
    error *= 0.01 * np.linalg.norm(href) / (np.linalg.norm(error) + 1e-12)
    residual = heat_residual_field(error, hdx, hdt, alpha)
    gains = []
    e_norm = float(np.linalg.norm(error[0, 1:-1, 1:-1]))
    for step in range(residual.shape[0]):
        gains.append(float(np.linalg.norm(residual[step, 1:-1, 1:-1]) / (e_norm + 1e-12)))
    keff = keff_dirichlet(error[0], hdx)
    static = {
        "alpha": alpha,
        "gain_mean": float(np.mean(gains)),
        "alpha_keff_sq": float(alpha * keff**2),
        "ratio": float(np.mean(gains) / (alpha * keff**2 + 1e-18)),
        "keff_h": float(keff * hdx),
    }
    heat_ratios = np.array([c["ratio"] for c in checks])
    fit = linear_fit(np.log([c["theory"] for c in checks]), np.log([c["empirical_mean"] for c in checks]))
    return {
        "heat_cases": checks,
        "heat_ratio_mean": float(heat_ratios.mean()),
        "heat_log_fit": fit,
        "burgers": burgers_check,
        "heat_static_identity": static,
    }


def run_mismatch(protocol, spec, burgers, heat) -> dict:
    scales = [float(s) for s in protocol["coefficient_mismatch_scales"]]
    out = {}
    for pde, data, key in (("burgers", burgers, "nu"), ("heat", heat, "alpha")):
        rows = []
        for scale in scales:
            strong, weak = [], []
            for i in range(data["u"].shape[0]):
                coef = float(data[key][i]) * scale
                ref = data["u"][i]
                if pde == "burgers":
                    scored = _score_burgers(ref, ref, data["x"], data["t"], coef, spec)
                else:
                    scored = _score_heat(ref, ref, data["x"], data["y"], data["t"], coef, spec)
                strong.append(scored["strong"])
                weak.append(scored["weak"])
            sm, ss = mean_std(strong)
            wm, ws = mean_std(weak)
            rows.append(
                {
                    "scale": scale,
                    "strong_mean": sm,
                    "strong_std": ss,
                    "weak_mean": wm,
                    "weak_std": ws,
                }
            )
        out[pde] = rows
    return out


def run_kphi(protocol, spec, burgers, heat) -> dict:
    cfg = protocol["kphi_ablation"]
    epsilon = float(cfg["epsilon"])
    seeds = [int(s) for s in cfg["seeds"]]
    rows = []
    for k_phi in cfg["values"]:
        local = _spec_with_k(spec, int(k_phi))
        for pde, data, key in (("burgers", burgers, "nu"), ("heat", heat, "alpha")):
            coords = {"x": data["x"], "t": data["t"]}
            if pde == "heat":
                coords["y"] = data["y"]
            clean, frozen, white, smooth = [], [], [], []
            for i in range(data["u"].shape[0]):
                ref = data["u"][i]
                coef = float(data[key][i])
                if pde == "burgers":
                    clean.append(_score_burgers(ref, ref, data["x"], data["t"], coef, local)["weak"])
                    held = np.repeat(ref[0:1], ref.shape[0], axis=0)
                    frozen.append(_score_burgers(held, ref, data["x"], data["t"], coef, local)["weak"])
                else:
                    clean.append(_score_heat(ref, ref, data["x"], data["y"], data["t"], coef, local)["weak"])
                    held = np.repeat(ref[0:1], ref.shape[0], axis=0)
                    frozen.append(_score_heat(held, ref, data["x"], data["y"], data["t"], coef, local)["weak"])
                for seed in seeds:
                    for kind, bucket in (("white", white), ("smooth", smooth)):
                        rng = np.random.default_rng(np.random.SeedSequence([9000 + int(k_phi), seed, i, 1 if kind == "white" else 2]))
                        error = _inject(ref, kind, epsilon, rng, pde=pde, coords=coords, noise_cfg=protocol["noise"])
                        field = ref + error
                        if pde == "burgers":
                            bucket.append(_score_burgers(field, ref, data["x"], data["t"], coef, local)["weak"])
                        else:
                            bucket.append(_score_heat(field, ref, data["x"], data["y"], data["t"], coef, local)["weak"])
            rows.append(
                {
                    "pde": pde,
                    "k_phi": int(k_phi),
                    "weak_clean": mean_std(clean)[0],
                    "weak_frozen": mean_std(frozen)[0],
                    "weak_white": mean_std(white)[0],
                    "weak_smooth": mean_std(smooth)[0],
                }
            )
    return {"epsilon": epsilon, "rows": rows}


def _baseline_fields_burgers(data, idx_train, idx_test):
    ref = data["u"][idx_test]
    zeros = np.zeros_like(ref)
    mean_field = np.repeat(data["u"][idx_train].mean(axis=0)[None, ...], len(idx_test), axis=0)
    frozen = np.repeat(ref[:, 0:1, :], ref.shape[1], axis=1)
    return {"zeros": zeros, "training_mean": mean_field, "frozen_initial": frozen, "reference": ref}


def _baseline_fields_heat(data, idx_train, idx_test):
    ref = data["u"][idx_test]
    zeros = np.zeros_like(ref)
    mean_field = np.repeat(data["u"][idx_train].mean(axis=0)[None, ...], len(idx_test), axis=0)
    frozen = np.repeat(ref[:, 0:1, :, :], ref.shape[1], axis=1)
    return {"zeros": zeros, "training_mean": mean_field, "frozen_initial": frozen, "reference": ref}


def _score_many_burgers(fields, ref, data, nu, spec):
    out = []
    for i in range(ref.shape[0]):
        out.append(_score_burgers(fields[i], ref[i], data["x"], data["t"], float(nu[i]), spec))
    return out


def _score_many_heat(fields, ref, data, alpha, spec):
    out = []
    for i in range(ref.shape[0]):
        out.append(_score_heat(fields[i], ref[i], data["x"], data["y"], data["t"], float(alpha[i]), spec))
    return out


def _pool(scored_list, key):
    return np.array([row[key] for row in scored_list], dtype=np.float64)


def _finite_mean(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def _coef_pair_burgers(pred, pred_cf, ref, ref_cf):
    delta_m = pred_cf - pred
    delta_s = ref_cf - ref
    denom = float(np.linalg.norm(delta_s))
    numer = float(np.linalg.norm(delta_m))
    s_coef = numer / denom if denom > 1e-12 else None
    if numer > 1e-12 and denom > 1e-12:
        cosine = float(np.sum(delta_m * delta_s) / (numer * denom))
    else:
        cosine = None
    return {
        "s_coef": s_coef,
        "cosine": cosine,
        "cf_rel_l2": relative_l2(pred_cf, ref_cf),
    }


def _classical_burgers_cf(data, idx, gamma):
    x, t = data["x"], data["t"]
    dx = float(x[1] - x[0])
    length = dx * len(x)
    horizon = float(t[-1])
    nt = len(t) - 1
    sols = []
    for i in idx:
        sols.append(
            solve_burgers(
                nu=float(data["nu"][i]) * gamma,
                nx=len(x),
                nt=nt,
                L=length,
                T=horizon,
                u0=data["u"][i, 0],
            )["u"]
        )
    return np.stack(sols, axis=0)


def _classical_heat_cf(data, idx, gamma):
    x, t = data["x"], data["t"]
    nt = len(t) - 1
    n = len(x)
    sols = []
    for i in idx:
        sols.append(
            solve_heat2d(
                alpha=float(data["alpha"][i]) * gamma,
                n=n,
                nt=nt,
                L=float(x[-1]),
                T=float(t[-1]),
                u0=data["u"][i, 0],
            )["u"]
        )
    return np.stack(sols, axis=0)


def _summarize_model_rows(per_seed: list[dict]) -> dict:
    summary = {"n_seeds": len(per_seed)}
    for key in ("rel_l2", "strong", "weak", "s_coef", "cosine", "cf_rel_l2", "keff_h"):
        vals = [row[key] for row in per_seed if row.get(key) is not None]
        mean, std = mean_std(vals)
        summary[key] = {"mean": mean, "std": std}
    return summary


def _average_traj(per_seed_traj: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(per_seed_traj, axis=0), axis=0)


def run_models(protocol, spec, burgers, heat, noise) -> dict:
    train_cfg = protocol["training"]
    gamma = float(protocol["coefficient_response"]["gamma"])
    stats = protocol["statistics"]
    architectures = ["fno", "unet", "deeponet"]
    results = {"hpo": {}, "models": {}, "baselines": {}, "scores": {}, "associations": {}, "wall": {}}
    retained: dict = {}
    datasets = {"burgers": burgers, "heat": heat}

    for pde in ("burgers", "heat"):
        data = datasets[pde]
        block = protocol["datasets"]["burgers" if pde == "burgers" else "heat"]
        splits = split_indices(block)
        coef = data["nu"] if pde == "burgers" else data["alpha"]
        grid = {"nt": int(block["nt"])}
        if pde == "burgers":
            grid["nx"] = int(block["nx"])
            fields = _baseline_fields_burgers(data, splits["train"], splits["test"])
            ref_cf = _classical_burgers_cf(data, splits["test"], gamma)
        else:
            grid["n"] = int(block["n"])
            fields = _baseline_fields_heat(data, splits["train"], splits["test"])
            ref_cf = _classical_heat_cf(data, splits["test"], gamma)
        ref = data["u"][splits["test"]]
        coef_test = coef[splits["test"]]

        def score_batch(batch):
            if pde == "burgers":
                return _score_many_burgers(batch, ref, data, coef_test, spec)
            return _score_many_heat(batch, ref, data, coef_test, spec)

        baseline_summary = {}
        baseline_traj = {}
        for name, batch in fields.items():
            scored = score_batch(batch)
            # Coefficient response: zeros, mean, and frozen ignore the coefficient.
            if name == "reference":
                pred_cf = ref_cf
                pred = batch
            elif name == "frozen_initial":
                pred = batch
                if pde == "burgers":
                    pred_cf = np.repeat(ref[:, 0:1, :], ref.shape[1], axis=1)
                else:
                    pred_cf = np.repeat(ref[:, 0:1, :, :], ref.shape[1], axis=1)
            else:
                pred = batch
                pred_cf = batch
            coef_rows = [
                _coef_pair_burgers(pred[i], pred_cf[i], ref[i], ref_cf[i]) for i in range(len(ref))
            ]
            baseline_summary[name] = {
                "rel_l2": float(np.mean(_pool(scored, "rel_l2"))),
                "strong": float(np.mean(_pool(scored, "strong"))),
                "weak": float(np.mean(_pool(scored, "weak"))),
                "keff_h": _finite_mean(_pool(scored, "keff_h"))),
                "s_coef": _nanmean([r["s_coef"] for r in coef_rows]),
                "cosine": _nanmean([r["cosine"] for r in coef_rows]),
                "cf_rel_l2": float(np.mean([r["cf_rel_l2"] for r in coef_rows])),
            }
            baseline_traj[name] = {
                "rel_l2": _pool(scored, "rel_l2"),
                "strong": _pool(scored, "strong"),
                "weak": _pool(scored, "weak"),
            }
        results["baselines"][pde] = baseline_summary

        hpo_pde = {}
        model_pde = {}
        assoc_x, assoc_y, assoc_x_heat, rank_s, rank_w = [], [], [], [], []
        for arch in architectures:
            trials = train_cfg["architectures_config"][arch]["trials"]
            hpo_rows = []
            best = None
            t_hpo = time.perf_counter()
            for index, trial in enumerate(trials):
                model = build_model(arch, pde, grid, trial)
                fitted = train_model(
                    model,
                    data["u"][splits["train"]],
                    coef[splits["train"]],
                    data["u"][splits["val"]],
                    coef[splits["val"]],
                    epochs=int(train_cfg["epochs_hpo"]),
                    lr=float(train_cfg["lr"]),
                    weight_decay=float(train_cfg["weight_decay"]),
                    batch_size=int(train_cfg["batch_size"]),
                    seed=int(train_cfg["hpo_seed"]),
                )
                row = {
                    "trial": index,
                    "config": trial,
                    "val_rel_l2": fitted["val_rel_l2"],
                    "n_params": parameter_count(model),
                }
                hpo_rows.append(row)
                if best is None or fitted["val_rel_l2"] < best["val_rel_l2"]:
                    best = row
            hpo_pde[arch] = {
                "trials": hpo_rows,
                "selected_trial": best["trial"],
                "selected_config": best["config"],
                "seconds": time.perf_counter() - t_hpo,
            }
            per_seed = []
            traj = {key: [] for key in ("rel_l2", "strong", "weak", "s_coef", "cf_rel_l2", "keff_sq_rel")}
            t_train = time.perf_counter()
            for seed in train_cfg["final_seeds"]:
                model = build_model(arch, pde, grid, best["config"])
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
                pred = predict(model, ref[:, 0] if pde == "burgers" else ref[:, 0], coef_test)
                if pde == "burgers":
                    pred_cf = predict(model, ref[:, 0], coef_test * gamma)
                else:
                    pred_cf = predict(model, ref[:, 0], coef_test * gamma)
                scored = score_batch(pred)
                coef_rows = [
                    _coef_pair_burgers(pred[i], pred_cf[i], ref[i], ref_cf[i]) for i in range(len(ref))
                ]
                per_seed.append(
                    {
                        "seed": int(seed),
                        "rel_l2": float(np.mean(_pool(scored, "rel_l2"))),
                        "strong": float(np.mean(_pool(scored, "strong"))),
                        "weak": float(np.mean(_pool(scored, "weak"))),
                        "keff_h": _finite_mean(_pool(scored, "keff_h"))),
                        "s_coef": _nanmean([r["s_coef"] for r in coef_rows]),
                        "cosine": _nanmean([r["cosine"] for r in coef_rows]),
                        "cf_rel_l2": float(np.mean([r["cf_rel_l2"] for r in coef_rows])),
                        "n_params": parameter_count(model),
                    }
                )
                traj["rel_l2"].append(_pool(scored, "rel_l2"))
                traj["strong"].append(_pool(scored, "strong"))
                traj["weak"].append(_pool(scored, "weak"))
                traj["s_coef"].append(np.array([np.nan if r["s_coef"] is None else r["s_coef"] for r in coef_rows]))
                traj["cf_rel_l2"].append(np.array([r["cf_rel_l2"] for r in coef_rows]))
                keff_sq_rel = _pool(scored, "keff") ** 2 * _pool(scored, "rel_l2")
                traj["keff_sq_rel"].append(keff_sq_rel)
                for i in range(len(ref)):
                    assoc_x.append(float(keff_sq_rel[i]))
                    assoc_y.append(float(scored[i]["strong"]))
                    if pde == "heat":
                        assoc_x_heat.append(float(coef_test[i]) * float(keff_sq_rel[i]))
                    rank_s.append(float(scored[i]["strong"]))
                    rank_w.append(float(scored[i]["weak"]))
            summary = _summarize_model_rows(per_seed)
            # Scaled scores per seed, then mean/std.
            score_pack = {}
            for metric, exclude, ref_value in (
                ("rel_l2", (), 0.0),
                ("strong", ("zeros",), baseline_summary["reference"]["strong"]),
                ("weak", ("zeros",), baseline_summary["reference"]["weak"]),
                ("cf_rel_l2", (), 0.0),
            ):
                trivials = {name: baseline_summary[name][metric] for name in ("zeros", "training_mean", "frozen_initial")}
                per_scores = []
                anchors = None
                for row in per_seed:
                    anchor = best_baseline(trivials, exclude=exclude)
                    anchors = anchor
                    if anchor is None or row[metric] is None:
                        per_scores.append(None)
                    else:
                        per_scores.append(scaled_score(row[metric], anchor[1], ref_value))
                if any(v is None for v in per_scores):
                    score_pack[metric] = {
                        "mean": None,
                        "std": None,
                        "undefined_reason": "best trivial baseline does no worse than the reference, or a seed was missing",
                        "baseline_name": None if anchors is None else anchors[0],
                        "baseline_value": None if anchors is None else anchors[1],
                        "reference": ref_value,
                    }
                else:
                    mean, std = mean_std(per_scores)
                    score_pack[metric] = {
                        "mean": mean,
                        "std": std,
                        "undefined_reason": None,
                        "baseline_name": anchors[0],
                        "baseline_value": anchors[1],
                        "reference": ref_value,
                    }
            # Bootstrap over test trajectories of the seed-averaged metric.
            boots = {}
            for metric in ("rel_l2", "strong", "weak"):
                averaged = _average_traj(traj[metric])
                boots[metric] = bootstrap_mean_ci(
                    averaged, n_boot=int(stats["n_boot"]), seed=int(stats["seed"])
                )
            levels, white_w = _curve(noise["pdes"][pde], "white", "weak")
            _levels_s, white_s = _curve(noise["pdes"][pde], "white", "strong")
            env_w = log_interp(levels, white_w, summary["rel_l2"]["mean"])
            env_s = log_interp(levels, white_s, summary["rel_l2"]["mean"])
            retained.setdefault(pde, {})[arch] = {
                metric: _average_traj(traj[metric]).tolist() for metric in ("rel_l2", "strong", "weak")
            }
            model_pde[arch] = {
                "summary": summary,
                "per_seed": per_seed,
                "scaled": score_pack,
                "bootstrap": boots,
                "white_envelope_weak": env_w,
                "white_envelope_strong": env_s,
                "weak_over_white_envelope": (
                    None
                    if env_w["value"] in (0, None)
                    else summary["weak"]["mean"] / env_w["value"]
                ),
                "train_seconds": time.perf_counter() - t_train,
                "n_params": per_seed[0]["n_params"],
            }
            print(
                f"[{pde} {arch}] rel {summary['rel_l2']['mean']:.4f} "
                f"strong {summary['strong']['mean']:.4f} weak {summary['weak']['mean']:.6f}",
                flush=True,
            )
        # Baseline scaled anchors are reported once.
        score_baselines = {}
        for name in ("zeros", "training_mean", "frozen_initial", "reference"):
            score_baselines[name] = {}
            for metric, exclude, ref_value in (
                ("rel_l2", (), 0.0),
                ("strong", ("zeros",), baseline_summary["reference"]["strong"]),
                ("weak", ("zeros",), baseline_summary["reference"]["weak"]),
            ):
                trivials = {
                    bname: baseline_summary[bname][metric]
                    for bname in ("zeros", "training_mean", "frozen_initial")
                }
                if metric != "rel_l2" and name == "zeros":
                    score_baselines[name][metric] = {"score": None, "reason": "excluded from residual anchor"}
                    continue
                if name == "reference":
                    anchor = best_baseline(trivials, exclude=exclude)
                    defined = (
                        scaled_score(baseline_summary[name][metric], anchor[1], ref_value)
                        if anchor is not None
                        else None
                    )
                    score_baselines[name][metric] = {
                        "score": defined,
                        "reason": None if defined is not None else "undefined",
                    }
                    continue
                anchor = best_baseline(trivials, exclude=exclude)
                if anchor is None:
                    score_baselines[name][metric] = {"score": None, "reason": "no anchor"}
                else:
                    score_baselines[name][metric] = {
                        "score": scaled_score(baseline_summary[name][metric], anchor[1], ref_value),
                        "reason": None,
                        "baseline_name": anchor[0],
                    }
        results["hpo"][pde] = hpo_pde
        results["models"][pde] = model_pde
        results["scores"][pde] = score_baselines
        results["associations"][pde] = {
            "strong_vs_keff_sq_rel": linear_fit(np.array(assoc_x), np.array(assoc_y)),
            "kendall_strong_weak": kendall(np.array(rank_s), np.array(rank_w)),
        }
        if pde == "heat":
            results["associations"][pde]["strong_vs_alpha_keff_sq_rel"] = linear_fit(
                np.array(assoc_x_heat), np.array(assoc_y)
            )
        results["wall"][pde] = {
            arch: {
                "hpo_seconds": hpo_pde[arch]["seconds"],
                "train_seconds": model_pde[arch]["train_seconds"],
            }
            for arch in architectures
        }
    results["_traj"] = retained
    results["paired_models"] = _paired_from_rerun(protocol, spec, burgers, heat, results)
    return results


def _nanmean(values) -> float | None:
    arr = np.array([np.nan if v is None else v for v in values], dtype=np.float64)
    if not np.isfinite(arr).any():
        return None
    return float(np.nanmean(arr))


def _paired_from_rerun(protocol, spec, burgers, heat, results) -> list[dict]:
    """Paired permutation tests between architectures on seed-averaged test metrics.

    Retrains nothing: the per-seed trajectory vectors were not all retained, so this
    uses one frozen evaluation seed's prediction only if present. The model study
    stores per-seed means; paired tests are computed from a dedicated re-score that
    the caller fills in ``results['_traj']`` when available.
    """
    stored = results.get("_traj")
    if not stored:
        return []
    stats = protocol["statistics"]
    rows = []
    for pde, archs in stored.items():
        names = list(archs)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                for metric in ("rel_l2", "strong", "weak"):
                    diff = np.asarray(archs[a][metric]) - np.asarray(archs[b][metric])
                    test = paired_permutation(diff, n_perm=int(stats["n_perm"]), seed=int(stats["seed"]))
                    rows.append({"pde": pde, "a": a, "b": b, "metric": metric, **test})
    return rows


def _retain_traj(protocol, spec, burgers, heat, chosen: dict) -> dict:
    """One forward pass per saved config is not possible without weights.

    Paired tests use the seed-averaged per-trajectory metrics captured during training.
    This helper is unused; retention happens inside run_models via results side channel.
    """
    return {}


def run_shipped(protocol, spec) -> dict:
    """Rescore the shipped FNO and PINO checkpoints. Does not retrain them."""
    from baselines.fno.io import load_checkpoint
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    cfg = protocol["shipped"]
    if float(cfg["legacy_threshold"]) != float(RESIDUAL_UNTRUSTED):
        raise AssertionError("protocol legacy threshold drifted from metrics.trust.RESIDUAL_UNTRUSTED")
    burgers = load_burgers(
        {
            **protocol["datasets"]["burgers"],
            "n_traj": cfg["burgers"]["n_traj"],
            "nx": cfg["burgers"]["nx"],
            "nt": cfg["burgers"]["nt"],
            "seed": cfg["burgers"]["seed"],
            "n_train": 1,
            "n_val": 1,
            "n_test": 1,
        }
    )
    heat = load_heat(
        {
            **protocol["datasets"]["heat"],
            "n_traj": cfg["heat"]["n_traj"],
            "n": cfg["heat"]["n"],
            "nt": cfg["heat"]["nt"],
            "seed": cfg["heat"]["seed"],
            "n_train": 1,
            "n_val": 1,
            "n_test": 1,
        }
    )
    levels = [float(v) for v in protocol["noise"]["levels"]]
    seeds = [int(s) for s in cfg["noise_seeds"]]

    def envelopes(pde, data, coef_key, idx):
        coords = {"x": data["x"], "t": data["t"]}
        if pde == "heat":
            coords["y"] = data["y"]
        acc = {(kind, level): [] for kind in ("white", "smooth") for level in levels}
        for local in idx:
            ref = data["u"][int(local)]
            coef = float(data[coef_key][int(local)])
            for seed in seeds:
                for level in levels:
                    for kind in ("white", "smooth"):
                        rng = np.random.default_rng(
                            np.random.SeedSequence([seed, int(local), int(round(level * 1_000_000)), 1 if kind == "white" else 2])
                        )
                        error = _inject(ref, kind, level, rng, pde=pde, coords=coords, noise_cfg=protocol["noise"])
                        field = ref + error
                        if pde == "burgers":
                            scored = _score_burgers(field, ref, data["x"], data["t"], coef, spec)
                        else:
                            scored = _score_heat(field, ref, data["x"], data["y"], data["t"], coef, spec)
                        acc[(kind, level)].append(scored)
        curve = []
        for level in levels:
            for kind in ("white", "smooth"):
                chunk = acc[(kind, level)]
                curve.append(
                    {
                        "level": level,
                        "kind": kind,
                        "strong": float(np.mean([c["strong"] for c in chunk])),
                        "weak": float(np.mean([c["weak"] for c in chunk])),
                        "rel_l2": float(np.mean([c["rel_l2"] for c in chunk])),
                    }
                )
        return curve

    def verdict(summary, curve):
        levels_w = np.array([r["level"] for r in curve if r["kind"] == "white"])
        weak_w = np.array([r["weak"] for r in curve if r["kind"] == "white"])
        strong_w = np.array([r["strong"] for r in curve if r["kind"] == "white"])
        legacy = bool(summary["strong"] > float(cfg["legacy_threshold"]))
        below = float(summary["rel_l2"]) < float(levels_w.min())
        env_w = log_interp(levels_w, weak_w, max(summary["rel_l2"], float(levels_w.min())))
        env_s = log_interp(levels_w, strong_w, max(summary["rel_l2"], float(levels_w.min())))
        if below:
            survives = None
            note = "relative L2 is below the calibrated noise grid; weak-form survival is not called"
        else:
            survives = bool(summary["weak"] > env_w["value"])
            note = None
        return {
            "legacy_untrusted": legacy,
            "legacy_threshold": float(cfg["legacy_threshold"]),
            "weak_above_matched_white_envelope": survives,
            "verdict_survives_weak_form": survives,
            "white_envelope_weak": env_w,
            "white_envelope_strong": env_s,
            "note": note,
            "rule": protocol["verdict"]["rule"],
        }

    rows = []
    curves = {}
    for pde, data, ckpt_name, coef_key in (
        ("burgers", burgers, "fno_burgers.pt", "nu"),
        ("burgers", burgers, "pino_burgers.pt", "nu"),
        ("heat", heat, "fno_heat2d.pt", "alpha"),
        ("heat", heat, "pino_heat2d.pt", "alpha"),
    ):
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
        key = (pde, tuple(idx.tolist()))
        if key not in curves:
            curves[key] = envelopes(pde, data, coef_key, idx)
        coef = data[coef_key][idx]
        ref = data["u"][idx]
        pred = predict(loaded["model"], ref[:, 0] if pde == "burgers" else ref[:, 0], coef)
        if pde == "burgers":
            scored = _score_many_burgers(pred, ref, data, coef, spec)
            frozen = np.repeat(ref[:, 0:1, :], ref.shape[1], axis=1)
            frozen_scored = _score_many_burgers(frozen, ref, data, coef, spec)
            zero_scored = _score_many_burgers(np.zeros_like(ref), ref, data, coef, spec)
        else:
            scored = _score_many_heat(pred, ref, data, coef, spec)
            frozen = np.repeat(ref[:, 0:1, :, :], ref.shape[1], axis=1)
            frozen_scored = _score_many_heat(frozen, ref, data, coef, spec)
            zero_scored = _score_many_heat(np.zeros_like(ref), ref, data, coef, spec)
        summary = {
            "rel_l2": float(np.mean(_pool(scored, "rel_l2"))),
            "strong": float(np.mean(_pool(scored, "strong"))),
            "weak": float(np.mean(_pool(scored, "weak"))),
            "n_eval": int(len(idx)),
            "eval_idx": idx.tolist(),
        }
        row = {
            "pde": pde,
            "model": model_name,
            "checkpoint": ckpt_name,
            "status": "ok",
            **summary,
            "decision": verdict(summary, curves[key]),
        }
        rows.append(row)
        # Baselines once per PDE, attached to the fno row's PDE via separate entries.
        if model_name == "fno":
            for label, chunk in (
                ("frozen_initial", frozen_scored),
                ("zeros", zero_scored),
                ("reference", _score_many_burgers(ref, ref, data, coef, spec) if pde == "burgers" else _score_many_heat(ref, ref, data, coef, spec)),
            ):
                base_summary = {
                    "rel_l2": float(np.mean(_pool(chunk, "rel_l2"))),
                    "strong": float(np.mean(_pool(chunk, "strong"))),
                    "weak": float(np.mean(_pool(chunk, "weak"))),
                    "n_eval": int(len(idx)),
                }
                rows.append(
                    {
                        "pde": pde,
                        "model": label,
                        "checkpoint": None,
                        "status": "ok",
                        **base_summary,
                        "decision": verdict(base_summary, curves[key]),
                    }
                )
    # Drop the bulky curve cache from the return; keep a compact copy on the first PDE.
    return {"rows": rows}


def run_spectrum_example(protocol, burgers) -> dict:
    fig = protocol["figures"]
    ref = burgers["u"][int(fig["traj"])]
    coef = float(burgers["nu"][int(fig["traj"])])
    rng_w = np.random.default_rng(np.random.SeedSequence([int(fig["seed"]), 1]))
    rng_s = np.random.default_rng(np.random.SeedSequence([int(fig["seed"]), 2]))
    coords = {"x": burgers["x"], "t": burgers["t"]}
    white = _inject(ref, "white", float(fig["epsilon"]), rng_w, pde="burgers", coords=coords, noise_cfg=protocol["noise"])
    smooth = _inject(ref, "smooth", float(fig["epsilon"]), rng_s, pde="burgers", coords=coords, noise_cfg=protocol["noise"])
    h = float(burgers["x"][1] - burgers["x"][0])
    n = white.shape[-1]
    m = np.fft.rfftfreq(n) * n

    def power(field):
        spec = np.fft.rfft(field, axis=-1)
        return (np.abs(spec) ** 2).mean(axis=0)

    lam = (4.0 / h**2) * np.sin(np.pi * m / n) ** 2
    return {
        "mode": m.tolist(),
        "white_power": power(white).tolist(),
        "smooth_power": power(smooth).tolist(),
        "white_weighted": (lam**2 * power(white)).tolist(),
        "smooth_weighted": (lam**2 * power(smooth)).tolist(),
        "epsilon": float(fig["epsilon"]),
        "traj": int(fig["traj"]),
        "seed": int(fig["seed"]),
    }


def evaluate_hypotheses(noise, resolution, symbol, mismatch) -> list[dict]:
    primary = 0.01
    rows = []
    for pde, block in noise["pdes"].items():
        test = block["paired_white_minus_smooth"]["strong"]
        supported = test["mean_diff"] is not None and test["mean_diff"] > 0 and test["p_value"] < 0.05
        rows.append(
            {
                "id": f"H1_{pde}_strong_white_gt_smooth",
                "supported": bool(supported),
                "statistic": test["mean_diff"],
                "mean_diff": test["mean_diff"],
                "p_value": test["p_value"],
            }
        )
    rows.append(
        {
            "id": "H2_heat_second_moment",
            "supported": bool(0.9 <= symbol["heat_ratio_mean"] <= 1.1),
            "statistic": symbol["heat_ratio_mean"],
            "ratio": symbol["heat_ratio_mean"],
        }
    )
    rows.append(
        {
            "id": "H3_burgers_second_moment",
            "supported": bool(0.9 <= symbol["burgers"]["ratio"] <= 1.1),
            "statistic": symbol["burgers"]["ratio"],
            "ratio": symbol["burgers"]["ratio"],
        }
    )
    for pde in ("burgers", "heat"):
        spatial = [r for r in resolution["rows"] if r["pde"] == pde and r["sweep"] == "spatial"]
        spatial = sorted(spatial, key=lambda r: r["n"])
        if len(spatial) >= 2:
            coarse, fine = spatial[0], spatial[-1]
            strong_gain = fine["strong_white_mean"] / coarse["strong_white_mean"]
            weak_gain = fine["weak_white_mean"] / max(coarse["weak_white_mean"], 1e-18)
            rows.append(
                {
                    "id": f"H4_{pde}_strong_grows_with_resolution",
                    "supported": bool(fine["strong_white_mean"] > coarse["strong_white_mean"]),
                    "statistic": fine["strong_white_mean"] / coarse["strong_white_mean"],
                    "coarse": coarse["strong_white_mean"],
                    "fine": fine["strong_white_mean"],
                }
            )
            rows.append(
                {
                    "id": f"H5_{pde}_weak_grows_less_than_strong",
                    "supported": bool(weak_gain < strong_gain),
                    "statistic": weak_gain / strong_gain,
                    "strong_gain": strong_gain,
                    "weak_gain": weak_gain,
                }
            )
    for pde, rows_m in mismatch.items():
        at1 = next(r for r in rows_m if abs(r["scale"] - 1.0) < 1e-12)
        at0 = next(r for r in rows_m if abs(r["scale"] - 0.0) < 1e-12)
        rows.append(
            {
                "id": f"H6_{pde}_weak_detects_zero_coefficient",
                "supported": bool(at0["weak_mean"] > at1["weak_mean"]),
                "statistic": at0["weak_mean"] / max(at1["weak_mean"], 1e-18),
                "scale0": at0["weak_mean"],
                "scale1": at1["weak_mean"],
            }
        )
    return rows


def run_all(protocol: dict) -> dict:
    if float(protocol["shipped"]["legacy_threshold"]) != float(RESIDUAL_UNTRUSTED):
        raise AssertionError("legacy threshold must match metrics.trust.RESIDUAL_UNTRUSTED")
    spec = spec_from_protocol(protocol["weak_form"])
    started = time.perf_counter()
    print("Generating paper datasets...", flush=True)
    burgers = load_burgers(protocol["datasets"]["burgers"])
    heat = load_heat(protocol["datasets"]["heat"])
    timings = {}
    t0 = time.perf_counter()
    print("Noise injection...", flush=True)
    noise = run_noise(protocol, spec, burgers, heat)
    timings["noise"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    print("Resolution sweep...", flush=True)
    resolution = run_resolution(protocol, spec)
    timings["resolution"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    print("Symbol checks...", flush=True)
    symbol = run_symbol(protocol, burgers, heat)
    timings["symbol"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    print("Coefficient mismatch...", flush=True)
    mismatch = run_mismatch(protocol, spec, burgers, heat)
    timings["mismatch"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    print("Test-function bandwidth...", flush=True)
    kphi = run_kphi(protocol, spec, burgers, heat)
    timings["kphi"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    print("Model study...", flush=True)
    models = run_models(protocol, spec, burgers, heat, noise)
    timings["models"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    print("Shipped checkpoints...", flush=True)
    shipped = run_shipped(protocol, spec)
    timings["shipped"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    print("Inverse probe...", flush=True)
    inverse = run_inverse(protocol, spec)
    timings["inverse"] = time.perf_counter() - t0
    spectrum = run_spectrum_example(protocol, burgers)
    hypotheses = evaluate_hypotheses(noise, resolution, symbol, mismatch)
    # Drop the raw noise rows from the returned payload; paired tests are already reduced.
    noise_public = {"levels": noise["levels"], "seeds": noise["seeds"], "pdes": noise["pdes"]}
    timings["total"] = time.perf_counter() - started
    print(f"Done in {timings['total']:.1f}s", flush=True)
    return {
        "noise": noise_public,
        "resolution": resolution,
        "symbol": symbol,
        "mismatch": mismatch,
        "kphi": kphi,
        "models": models,
        "shipped": shipped,
        "inverse": inverse,
        "spectrum": spectrum,
        "hypotheses": hypotheses,
        "timings_seconds": timings,
        "weak_spec": spec.as_dict(),
    }
