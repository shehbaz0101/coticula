"""Write paper_artifacts and docs/research/weak_form_study.md from a run."""
from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np


def to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _fmt(value, digits=6) -> str:
    if value is None:
        return "—"
    if isinstance(value, dict):
        return _pm(value.get("mean"), value.get("std"))
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    return f"{float(value):.{digits}g}"


def _pm(mean, std) -> str:
    if mean is None:
        return "—"
    if std is None:
        return f"{mean:.6g}"
    return f"{mean:.6g} ± {std:.3g}"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


def _flatten_noise(noise: dict) -> list[dict]:
    rows = []
    for pde, block in noise["pdes"].items():
        for entry in block["table"]:
            rows.append(
                {
                    "pde": pde,
                    "kind": entry["kind"],
                    "level": entry["level"],
                    "rel_l2_mean": entry["rel_l2"]["mean"],
                    "rel_l2_std": entry["rel_l2"]["std"],
                    "strong_mean": entry["strong"]["mean"],
                    "strong_std": entry["strong"]["std"],
                    "weak_mean": entry["weak"]["mean"],
                    "weak_std": entry["weak"]["std"],
                    "keff_h_mean": entry["keff_h"]["mean"],
                    "keff_h_std": entry["keff_h"]["std"],
                    "strong_over_weak": entry["strong"]["mean"] / entry["weak"]["mean"]
                    if entry["weak"]["mean"]
                    else None,
                }
            )
        rows.append(
            {
                "pde": pde,
                "kind": "reference",
                "level": 0.0,
                "rel_l2_mean": 0.0,
                "rel_l2_std": 0.0,
                "strong_mean": block["clean"]["strong_mean"],
                "strong_std": 0.0,
                "weak_mean": block["clean"]["weak_mean"],
                "weak_std": 0.0,
                "keff_h_mean": None,
                "keff_h_std": None,
                "strong_over_weak": None,
            }
        )
    return rows


def _model_rows(models: dict) -> list[dict]:
    rows = []
    for pde, baselines in models["baselines"].items():
        for name, stats in baselines.items():
            scores = models["scores"][pde][name]
            rows.append(
                {
                    "pde": pde,
                    "model": name,
                    "kind": "baseline",
                    "rel_l2_mean": stats["rel_l2"],
                    "rel_l2_std": 0.0,
                    "S_rel_l2": (scores.get("rel_l2") or {}).get("score"),
                    "strong_mean": stats["strong"],
                    "strong_std": 0.0,
                    "S_strong": (scores.get("strong") or {}).get("score"),
                    "weak_mean": stats["weak"],
                    "weak_std": 0.0,
                    "S_weak": (scores.get("weak") or {}).get("score"),
                    "s_coef_mean": stats.get("s_coef"),
                    "s_coef_std": 0.0,
                    "cosine_mean": stats.get("cosine"),
                    "cf_rel_l2_mean": stats.get("cf_rel_l2"),
                    "n_params": 0,
                }
            )
        for arch, block in models["models"][pde].items():
            summary = block["summary"]
            scaled = block["scaled"]
            rows.append(
                {
                    "pde": pde,
                    "model": arch,
                    "kind": "trained",
                    "rel_l2_mean": summary["rel_l2"]["mean"],
                    "rel_l2_std": summary["rel_l2"]["std"],
                    "S_rel_l2": scaled["rel_l2"]["mean"],
                    "strong_mean": summary["strong"]["mean"],
                    "strong_std": summary["strong"]["std"],
                    "S_strong": scaled["strong"]["mean"],
                    "weak_mean": summary["weak"]["mean"],
                    "weak_std": summary["weak"]["std"],
                    "S_weak": scaled["weak"]["mean"],
                    "s_coef_mean": summary["s_coef"]["mean"],
                    "s_coef_std": summary["s_coef"]["std"],
                    "cosine_mean": summary["cosine"]["mean"],
                    "cf_rel_l2_mean": summary["cf_rel_l2"]["mean"],
                    "weak_over_white_envelope": block["weak_over_white_envelope"],
                    "n_params": block["n_params"],
                }
            )
    return rows


def _figures(result: dict, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {"white": "#0072B2", "smooth": "#E69F00"}

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4), sharex=True)
    for row, pde in enumerate(("burgers", "heat")):
        table = result["noise"]["pdes"][pde]["table"]
        for col, metric in enumerate(("strong", "weak")):
            ax = axes[row, col]
            for kind in ("white", "smooth"):
                xs, ys, yerr = [], [], []
                for entry in table:
                    if entry["kind"] != kind:
                        continue
                    xs.append(entry["level"])
                    ys.append(entry[metric]["mean"])
                    yerr.append(entry[metric]["std"] or 0.0)
                ax.errorbar(xs, ys, yerr=yerr, marker="o", color=colors[kind], label=kind, capsize=2)
            clean = result["noise"]["pdes"][pde]["clean"]
            floor = clean["strong_mean"] if metric == "strong" else clean["weak_mean"]
            ax.axhline(floor, color="#009E73", ls="--", lw=1, label="solver floor")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{pde} {metric}-form")
            ax.set_xlabel("matched rel-L2")
            ax.set_ylabel("residual")
            ax.grid(True, which="both", ls=":", alpha=0.5)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_noise_strong_vs_weak.png", dpi=150)
    fig.savefig(fig_dir / "fig_noise_strong_vs_weak.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), sharey=False)
    for ax, pde in zip(axes, ("burgers", "heat")):
        rows = [r for r in result["resolution"]["rows"] if r["pde"] == pde and r["sweep"] == "spatial"]
        rows = sorted(rows, key=lambda r: r["n"])
        ns = [r["n"] for r in rows]
        ax.plot(ns, [r["strong_white_mean"] for r in rows], "o-", color=colors["white"], label="strong white")
        ax.plot(ns, [r["strong_smooth_mean"] for r in rows], "s-", color=colors["smooth"], label="strong smooth")
        ax.plot(ns, [r["weak_white_mean"] for r in rows], "o--", color="#56B4E9", label="weak white")
        ax.plot(ns, [r["weak_smooth_mean"] for r in rows], "s--", color="#F0E442", label="weak smooth")
        ax.plot(ns, [r["theory_white_mean"] for r in rows], "x:", color="#009E73", label="theory white")
        ax.set_yscale("log")
        ax.set_xlabel("grid points")
        ax.set_title(pde)
        ax.grid(True, ls=":", alpha=0.5)
    axes[0].set_ylabel(f"residual at rel-L2 = {result['resolution']['epsilon']}")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_resolution_sweep.png", dpi=150)
    fig.savefig(fig_dir / "fig_resolution_sweep.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    cases = result["symbol"]["heat_cases"]
    theory = [c["theory"] for c in cases]
    emp = [c["empirical_mean"] for c in cases]
    axes[0].scatter(theory, emp, color="#0072B2")
    lo = min(theory + emp)
    hi = max(theory + emp)
    axes[0].plot([lo, hi], [lo, hi], color="#009E73", ls="--")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("formula E|R|^2 / s^2")
    axes[0].set_ylabel("measured mean R^2")
    axes[0].set_title("heat white second moment")
    heat_rows = [r for r in result["resolution"]["rows"] if r["pde"] == "heat"]
    axes[1].scatter(
        [r["theory_white_mean"] for r in heat_rows],
        [r["strong_white_mean"] for r in heat_rows],
        color="#0072B2",
        label="heat white",
    )
    xs = [r["theory_white_mean"] for r in heat_rows]
    ys = [r["strong_white_mean"] for r in heat_rows]
    if xs and ys:
        lo = min(xs + ys)
        hi = max(xs + ys)
        axes[1].plot([lo, hi], [lo, hi], color="#009E73", ls="--", label="identity")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$C_h \varepsilon$")
    axes[1].set_ylabel("measured strong residual")
    axes[1].set_title("resolution sweep vs formula")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_symbol_fit.png", dpi=150)
    fig.savefig(fig_dir / "fig_symbol_fit.pdf")
    plt.close(fig)

    spec = result["spectrum"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    modes = spec["mode"]
    axes[0].semilogy(modes, spec["white_power"], color=colors["white"], label="white")
    axes[0].semilogy(modes, spec["smooth_power"], color=colors["smooth"], label="smooth")
    axes[0].set_xlabel("Fourier mode")
    axes[0].set_ylabel("power")
    axes[0].set_title("error spectrum")
    axes[0].legend(frameon=False)
    axes[1].semilogy(modes, spec["white_weighted"], color=colors["white"], label="white")
    axes[1].semilogy(modes, spec["smooth_weighted"], color=colors["smooth"], label="smooth")
    axes[1].set_xlabel("Fourier mode")
    axes[1].set_ylabel(r"$\lambda_h(k)^2$ power")
    axes[1].set_title("symbol-weighted spectrum")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_error_spectra.png", dpi=150)
    fig.savefig(fig_dir / "fig_error_spectra.pdf")
    plt.close(fig)


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([line, sep, *body])


def _markdown(result: dict, protocol: dict, digest: str) -> str:
    noise_rows = []
    for pde, block in result["noise"]["pdes"].items():
        for entry in block["table"]:
            if entry["kind"] not in ("white", "smooth"):
                continue
            noise_rows.append(
                [
                    pde,
                    entry["kind"],
                    _fmt(entry["level"]),
                    _fmt(entry["rel_l2"]),
                    _fmt(entry["keff_h"]),
                    _fmt(entry["strong"]),
                    _fmt(entry["weak"]),
                    _fmt(
                        entry["strong"]["mean"] / entry["weak"]["mean"]
                        if entry["weak"]["mean"]
                        else None
                    ),
                ]
            )
        clean = block["clean"]
        noise_rows.append(
            [pde, "reference", "0", "0", "—", _fmt(clean["strong_mean"]), _fmt(clean["weak_mean"]), "—"]
        )
    res_rows = []
    for row in result["resolution"]["rows"]:
        res_rows.append(
            [
                row["pde"],
                row["sweep"],
                str(row["n"]),
                f"{row['dt']:.4g}",
                f"{row['mesh_ratio_r']:.3g}",
                _fmt(row["strong_white_mean"]),
                _fmt(row["theory_white_mean"]),
                _fmt(row["strong_smooth_mean"]),
                _fmt(row["weak_white_mean"]),
                _fmt(row["weak_smooth_mean"]),
            ]
        )
    model_rows = []
    for row in _model_rows(result["models"]):
        model_rows.append(
            [
                row["pde"],
                row["model"],
                _pm(row["rel_l2_mean"], row["rel_l2_std"]),
                _fmt(row["S_rel_l2"]),
                _pm(row["strong_mean"], row["strong_std"]),
                _fmt(row["S_strong"]),
                _pm(row["weak_mean"], row["weak_std"]),
                _fmt(row["S_weak"]),
            ]
        )
    coef_rows = []
    for pde in ("burgers", "heat"):
        for arch, block in result["models"]["models"][pde].items():
            summary = block["summary"]
            coef_rows.append(
                [
                    pde,
                    arch,
                    _pm(summary["s_coef"]["mean"], summary["s_coef"]["std"]),
                    _pm(summary["cosine"]["mean"], summary["cosine"]["std"]),
                    _pm(summary["cf_rel_l2"]["mean"], summary["cf_rel_l2"]["std"]),
                ]
            )
    mismatch_rows = []
    for pde, rows in result["mismatch"].items():
        for row in rows:
            mismatch_rows.append(
                [
                    pde,
                    _fmt(row["scale"]),
                    _pm(row["strong_mean"], row["strong_std"]),
                    _pm(row["weak_mean"], row["weak_std"]),
                ]
            )
    shipped_rows = []
    for row in result["shipped"]["rows"]:
        decision = row.get("decision") or {}
        shipped_rows.append(
            [
                row["pde"],
                row["model"],
                row["status"],
                _fmt(row.get("rel_l2")),
                _fmt(row.get("strong")),
                _fmt(row.get("weak")),
                _fmt(decision.get("legacy_untrusted")),
                _fmt(decision.get("verdict_survives_weak_form")),
            ]
        )
    symbol = result["symbol"]
    hyp_rows = []
    for row in result["hypotheses"]:
        hyp_rows.append([row["id"], "yes" if row["supported"] else "no", _fmt(row.get("statistic"))])
    inv_rows = []
    for split in ("id", "hard_ood"):
        block = result["inverse"][split]
        for method in ("strong_form_ls", "weak_form_ls", "solver_in_the_loop"):
            stats = block[method]
            inv_rows.append(
                [
                    split,
                    method,
                    _pm(stats["rel_err_mean"], stats["rel_err_std"]),
                    str(stats["n_fail"]),
                ]
            )
    k_rows = []
    for row in result["kphi"]["rows"]:
        k_rows.append(
            [
                row["pde"],
                str(row["k_phi"]),
                _fmt(row["weak_white"]),
                _fmt(row["weak_smooth"]),
                _fmt(row["weak_frozen"]),
                _fmt(row["weak_clean"]),
            ]
        )
    data = protocol["datasets"]
    b, h = data["burgers"], data["heat"]
    timings = result["timings_seconds"]
    fno_ship = [
        r
        for r in result["shipped"]["rows"]
        if r["model"] == "fno" and r["status"] == "ok"
    ]
    survive_txt = []
    for row in fno_ship:
        decision = row["decision"]["verdict_survives_weak_form"]
        legacy = row["decision"]["legacy_untrusted"]
        survive_txt.append(
            f"{row['pde']} FNO legacy untrusted={str(bool(legacy)).lower()}, "
            f"weak-form survival={str(bool(decision)).lower()} "
            f"(strong={row['strong']:.6g}, weak={row['weak']:.6g}, rel-L2={row['rel_l2']:.6g})"
        )
    lines = [
        "# Weak-form noise-calibrated residual study",
        "",
        "Experimental backbone for **Small Errors, Large Residuals**. Every number below is written by `scripts/reproduce_paper.py` from the frozen protocol. The default Conserve metric, the trust rule, and the README are unchanged.",
        "",
        f"Protocol SHA-256: `{digest}`",
        "",
        "## Definitions",
        "",
        "Strong residual ρ_s is the per-step finite-difference residual in `metrics/conserve.py`: mean over time of ||R||_2 / ||u_mid||_2. Heat uses zero Dirichlet boundaries and drops the boundary nodes.",
        "",
        "Weak residual ρ_w = sqrt(sum_m r_m^2) / (||u_ref||_2 sqrt(tr G)), with G the Gram matrix of ψ_m = L* φ_m. Test functions are compact polynomial bumps in time times Fourier modes (Burgers) or Dirichlet sine modes (heat). Bandwidth K_φ and the bump geometry are in `paper_artifacts/protocol.json`. The denominator uses the reference field, as in the draft.",
        "",
        "Smooth errors are time-varying trigonometric polynomials of degree "
        f"{protocol['noise']['n_time_modes']} on spatial modes 1..{protocol['noise']['k_smooth']}. "
        "They are not frozen in time. White heat noise is i.i.d. on the interior and exactly zero on the Dirichlet boundary, then rescaled.",
        "",
        "Scaled score S(M) = 100 (M_0 - M) / (M_0 - M_ref), not clipped. M_ref is 0 for relative L2 and the solver residual for residual metrics. M_0 is the best (lowest) trivial baseline. The zero field is excluded from M_0 for residual metrics. S is undefined when M_0 ≤ M_ref.",
        "",
        "The legacy untrusted flag is strong residual > 1, the same constant as `metrics.trust.RESIDUAL_UNTRUSTED`. It survives the weak form only when ρ_w is strictly above the white-noise weak envelope at the model's own relative L2. Fields whose relative L2 lies below the noise grid do not receive that call.",
        "",
        "## Datasets and budget",
        "",
        _md_table(
            ["PDE", "grid", "h", "dt", "Nt", "train/val/test", "coefficient"],
            [
                [
                    "Burgers",
                    f"nx={b['nx']}",
                    f"{b['L']/b['nx']:.4g}",
                    f"{b['T']/b['nt']:.4g}",
                    str(b["nt"]),
                    f"{b['n_train']}/{b['n_val']}/{b['n_test']}",
                    f"ν in {b['nu_range']}",
                ],
                [
                    "Heat",
                    f"n={h['n']} Dirichlet",
                    f"{h['L']/(h['n']-1):.4g}",
                    f"{h['T']/h['nt']:.4g}",
                    str(h["nt"]),
                    f"{h['n_train']}/{h['n_val']}/{h['n_test']}",
                    f"α in {h['alpha_range']}",
                ],
            ],
        ),
        "",
        f"Equal tuning budget: {protocol['training']['trials_per_architecture']} trials per architecture, "
        f"{protocol['training']['epochs_hpo']} epochs on the validation split (seed {protocol['training']['hpo_seed']}), "
        f"then {protocol['training']['epochs_final']} epochs from scratch for seeds {protocol['training']['final_seeds']}. "
        f"Optimizer {protocol['training']['optimizer']}, lr={protocol['training']['lr']}, "
        f"weight decay={protocol['training']['weight_decay']}, batch size {protocol['training']['batch_size']}, data MSE only. "
        "No trial sees the test split. DeepONet trunks use fixed Fourier features and a fixed coefficient scale (0.05 for ν, 0.2 for α).",
        "",
        "## Noise injection",
        "",
        "Mean ± sample standard deviation across noise seeds of the per-seed trajectory mean.",
        "",
        _md_table(
            ["PDE", "error", "ε", "realized rel-L2", "keff·h", "ρ_s", "ρ_w", "ρ_s/ρ_w"],
            noise_rows,
        ),
        "",
        "## Resolution and time-step sweep",
        "",
        f"Fixed ε = {result['resolution']['epsilon']}. Theory for heat white noise is C_h ε from the second-moment formula. "
        "Theory for Burgers is the same per-step normalization applied to the pointwise Burgers second-moment formula, and it ignores the discretization floor.",
        "",
        _md_table(
            ["PDE", "sweep", "N", "dt", "r", "ρ_s white", "theory", "ρ_s smooth", "ρ_w white", "ρ_w smooth"],
            res_rows,
        ),
        "",
        "## Symbol check",
        "",
        f"Heat synthetic i.i.d. check, mean measured/formula = {symbol['heat_ratio_mean']:.6g}. "
        f"Log-log fit of measured vs formula: slope {symbol['heat_log_fit']['slope']:.6g}, "
        f"R² {symbol['heat_log_fit']['r_squared']:.6g}.",
        "",
        f"Burgers synthetic check ratio = {symbol['burgers']['ratio']:.6g} "
        f"(empirical {symbol['burgers']['empirical_mean']:.6g}, formula {symbol['burgers']['theory_mean']:.6g}).",
        "",
        f"Static heat error, ||R[e]||/||e|| over α keff(e)² = {symbol['heat_static_identity']['ratio']:.6g}.",
        "",
        "## Models",
        "",
        _md_table(
            ["PDE", "model", "rel-L2", "S(rel-L2)", "ρ_s", "S(ρ_s)", "ρ_w", "S(ρ_w)"],
            model_rows,
        ),
        "",
        "S is undefined (—) when the best trivial baseline does no worse than the reference. Zeros are excluded from the residual anchor and their residual score is blank.",
        "",
        "### Coefficient response",
        "",
        f"γ = {protocol['coefficient_response']['gamma']}. S_coef = ||G(γc) - G(c)|| / ||G*(γc) - G*(c)||. Cosine is undefined when either difference vanishes.",
        "",
        _md_table(["PDE", "model", "S_coef", "cosine", "rel-L2 at γc"], coef_rows),
        "",
        "### Associations",
        "",
    ]
    for pde, assoc in result["models"]["associations"].items():
        fit = assoc["strong_vs_keff_sq_rel"]
        tau = assoc["kendall_strong_weak"]
        lines.append(
            f"{pde}: ρ_s vs keff²·rel-L2 slope {fit['slope']:.6g}, intercept {fit['intercept']:.6g}, "
            f"R² {fit['r_squared']:.6g} (n={fit['n']}). Kendall τ(ρ_s, ρ_w) = {tau['tau']:.6g} (p={tau['p_value']:.3g}, n={tau['n']})."
        )
        if "strong_vs_alpha_keff_sq_rel" in assoc:
            fit2 = assoc["strong_vs_alpha_keff_sq_rel"]
            lines.append(
                f"Heat ρ_s vs α keff²·rel-L2 slope {fit2['slope']:.6g}, R² {fit2['r_squared']:.6g}."
            )
    lines += [
        "",
        "## Coefficient mismatch on exact labels",
        "",
        _md_table(["PDE", "scale", "ρ_s", "ρ_w"], mismatch_rows),
        "",
        "## Shipped checkpoints",
        "",
        "These are the v0 checkpoints on the original grids, not the five-seed study.",
        "",
        _md_table(
            ["PDE", "model", "status", "rel-L2", "ρ_s", "ρ_w", "legacy untrusted", "weak survival"],
            shipped_rows,
        ),
        "",
        " ".join(survive_txt),
        "",
        "## Bandwidth ablation",
        "",
        _md_table(["PDE", "K_φ", "ρ_w white", "ρ_w smooth", "ρ_w frozen", "ρ_w clean"], k_rows),
        "",
        "## Inverse probe",
        "",
        "Burgers viscosity from evenly spaced sensors. The FNO-conditioned inverse was not run.",
        "",
        _md_table(["split", "method", "rel err in ν", "failures"], inv_rows),
        "",
        "## Pre-registered checks",
        "",
        _md_table(["check", "supported", "statistic"], hyp_rows),
        "",
        "## Wall clock (seconds)",
        "",
        _md_table(
            ["stage", "seconds"],
            [[name, f"{seconds:.3f}"] for name, seconds in timings.items()],
        ),
        "",
        "## Limitations",
        "",
        "- The exact spectral identity is for linear constant-coefficient operators. Burgers is checked through the white-noise second-moment formula, which matched on this run to the ratio reported above.",
        "- ρ_w uses a finite test space. High-frequency errors are nearly invisible to it. The zero field has a zero residual and is excluded from the residual score anchor for that reason.",
        "- Heat white noise in the injection study is zero on the boundary, so the nodewise formula is checked on a fully i.i.d. array, separately from the Dirichlet metric.",
        "- The model study uses small grids, a two-trial search, and CPU training. DeepONet's trunk uses a fixed Fourier feature map. Results can disagree with a larger search.",
        "- The weak-form survival rule compares a model with the white envelope at its own relative L2. A large weak residual means the error is visible to the test functions; it is not by itself a certificate of correctness.",
        "- The inverse probe interpolates sparse sensors before the strong and weak least-squares fits. Solver-in-the-loop sees only the sensors. No learned inverse was trained.",
        "- Shipped FNO/PINO checkpoints were trained under the original one-seed recipe. They are a rescore, not part of the five-seed budget.",
        "",
    ]
    return "\n".join(lines) + "\n"


def write_artifacts(result: dict, protocol: dict, digest: str, out_dir: Path, doc_path: Path, protocol_src: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = to_builtin(result)
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(protocol_src, out_dir / "protocol.json")
    (out_dir / "protocol.sha256").write_text(digest + "\n", encoding="utf-8")

    tables = out_dir / "tables"
    _write_csv(tables / "noise_injection.csv", _flatten_noise(result["noise"]))
    _write_csv(tables / "resolution_sweep.csv", result["resolution"]["rows"])
    _write_csv(tables / "symbol_heat_cases.csv", result["symbol"]["heat_cases"])
    _write_csv(tables / "coefficient_mismatch.csv", [
        {"pde": pde, **row} for pde, rows in result["mismatch"].items() for row in rows
    ])
    _write_csv(tables / "models.csv", _model_rows(result["models"]))
    _write_csv(tables / "kphi_ablation.csv", result["kphi"]["rows"])
    _write_csv(tables / "shipped_rescore.csv", [
        {
            "pde": r["pde"],
            "model": r["model"],
            "status": r["status"],
            "rel_l2": r.get("rel_l2"),
            "strong": r.get("strong"),
            "weak": r.get("weak"),
            "legacy_untrusted": None if not r.get("decision") else r["decision"].get("legacy_untrusted"),
            "verdict_survives_weak_form": None if not r.get("decision") else r["decision"].get("verdict_survives_weak_form"),
        }
        for r in result["shipped"]["rows"]
    ])
    _write_csv(tables / "hypotheses.csv", result["hypotheses"])
    inv_rows = []
    for split in ("id", "hard_ood"):
        for method, stats in result["inverse"][split].items():
            if not isinstance(stats, dict):
                continue
            if method == "fno_conditioned":
                inv_rows.append({"split": split, "method": method, "status": stats["status"], "rel_err_mean": None, "n_fail": None})
            else:
                inv_rows.append(
                    {
                        "split": split,
                        "method": method,
                        "status": "ok",
                        "rel_err_mean": stats["rel_err_mean"],
                        "rel_err_std": stats["rel_err_std"],
                        "n_fail": stats["n_fail"],
                    }
                )
    _write_csv(tables / "inverse.csv", inv_rows)
    paired_rows = [
        {
            "pde": pde,
            "contrast": "white_minus_smooth",
            "a": "white",
            "b": "smooth",
            "metric": metric,
            **test,
        }
        for pde, block in result["noise"]["pdes"].items()
        for metric, test in block["paired_white_minus_smooth"].items()
    ]
    for row in result["models"]["paired_models"]:
        paired_rows.append({"contrast": "model_minus_model", **row})
    _write_csv(tables / "paired_tests.csv", paired_rows)
    _figures(result, out_dir / "figures")
    text = _markdown(result, protocol, digest)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(text, encoding="utf-8")
    (out_dir / "weak_form_study.md").write_text(text, encoding="utf-8")
    manifest = {
        "protocol_sha256": digest,
        "script": "scripts/reproduce_paper.py",
        "artifacts": [
            {"path": "paper_artifacts/protocol.json", "source": "frozen protocol, copied byte-for-byte"},
            {"path": "paper_artifacts/summary.json", "source": "full run payload"},
            {"path": "paper_artifacts/tables/noise_injection.csv", "source": "noise injection on the paper datasets"},
            {"path": "paper_artifacts/tables/resolution_sweep.csv", "source": "resolution and dt sweep"},
            {"path": "paper_artifacts/tables/symbol_heat_cases.csv", "source": "heat white second-moment formula"},
            {"path": "paper_artifacts/tables/models.csv", "source": "five-seed model study and trivial baselines"},
            {"path": "paper_artifacts/tables/coefficient_mismatch.csv", "source": "labels scored at scaled coefficients"},
            {"path": "paper_artifacts/tables/shipped_rescore.csv", "source": "shipped FNO and PINO checkpoints"},
            {"path": "paper_artifacts/tables/inverse.csv", "source": "Burgers viscosity probe"},
            {"path": "paper_artifacts/tables/kphi_ablation.csv", "source": "test-function bandwidth"},
            {"path": "paper_artifacts/tables/paired_tests.csv", "source": "paired permutation tests"},
            {"path": "paper_artifacts/tables/hypotheses.csv", "source": "pre-registered checks"},
            {"path": "paper_artifacts/figures/fig_noise_strong_vs_weak.png", "source": "noise curves"},
            {"path": "paper_artifacts/figures/fig_resolution_sweep.png", "source": "resolution sweep"},
            {"path": "paper_artifacts/figures/fig_symbol_fit.png", "source": "formula comparison"},
            {"path": "paper_artifacts/figures/fig_error_spectra.png", "source": "one Burgers spectrum example"},
            {"path": "docs/research/weak_form_study.md", "source": "same prose as paper_artifacts/weak_form_study.md"},
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
