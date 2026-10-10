"""Write protocol-v2 tables and append them to the study note.

v1 files are left in place. The study markdown keeps its v1 sections and
replaces only the amendment section on a repeat run.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from research.report import _fmt, _md_table, _pm, to_builtin

MARKER = "## Protocol v2 amendment"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


def _ladder_rows(result: dict) -> list[dict]:
    rows = []
    curves = result["curves"]
    mismatch = result["mismatch"]
    for pde, block in curves.items():
        levels = block["levels"]
        matched = {row["level"]: row for row in mismatch["matched"][pde]}
        for i, level in enumerate(levels):
            smooth_s = block["smooth"]["strong"]["median"][i]
            smooth_w = block["smooth"]["weak"]["median"][i]
            white_s = block["white"]["strong"]["median"][i]
            white_w = block["white"]["weak"]["median"][i]
            mis = matched[level]
            rows.append(
                {
                    "pde": pde,
                    "comparison": "noise_grid",
                    "rel_l2": level,
                    "scale": mis["direction_scale"],
                    "white_rho_s": white_s,
                    "smooth_rho_s": smooth_s,
                    "mismatch_rho_s": mis["strong_median"],
                    "white_rho_w": white_w,
                    "smooth_rho_w": smooth_w,
                    "mismatch_rho_w": mis["weak_median"],
                    "smooth_rho_s_p05": block["smooth"]["strong"]["p05"][i],
                    "smooth_rho_s_p95": block["smooth"]["strong"]["p95"][i],
                    "smooth_rho_w_p05": block["smooth"]["weak"]["p05"][i],
                    "smooth_rho_w_p95": block["smooth"]["weak"]["p95"][i],
                    "white_C_s": white_s / smooth_s,
                    "white_C_w": white_w / smooth_w,
                    "mismatch_C_s": mis["C_strong"],
                    "mismatch_C_w": mis["C_weak"],
                    "extrapolated": False,
                    "n_band": block["smooth"]["strong"]["n"][i],
                }
            )
        for row in mismatch["natural"][pde]:
            rows.append(
                {
                    "pde": pde,
                    "comparison": "natural_scale",
                    "rel_l2": row["rel_l2_mean"],
                    "scale": row["scale"],
                    "white_rho_s": row["white_strong_at_mean_rel"],
                    "smooth_rho_s": row["smooth_strong_at_mean_rel"],
                    "mismatch_rho_s": row["strong_median"],
                    "white_rho_w": row["white_weak_at_mean_rel"],
                    "smooth_rho_w": row["smooth_weak_at_mean_rel"],
                    "mismatch_rho_w": row["weak_median"],
                    "white_C_s": None
                    if not row["smooth_strong_at_mean_rel"]
                    else row["white_strong_at_mean_rel"] / row["smooth_strong_at_mean_rel"],
                    "white_C_w": None
                    if not row["smooth_weak_at_mean_rel"]
                    else row["white_weak_at_mean_rel"] / row["smooth_weak_at_mean_rel"],
                    "mismatch_C_s": row["C_strong"]["mean"],
                    "mismatch_C_w": row["C_weak"]["mean"],
                    "mismatch_C_s_std": row["C_strong"]["std"],
                    "mismatch_C_w_std": row["C_weak"]["std"],
                    "extrapolated": row["smooth_extrapolated"],
                    "n_extrapolated": row["n_extrapolated"],
                    "n": row["n"],
                }
            )
    return rows


def _model_rows(result: dict) -> list[dict]:
    rows = []
    models = result["models"]
    for pde, baselines in models["baselines"].items():
        for name, pack in baselines.items():
            rows.append(
                {
                    "pde": pde,
                    "model": name,
                    "kind": "baseline",
                    "seed": None,
                    "rel_l2": pack["rel_l2"]["mean"],
                    "S_rel_l2": pack["S_rel_l2"],
                    "strong": pack["strong"]["mean"],
                    "weak": pack["weak"]["mean"],
                    "C_strong": pack["C_strong"]["mean"],
                    "C_strong_std": pack["C_strong"]["std"],
                    "C_weak": pack["C_weak"]["mean"],
                    "C_weak_std": pack["C_weak"]["std"],
                }
            )
        for arch, block in models["models"][pde].items():
            summary = block["summary"]
            rows.append(
                {
                    "pde": pde,
                    "model": arch,
                    "kind": "trained_mean",
                    "seed": None,
                    "rel_l2": summary["rel_l2"]["mean"],
                    "rel_l2_std": summary["rel_l2"]["std"],
                    "S_rel_l2": summary["S_rel_l2"]["mean"],
                    "strong": summary["strong"]["mean"],
                    "weak": summary["weak"]["mean"],
                    "C_strong": summary["C_strong"]["mean"],
                    "C_strong_std": summary["C_strong"]["std"],
                    "C_weak": summary["C_weak"]["mean"],
                    "C_weak_std": summary["C_weak"]["std"],
                }
            )
            for seed_row in block["per_seed"]:
                rows.append(
                    {
                        "pde": pde,
                        "model": arch,
                        "kind": "seed",
                        "seed": seed_row["seed"],
                        "rel_l2": seed_row["rel_l2"],
                        "S_rel_l2": seed_row["S_rel_l2"],
                        "strong": seed_row["strong"],
                        "weak": seed_row["weak"],
                        "C_strong": seed_row["C_strong"],
                        "C_weak": seed_row["C_weak"],
                    }
                )
    return rows


def _figures(result: dict, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.6), sharex=True)
    for row, pde in enumerate(("burgers", "heat")):
        block = result["curves"][pde]
        levels = block["levels"]
        matched = {item["level"]: item for item in result["mismatch"]["matched"][pde]}
        for col, metric, label in ((0, "strong", r"$\rho_s$"), (1, "weak", r"$\rho_w$")):
            ax = axes[row, col]
            med = block["smooth"][metric]["median"]
            lo = block["smooth"][metric]["p05"]
            hi = block["smooth"][metric]["p95"]
            ax.fill_between(levels, lo, hi, color="#E69F00", alpha=0.25, label="smooth 5–95%")
            ax.plot(levels, med, color="#E69F00", marker="s", label="smooth median")
            ax.plot(levels, block["white"][metric]["median"], color="#0072B2", marker="o", label="white median")
            ax.plot(
                levels,
                [matched[level]["strong_median" if metric == "strong" else "weak_median"] for level in levels],
                color="#D55E00",
                marker="D",
                ls="--",
                label="mismatch x2, matched rel-L2",
            )
            markers = {0.0: "x", 0.5: "P", 2.0: "*"}
            for item in result["mismatch"]["natural"][pde]:
                y = item["strong_median"] if metric == "strong" else item["weak_median"]
                scale = float(item["scale"])
                ax.scatter(
                    [item["rel_l2_mean"]],
                    [y],
                    s=42,
                    zorder=4,
                    color="#000000",
                    marker=markers.get(scale, "o"),
                    label=f"scale {scale:g} natural" if col == 0 and row == 0 else None,
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{pde} {label}")
            ax.set_xlabel("relative L2")
            ax.grid(True, which="both", ls=":", alpha=0.5)
    axes[0, 0].legend(frameon=False, fontsize=7)
    axes[0, 0].set_ylabel("residual")
    axes[1, 0].set_ylabel("residual")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_matched_ladder.png", dpi=150)
    fig.savefig(fig_dir / "fig_matched_ladder.pdf")
    plt.close(fig)


def _story_lines(result: dict) -> list[str]:
    lines = []
    for pde, note in result["story"].items():
        s_lo, s_hi = min(note["strong_white_over_smooth"]), max(note["strong_white_over_smooth"])
        w_lo, w_hi = min(note["weak_white_over_smooth"]), max(note["weak_white_over_smooth"])
        m_lo, m_hi = min(note["strong_mismatch_over_smooth"]), max(note["strong_mismatch_over_smooth"])
        k_lo, k_hi = min(note["weak_mismatch_over_smooth"]), max(note["weak_mismatch_over_smooth"])
        lines.append(
            f"{pde}: strong white/smooth median ratio spans {_fmt(s_lo)} to {_fmt(s_hi)}; "
            f"weak white/smooth spans {_fmt(w_lo)} to {_fmt(w_hi)}. "
            f"Amplitude-matched scale-2 mismatch over smooth spans {_fmt(m_lo)} to {_fmt(m_hi)} (strong) "
            f"and {_fmt(k_lo)} to {_fmt(k_hi)} (weak)."
        )
        if s_lo > 1:
            lines.append(f"{pde} strong form ranks white above smooth at every noise level on this grid.")
        if s_hi < 10:
            lines.append(
                f"{pde} strong white/smooth stays below 10. On this coarse grid the midpoint time term "
                "dominates the spatial roughness term, so white noise is not orders of magnitude above smooth noise."
            )
        if 0.5 <= m_lo and m_hi <= 2:
            lines.append(f"{pde} strong mismatch stays within a factor of two of smooth noise at matched relative L2.")
        elif m_hi < 1:
            lines.append(f"{pde} strong mismatch is below the smooth median at every matched level.")
        else:
            lines.append(
                f"{pde} strong mismatch is not interchangeable with smooth noise across the whole ladder."
            )
        if k_hi < 1:
            lines.append(
                f"{pde} weak mismatch C is below 1 at every matched level: the coefficient-shift shape is "
                "more PDE-consistent than a generic smooth error of the same size."
            )
        elif k_lo > 1:
            lines.append(
                f"{pde} weak mismatch C is above 1 at every matched level: the coefficient-shift shape is "
                "less PDE-consistent than a generic smooth error of the same size."
            )
        elif k_hi < 1.05 and sum(v > 1 for v in note["weak_mismatch_over_smooth"]) <= 1:
            lines.append(
                f"{pde} weak mismatch C stays below 1 except for a single level at {_fmt(k_hi)}, "
                "next to the discretization floor."
            )
        else:
            lines.append(
                f"{pde} weak mismatch C crosses 1 across the ladder, so the weak form does not rank "
                "coefficient shifts uniformly against smooth noise."
            )
        above_floor = note["strong_mismatch_over_smooth"][1:]
        if above_floor and max(above_floor) < 1:
            lines.append(
                f"{pde} does not support smooth ≈ mismatch in the strong form. "
                f"For ε ≥ 0.01, strong mismatch/smooth spans {_fmt(min(above_floor))} to {_fmt(max(above_floor))}."
            )
    return lines


def _markdown(result: dict, amendment: dict, digest: str) -> str:
    grid_rows = []
    natural_rows = []
    for row in _ladder_rows(result):
        if row["comparison"] == "noise_grid":
            grid_rows.append(
                [
                    row["pde"],
                    _fmt(row["rel_l2"]),
                    _fmt(row["white_rho_s"]),
                    _fmt(row["smooth_rho_s"]),
                    _fmt(row["mismatch_rho_s"]),
                    _fmt(row["white_C_s"]),
                    _fmt(row["mismatch_C_s"]),
                    _fmt(row["white_rho_w"]),
                    _fmt(row["smooth_rho_w"]),
                    _fmt(row["mismatch_rho_w"]),
                    _fmt(row["white_C_w"]),
                    _fmt(row["mismatch_C_w"]),
                ]
            )
        else:
            outside = "yes" if row["extrapolated"] else "no"
            natural_rows.append(
                [
                    row["pde"],
                    _fmt(row["scale"]),
                    _fmt(row["rel_l2"]),
                    outside,
                    "outside grid" if row["extrapolated"] else _fmt(row["white_rho_s"]),
                    "outside grid" if row["extrapolated"] else _fmt(row["smooth_rho_s"]),
                    _fmt(row["mismatch_rho_s"]),
                    "—" if row["extrapolated"] else _fmt(row["mismatch_C_s"]),
                    "outside grid" if row["extrapolated"] else _fmt(row["white_rho_w"]),
                    "outside grid" if row["extrapolated"] else _fmt(row["smooth_rho_w"]),
                    _fmt(row["mismatch_rho_w"]),
                    "—" if row["extrapolated"] else _fmt(row["mismatch_C_w"]),
                ]
            )
    model_rows = []
    for row in _model_rows(result):
        if row["kind"] == "seed":
            continue
        model_rows.append(
            [
                row["pde"],
                row["model"],
                _fmt(row["rel_l2"]) if row["kind"] == "baseline" else _pm(row["rel_l2"], row.get("rel_l2_std")),
                _fmt(row["S_rel_l2"]),
                _fmt(row["strong"]),
                _fmt(row["C_strong"]),
                _fmt(row["weak"]),
                _fmt(row["C_weak"]),
            ]
        )
    paired_rows = []
    for row in result["models"]["paired"]:
        paired_rows.append(
            [
                row["pde"],
                f"{row['a']} - {row['b']}",
                row["metric"],
                _fmt(row["mean_diff"]),
                _fmt(row["cohen_dz"]),
                _fmt(row["p_value"]),
                str(row["n"]),
            ]
        )
    shipped_rows = []
    for row in result["shipped"]:
        shipped_rows.append(
            [
                row["pde"],
                row["model"],
                row["status"],
                _fmt(row.get("rel_l2")),
                _fmt(row.get("strong")),
                "—" if row.get("extrapolated") else _fmt(row.get("C_strong")),
                _fmt(row.get("weak")),
                "—" if row.get("extrapolated") else _fmt(row.get("C_weak")),
                _fmt(row.get("legacy_untrusted")),
                _fmt(row.get("verdict_survives_weak_form")),
            ]
        )
    change_lines = [f"- {item['what']} {item['why']}" for item in amendment["changes"]]
    diff = result["models"]["retrain_max_abs_rel_diff"]
    lines = [
        MARKER,
        "",
        f"Protocol v2 SHA-256: `{digest}`",
        "",
        "This section amends the frozen v1 protocol. v1 tables above are unchanged. "
        "S is used only for relative L2. Residual columns are the calibrated ratio "
        "C = rho / median smooth residual at the same relative L2. "
        "The smooth error is the v1 time-varying low-mode error. "
        "The band is the 5th to 95th percentile, log-log interpolated on the noise grid. "
        "C is omitted from the reported mean when the relative L2 falls outside that grid.",
        "",
        "### What changed",
        "",
        *change_lines,
        "",
        "### Matched relative L2: white, smooth, and a scale-2 coefficient shift",
        "",
        "Mismatch on this grid is the error u(2c) − u(c), rescaled to the row's relative L2 and added to u(c). "
        "Residuals use the original coefficient. Entries are medians.",
        "",
        _md_table(
            ["PDE", "ε", "white ρ_s", "smooth ρ_s", "mismatch ρ_s", "white C_s", "mismatch C_s", "white ρ_w", "smooth ρ_w", "mismatch ρ_w", "white C_w", "mismatch C_w"],
            grid_rows,
        ),
        "",
        "### Natural coefficient scales",
        "",
        "The field is the classical solution at scale × coefficient, same initial condition. "
        "White and smooth columns are the noise medians interpolated to that field's mean relative L2. "
        "C is the mean of per-trajectory ratios inside the noise grid.",
        "",
        _md_table(
            ["PDE", "scale", "rel-L2", "clipped", "white ρ_s", "smooth ρ_s", "mismatch ρ_s", "C_s", "white ρ_w", "smooth ρ_w", "mismatch ρ_w", "C_w"],
            natural_rows,
        ),
        "",
        "### Models",
        "",
        "Five seeds were retrained from the v1 selected hyperparameters. "
        f"Max absolute difference versus the v1 per-seed relative L2 is burgers {diff['burgers']:.3g}, heat {diff['heat']:.3g}. "
        "The v1 model table is not replaced.",
        "",
        _md_table(
            ["PDE", "model", "rel-L2", "S(rel-L2)", "ρ_s", "C_s", "ρ_w", "C_w"],
            model_rows,
        ),
        "",
        "### Paired tests",
        "",
        "Wilcoxon signed-rank test on paired differences over seeds × test trajectories. "
        "Cohen's dz is the mean difference divided by the sample standard deviation of the differences.",
        "",
        _md_table(
            ["PDE", "contrast", "metric", "mean diff", "Cohen dz", "p", "n"],
            paired_rows,
        ),
        "",
        "### Shipped checkpoints under the smooth band",
        "",
        "Survival means the weak residual is strictly above the 95% smooth band at the field's relative L2. "
        "The band is built on that checkpoint's evaluation trajectories, not on the paper grid. "
        "Outside the noise grid, survival is not called.",
        "",
        _md_table(
            ["PDE", "model", "status", "rel-L2", "ρ_s", "C_s", "ρ_w", "C_w", "legacy untrusted", "weak survival"],
            shipped_rows,
        ),
        "",
        "### Reading against the expected ranking",
        "",
        *_story_lines(result),
        "",
        "### Wall clock (seconds)",
        "",
        _md_table(
            ["stage", "seconds"],
            [[name, f"{seconds:.3f}"] for name, seconds in result["timings_seconds"].items()],
        ),
        "",
    ]
    return "\n".join(lines) + "\n"


def _append(path: Path, section: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if MARKER in existing:
        existing = existing[: existing.index(MARKER)].rstrip() + "\n\n"
    elif existing and not existing.endswith("\n"):
        existing += "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(existing + "\n" + section, encoding="utf-8")


def write_amendment(result: dict, amendment: dict, digest: str, out_dir: Path, doc_path: Path, protocol_src: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(protocol_src, out_dir / "protocol_v2.json")
    (out_dir / "protocol_v2.sha256").write_text(digest + "\n", encoding="utf-8")
    payload = to_builtin(result)
    (out_dir / "v2_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tables = out_dir / "tables"
    _write_csv(tables / "v2_matched_ladder.csv", _ladder_rows(result))
    _write_csv(tables / "v2_calibrated_ratios.csv", _model_rows(result))
    _write_csv(tables / "v2_paired_tests.csv", result["models"]["paired"])
    _write_csv(tables / "v2_shipped_survival.csv", result["shipped"])
    _figures(result, out_dir / "figures")
    section = _markdown(result, amendment, digest)
    _append(doc_path, section)
    _append(out_dir / "weak_form_study.md", section)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"artifacts": []}
    manifest["protocol_v2_sha256"] = digest
    kept = [
        item
        for item in manifest.get("artifacts", [])
        if "v2" not in item["path"] and "matched_ladder" not in item["path"]
    ]
    kept.extend(
        [
            {"path": "paper_artifacts/protocol_v2.json", "source": "v2 amendment, copied byte-for-byte"},
            {"path": "paper_artifacts/protocol_v2.sha256", "source": "SHA-256 of protocol_v2.json"},
            {"path": "paper_artifacts/v2_summary.json", "source": "amendment run payload"},
            {"path": "paper_artifacts/tables/v2_matched_ladder.csv", "source": "white, smooth, and mismatch at matched rel-L2"},
            {"path": "paper_artifacts/tables/v2_calibrated_ratios.csv", "source": "S for rel-L2 and C for residuals"},
            {"path": "paper_artifacts/tables/v2_paired_tests.csv", "source": "Wilcoxon tests over seeds x trajectories"},
            {"path": "paper_artifacts/tables/v2_shipped_survival.csv", "source": "shipped models against the smooth band"},
            {"path": "paper_artifacts/figures/fig_matched_ladder.png", "source": "matched residual ladder"},
        ]
    )
    manifest["artifacts"] = kept
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
