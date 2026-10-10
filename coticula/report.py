"""Scorecard writers: terminal text, Markdown, JSON, HTML, PNG."""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np

from coticula.grading import reading_lines


def json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj))


def fmt(value, spec: str = ".6e") -> str:
    if value is None:
        return "-"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def _rows(section: dict) -> list[dict]:
    return list(section.get("models") or [])


def markdown_table(card: dict) -> str:
    lines = [
        f"# Coticula {card.get('version')} scorecard",
        "",
        f"Command: `{card.get('command')}`",
        "",
        f"Generated (UTC): {card.get('generated_at_utc')}",
        "",
    ]
    if card.get("smoke"):
        lines.append("Smoke run. Not the shipped checkpoint.")
        lines.append("")
    policy = card.get("trust_policy") or {}
    lines.append(
        f"Fail-closed cuts: Predict rel-L2 > {policy.get('predict_untrusted')} "
        f"or residual rel-L2 > {policy.get('residual_untrusted')} is untrusted. "
        "Outside train support is ood. untrusted wins if both apply."
    )
    lines.append("")
    for pde, section in (card.get("pdes") or {}).items():
        lines.append(f"## {pde}")
        lines.append("")
        lines.append(f"Labels: {section.get('label_source')}")
        lines.append("")
        lines.append(f"Holdout n={section.get('n_eval')} idx={section.get('eval_idx')}")
        lines.append("")
        lines.append(
            "Learned rows and classical_fd share this holdout. "
            "`scripts.run_eval` scores classical_fd on the first trajectories instead, "
            "so that classical residual can differ."
        )
        lines.append("")
        lines.append("| model | Predict rel-L2 | residual rel-L2 | trust | reasons |")
        lines.append("|---|---:|---:|---|---|")
        for row in _rows(section):
            pred = (row.get("predict") or {}).get("rel_l2_mean")
            res = (row.get("conserve") or {}).get("residual_rel_l2")
            trust = (row.get("trust") or {}).get("status") or row.get("status")
            reasons = ",".join((row.get("trust") or {}).get("reasons") or [])
            if row.get("status") != "ok":
                reasons = row.get("reason") or reasons
            lines.append(
                f"| {row.get('name')} | {fmt(pred)} | {fmt(res)} | {trust} | {reasons or '-'} |"
            )
        lines.append("")
        for probe in section.get("ood") or []:
            lines.append(
                f"### OOD {probe.get('id')} (param={probe.get('param')}, "
                f"n={probe.get('n')}, resolution={probe.get('resolution')})"
            )
            lines.append("")
            lines.append(probe.get("note") or "")
            lines.append("")
            lines.append("| model | Predict rel-L2 | residual rel-L2 | trust | reasons |")
            lines.append("|---|---:|---:|---|---|")
            for row in probe.get("models") or []:
                pred = (row.get("predict") or {}).get("rel_l2_mean")
                res = (row.get("conserve") or {}).get("residual_rel_l2")
                trust = (row.get("trust") or {}).get("status") or row.get("status")
                reasons = ",".join((row.get("trust") or {}).get("reasons") or [])
                if row.get("status") != "ok":
                    reasons = row.get("reason") or reasons
                lines.append(
                    f"| {row.get('name')} | {fmt(pred)} | {fmt(res)} | {trust} | {reasons or '-'} |"
                )
            lines.append("")
    explain = card.get("explain") or {}
    lines.append("## Exam 4")
    lines.append("")
    sha = explain.get("item_set_sha256") or ""
    lines.append(
        f"Items: {explain.get('n_items')} from `{explain.get('item_set')}` "
        f"sha256 `{sha[:16]}`."
    )
    lines.append("")
    for key, label in (
        ("gold", "gold"),
        ("rule_based", "rule-based"),
        ("metric_dump", "metric-dump"),
    ):
        block = explain.get(key) or {}
        lines.append(
            f"- {label}: {fmt(block.get('mean_coverage'), '.6f')} "
            f"(status {block.get('status')}, perfect {block.get('n_perfect')}, "
            f"zero {block.get('n_zero')})"
        )
    llm = explain.get("llm") or {}
    lines.append(f"- llm: {llm.get('status')} (score {llm.get('score')})")
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    for sentence in card.get("reading") or reading_lines(card):
        lines.append(f"- {sentence}")
    lines.append("")
    return "\n".join(lines)


def terminal_scorecard(card: dict) -> str:
    lines = [
        f"Coticula {card.get('version')} scorecard",
        f"command: {card.get('command')}",
    ]
    if card.get("smoke"):
        lines.append("smoke: tiny grid, not the shipped checkpoint")
    for pde, section in (card.get("pdes") or {}).items():
        lines.append("")
        lines.append(
            f"{pde}  n_eval={section.get('n_eval')}  labels={section.get('label_source')}"
        )
        lines.append(
            f"{'model':<16} {'predict rel-L2':>14} {'residual':>14} {'trust':<12} reasons"
        )
        for row in _rows(section):
            pred = (row.get("predict") or {}).get("rel_l2_mean")
            res = (row.get("conserve") or {}).get("residual_rel_l2")
            trust = (row.get("trust") or {}).get("status") or row.get("status")
            reasons = ",".join((row.get("trust") or {}).get("reasons") or [])
            if row.get("status") != "ok":
                reasons = str(row.get("reason") or "")
            lines.append(
                f"{str(row.get('name')):<16} {fmt(pred):>14} {fmt(res):>14} {str(trust):<12} {reasons}"
            )
        for probe in section.get("ood") or []:
            lines.append(
                f"OOD {probe.get('id')} param={probe.get('param')} n={probe.get('n')}"
            )
            for row in probe.get("models") or []:
                if row.get("role") == "labeler":
                    continue
                pred = (row.get("predict") or {}).get("rel_l2_mean")
                res = (row.get("conserve") or {}).get("residual_rel_l2")
                trust = (row.get("trust") or {}).get("status") or row.get("status")
                reasons = ",".join((row.get("trust") or {}).get("reasons") or [])
                if row.get("status") != "ok":
                    reasons = str(row.get("reason") or "")
                lines.append(
                    f"  {str(row.get('name')):<14} {fmt(pred):>14} {fmt(res):>14} {str(trust):<12} {reasons}"
                )
    explain = card.get("explain") or {}
    gold = (explain.get("gold") or {}).get("mean_coverage")
    rule = (explain.get("rule_based") or {}).get("mean_coverage")
    dump = (explain.get("metric_dump") or {}).get("mean_coverage")
    llm = (explain.get("llm") or {}).get("status")
    lines.append("")
    lines.append(
        f"Exam 4 n={explain.get('n_items')} gold={fmt(gold, '.6f')} "
        f"rule-based={fmt(rule, '.6f')} metric-dump={fmt(dump, '.6f')} llm={llm}"
    )
    lines.append("")
    for sentence in card.get("reading") or []:
        lines.append(sentence)
    return "\n".join(lines)


def html_report(card: dict, plot_names: list[str]) -> str:
    parts = [
        "<!DOCTYPE html>",
        "<html lang='en'><head><meta charset='utf-8'>",
        f"<title>Coticula {escape(str(card.get('version')))} scorecard</title>",
        "<style>",
        "body{font-family:Georgia,serif;max-width:880px;margin:2rem auto;padding:0 1rem;color:#1c1c1c;}",
        "table{border-collapse:collapse;width:100%;margin:0.5rem 0 1.25rem;}",
        "th,td{border-bottom:1px solid #ddd;padding:0.35rem 0.5rem;text-align:left;}",
        "td.num{text-align:right;font-family:ui-monospace,monospace;}",
        ".reading{background:#f6f4ef;padding:0.75rem 1rem;}",
        "img{max-width:100%;height:auto;margin:0.5rem 0 1rem;}",
        "code{font-family:ui-monospace,monospace;}",
        "</style></head><body>",
        f"<h1>Coticula {escape(str(card.get('version')))} scorecard</h1>",
        f"<p>Command: <code>{escape(str(card.get('command')))}</code><br>",
        f"Generated (UTC): {escape(str(card.get('generated_at_utc')))}</p>",
    ]
    if card.get("smoke"):
        parts.append("<p>Smoke run. Not the shipped checkpoint.</p>")
    policy = card.get("trust_policy") or {}
    parts.append(
        "<p>Fail-closed cuts: Predict rel-L2 &gt; "
        f"{escape(str(policy.get('predict_untrusted')))} or residual rel-L2 &gt; "
        f"{escape(str(policy.get('residual_untrusted')))} is untrusted. "
        "Outside train support is ood. untrusted wins if both apply.</p>"
    )
    for pde, section in (card.get("pdes") or {}).items():
        parts.append(f"<h2>{escape(pde)}</h2>")
        parts.append(
            f"<p>Labels: {escape(str(section.get('label_source')))}. "
            f"Holdout n={escape(str(section.get('n_eval')))}.</p>"
        )
        parts.append(_html_model_table(section.get("models") or []))
        for probe in section.get("ood") or []:
            parts.append(
                f"<h3>OOD {escape(str(probe.get('id')))}</h3>"
                f"<p>{escape(str(probe.get('note') or ''))} "
                f"param={escape(str(probe.get('param')))}, n={escape(str(probe.get('n')))}.</p>"
            )
            parts.append(_html_model_table(probe.get("models") or []))
    explain = card.get("explain") or {}
    parts.append("<h2>Exam 4</h2><ul>")
    for key, label in (
        ("gold", "gold"),
        ("rule_based", "rule-based"),
        ("metric_dump", "metric-dump"),
    ):
        block = explain.get(key) or {}
        parts.append(
            f"<li>{label}: {escape(fmt(block.get('mean_coverage'), '.6f'))} "
            f"({escape(str(block.get('status')))})</li>"
        )
    llm = explain.get("llm") or {}
    parts.append(f"<li>llm: {escape(str(llm.get('status')))}</li></ul>")
    parts.append("<h2>Reading</h2><div class='reading'><ul>")
    for sentence in card.get("reading") or []:
        parts.append(f"<li>{escape(sentence)}</li>")
    parts.append("</ul></div>")
    if plot_names:
        parts.append("<h2>Plots</h2>")
        for name in plot_names:
            parts.append(f"<img alt='{escape(name)}' src='{escape(name)}'>")
    parts.append("</body></html>")
    return "\n".join(parts)


def _html_model_table(rows: list[dict]) -> str:
    body = [
        "<table><thead><tr><th>model</th><th>Predict rel-L2</th>"
        "<th>residual rel-L2</th><th>trust</th><th>reasons</th></tr></thead><tbody>"
    ]
    for row in rows:
        pred = (row.get("predict") or {}).get("rel_l2_mean")
        res = (row.get("conserve") or {}).get("residual_rel_l2")
        trust = (row.get("trust") or {}).get("status") or row.get("status")
        reasons = ",".join((row.get("trust") or {}).get("reasons") or [])
        if row.get("status") != "ok":
            reasons = str(row.get("reason") or reasons)
        body.append(
            "<tr>"
            f"<td>{escape(str(row.get('name')))}</td>"
            f"<td class='num'>{escape(fmt(pred))}</td>"
            f"<td class='num'>{escape(fmt(res))}</td>"
            f"<td>{escape(str(trust))}</td>"
            f"<td>{escape(reasons or '-')}</td>"
            "</tr>"
        )
    body.append("</tbody></table>")
    return "".join(body)


def write_plots(card: dict, arrays: dict, out_dir: Path) -> list[str]:
    """Write PNG files. Returns filenames. Empty if matplotlib is missing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    names: list[str] = []
    policy = card.get("trust_policy") or {}
    pred_cut = float(policy.get("predict_untrusted") or 0.5)
    res_cut = float(policy.get("residual_untrusted") or 1.0)
    for pde, section in (card.get("pdes") or {}).items():
        ok = [row for row in _rows(section) if row.get("status") == "ok"]
        if not ok:
            continue
        labels = [str(row["name"]) for row in ok]
        preds = [float(row["predict"]["rel_l2_mean"]) for row in ok]
        ress = [float(row["conserve"]["residual_rel_l2"]) for row in ok]
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
        axes[0].bar(labels, preds, color="#3d5a80")
        axes[0].axhline(pred_cut, color="#9b2226", linestyle="--", linewidth=1)
        axes[0].set_title("Predict rel-L2")
        axes[0].set_ylabel("rel-L2")
        axes[1].bar(labels, ress, color="#3d5a80")
        axes[1].axhline(res_cut, color="#9b2226", linestyle="--", linewidth=1)
        axes[1].set_title("Conserve residual rel-L2")
        for ax, values in ((axes[0], preds), (axes[1], ress)):
            top = max(values) if values else 1.0
            ax.set_ylim(0, top * 1.18 + 1e-6)
            for i, val in enumerate(values):
                ax.text(i, val, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
        for ax in axes:
            ax.tick_params(axis="x", labelrotation=20)
        fig.suptitle(f"{pde}: dashed line is the fail-closed cut")
        fig.tight_layout()
        filename = f"bars_{pde}.png"
        fig.savefig(out_dir / filename, dpi=120)
        plt.close(fig)
        names.append(filename)

        sample = arrays.get((pde, "fno")) or arrays.get((pde, "persist_ic"))
        if sample is None:
            continue
        truth = np.asarray(sample["truth"])
        pred = np.asarray(sample["pred"])
        fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2))
        if truth.ndim == 2:
            panels = (truth, pred, np.abs(pred - truth))
            titles = ("truth", sample["name"], "|error|")
            for ax, img, title in zip(axes, panels, titles):
                im = ax.imshow(img, aspect="auto", origin="lower", cmap="viridis")
                ax.set_title(title)
                fig.colorbar(im, ax=ax, fraction=0.046)
        else:
            mid = truth.shape[0] // 2
            panels = (truth[mid], pred[mid], np.abs(pred[mid] - truth[mid]))
            titles = (f"truth t={mid}", sample["name"], "|error|")
            for ax, img, title in zip(axes, panels, titles):
                im = ax.imshow(img, origin="lower", cmap="viridis")
                ax.set_title(title)
                fig.colorbar(im, ax=ax, fraction=0.046)
        banner = str(sample.get("banner") or "")
        fig.suptitle(banner[:180] if banner else sample["name"])
        fig.tight_layout()
        filename = f"field_{pde}_{sample['name']}.png"
        fig.savefig(out_dir / filename, dpi=120)
        plt.close(fig)
        names.append(filename)
    return names


def write_scorecard(card: dict, arrays: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    json_path = out_dir / "scorecard.json"
    md_path = out_dir / "scorecard.md"
    html_path = out_dir / "report.html"
    json_path.write_text(
        json.dumps(card, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(markdown_table(card), encoding="utf-8")
    plot_names = write_plots(card, arrays, out_dir)
    html_path.write_text(html_report(card, plot_names), encoding="utf-8")
    paths.extend([json_path, md_path, html_path])
    paths.extend(out_dir / name for name in plot_names)
    return paths


def render_published_table(report: dict) -> str:
    """Markdown table from ``reports/latest.json`` (``scripts.run_eval``)."""
    exams = report["exams"]
    baselines = report["baselines"]
    lines = [
        "| model | pde | Predict rel-L2 | residual rel-L2 | trust |",
        "|---|---|---:|---:|---|",
    ]

    def add(model: str, pde: str, pred, res, trust: str) -> None:
        lines.append(
            f"| {model} | {pde} | {fmt(pred)} | {fmt(res)} | {trust} |"
        )

    b = baselines["classical"]["burgers"]
    h = baselines["classical"]["heat2d"]
    add("classical_fd", "burgers", b["predict"]["rel_l2_mean"], b["conserve"]["residual_rel_l2"], b["trust"]["status"])
    add("classical_fd", "heat2d", h["predict"]["rel_l2_mean"], h["conserve"]["residual_rel_l2"], h["trust"]["status"])
    for model in ("fno", "pino"):
        for pde in ("burgers", "heat2d"):
            row = (baselines.get(model) or {}).get(pde) or {}
            if row.get("status") != "ok":
                add(model, pde, None, None, row.get("status") or "not_trained")
                continue
            add(
                model,
                pde,
                row["predict"]["rel_l2_mean"],
                row["conserve"]["residual_rel_l2"],
                (row.get("trust") or {}).get("status") or row.get("status"),
            )
    explain = exams["explain"]
    gold = explain["gold_reference"]["mean_coverage"]
    rule = explain["rule_based"]["mean_coverage"]
    dump = explain["metric_dump"]["mean_coverage"]
    sha = explain.get("item_set_sha256") or ""
    diag = (report.get("diagnostics") or {}).get("residual_vs_l2") or {}
    lines.extend(
        [
            "",
            f"Exam 4 n={explain.get('n_items')} sha256={sha[:16]} "
            f"gold={gold:.6f} rule-based={rule:.6f} metric-dump={dump:.6f} "
            f"llm={explain['llm']['status']}",
            "",
            f"Residual-vs-L2 points n={diag.get('n')} "
            f"pearson={diag.get('pearson_residual_vs_l2')} "
            f"spearman={diag.get('spearman_residual_vs_l2')} "
            f"untrusted={diag.get('n_untrusted')} ood_flag={diag.get('n_ood')}",
            "",
            f"Generated (UTC): {report.get('generated_at_utc')}",
        ]
    )
    return "\n".join(lines) + "\n"
