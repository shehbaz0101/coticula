"""Scorecard writers: terminal text, Markdown, JSON, HTML, PNG."""
from __future__ import annotations

import json
import sys
import textwrap
from html import escape
from pathlib import Path

import numpy as np

from coticula.grading import eval_set_note, reading_lines

_STATUS_COLOR = {
    "trusted": "32",
    "untrusted": "31",
    "ood": "33",
    "reference": "36",
    "not_trained": "2",
    "error": "31",
}


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


def fmt_short(value) -> str:
    """Three decimal places for the terminal table. JSON and Markdown stay full."""
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return str(value)
    return f"{number:.3f}"


def _paint(text: str, status: str, *, color: bool) -> str:
    if not color:
        return text
    code = _STATUS_COLOR.get(str(status))
    if not code:
        return text
    return f"\033[{code}m{text}\033[0m"


def _rows(section: dict) -> list[dict]:
    return list(section.get("models") or [])


def _reason_text(row: dict) -> str:
    reasons = ",".join((row.get("trust") or {}).get("reasons") or [])
    if row.get("status") != "ok":
        reasons = str(row.get("reason") or reasons)
    return reasons


def residual_limitation_note() -> str:
    """Copy of the measured Burgers noise probe. Recompute with the script.

    ``python -m scripts.probe_residual_noise`` on 2026-10-10:
    clean residual rel-L2 1.597894e-02, 1% white noise 1.134730e+00.
    """
    return (
        "Known limitation: the strong-form finite-difference residual amplifies "
        "high-frequency error. On one exact Burgers trajectory from the label recipe "
        "(nx=64, nt=80, seed 0), 1% white noise (standard deviation 0.01 times the "
        "field standard deviation, noise seed 0) raises the residual rel-L2 from "
        "1.597894e-02 to 1.134730e+00, which crosses the residual cut of 1. "
        "Treat the residual flag as a roughness indicator for now. "
        "A weak-form, noise-calibrated residual is planned. "
        "Recompute with python -m scripts.probe_residual_noise."
    )


def _flag_text(row: dict) -> str:
    status = (row.get("trust") or {}).get("status") or row.get("status")
    reasons = list((row.get("trust") or {}).get("reasons") or [])
    if row.get("status") != "ok":
        reason = row.get("reason")
        if reason and reason not in {status, None, ""}:
            return f"{row.get('status')}, {reason}"
        return str(row.get("status") or status)
    if reasons:
        return f"{status}, {', '.join(reasons)}"
    return str(status)


def verdict_line(row: dict, *, pde: str | None = None) -> str:
    """Measured Predict, residual, and flag. No pass/fail claim about the PDE."""
    name = str(row.get("name"))
    prefix = f"{name} {pde}" if pde else name
    status = row.get("status")
    if status != "ok":
        if status == "not_trained" and row.get("reason") in {None, "", "missing_checkpoint"}:
            return f"{prefix}: no checkpoint shipped (flag: not_trained)"
        reason = row.get("reason") or "no metrics"
        return f"{prefix}: {status} ({reason}) (flag: {status})"
    pred = (row.get("predict") or {}).get("rel_l2_mean")
    res = (row.get("conserve") or {}).get("residual_rel_l2")
    return (
        f"{prefix}: Predict {fmt_short(pred)}, residual {fmt_short(res)} "
        f"(flag: {_flag_text(row)})"
    )


def markdown_table(card: dict) -> str:
    lines = [
        f"# Coticula {card.get('version')} scorecard",
        "",
        f"Command: `{card.get('command')}`",
        "",
        f"Generated (UTC): {card.get('generated_at_utc')}",
        "",
    ]
    note = card.get("eval_set") or eval_set_note(card)
    if note:
        lines.append(note)
        lines.append("")
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
        n_total = section.get("n_total")
        total = f" of {n_total}" if n_total is not None else ""
        lines.append(
            f"Split: {section.get('split_label') or 'eval'} "
            f"n_eval={section.get('n_eval')}{total} idx={section.get('eval_idx')}"
        )
        lines.append("")
        if section.get("split") == "holdout":
            lines.append(
                "Learned rows and classical_fd share this checkpoint holdout. "
                "`scripts.run_eval` scores classical_fd on the first trajectories instead, "
                "so that classical residual can differ. "
                "`coticula grade` without a shipped checkpoint scores every trajectory instead."
            )
        else:
            lines.append(
                "Learned rows and classical_fd share this split. "
                "`coticula demo` scores the checkpoint holdout "
                "(Burgers 12 trajectories, heat 8), so a toy adapter can differ there."
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
    lines.append("## Known limitation")
    lines.append("")
    lines.append(residual_limitation_note())
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    for sentence in card.get("reading") or reading_lines(card):
        lines.append(f"- {sentence}")
    lines.append("")
    return "\n".join(lines)


def _table_row(name: str, pred, res, trust, reasons: str, *, color: bool) -> str:
    trust_s = "-" if trust in {None, ""} else str(trust)
    reason_s = reasons or "-"
    trust_cell = f"{trust_s:<12}"
    return (
        f"{name:<16} {fmt_short(pred):>8} {fmt_short(res):>8}  "
        f"{_paint(trust_cell, trust_s, color=color)} {reason_s}"
    )


def _table_header() -> list[str]:
    return [
        f"{'model':<16} {'predict':>8} {'residual':>8}  {'trust':<12} reasons",
        f"{'-' * 16} {'-' * 8} {'-' * 8}  {'-' * 12} -------",
    ]


def terminal_scorecard(card: dict, *, color: bool | None = None) -> str:
    """Aligned scorecard. Color when ``color`` is true, or when stdout is a TTY."""
    if color is None:
        color = sys.stdout.isatty()
    lines = [
        f"Coticula {card.get('version')} scorecard",
        f"command: {card.get('command')}",
        card.get("eval_set") or eval_set_note(card),
    ]
    if card.get("smoke"):
        lines.append("smoke: tiny grid, not the shipped checkpoint")
    pdes = card.get("pdes") or {}
    multi = len(pdes) > 1
    for pde, section in pdes.items():
        n_total = section.get("n_total")
        total = f" of {n_total}" if n_total is not None else ""
        idx = [int(i) for i in (section.get("eval_idx") or [])]
        span = ""
        if idx and idx == list(range(idx[0], idx[-1] + 1)):
            span = f"  indices {idx[0]}-{idx[-1]}"
        lines.append("")
        lines.append(
            f"{pde}  {section.get('split_label') or 'eval'}  "
            f"n={section.get('n_eval')}{total}{span}"
        )
        lines.extend(_table_header())
        for row in _rows(section):
            trust = (row.get("trust") or {}).get("status") or row.get("status")
            pred = (row.get("predict") or {}).get("rel_l2_mean") if row.get("predict") else None
            res = (row.get("conserve") or {}).get("residual_rel_l2") if row.get("conserve") else None
            lines.append(
                _table_row(
                    str(row.get("name")),
                    pred,
                    res,
                    trust,
                    _reason_text(row),
                    color=color,
                )
            )
        for probe in section.get("ood") or []:
            lines.append(
                f"OOD {probe.get('id')}  param={probe.get('param')}  n={probe.get('n')}"
            )
            lines.extend(_table_header())
            for row in probe.get("models") or []:
                if row.get("role") == "labeler":
                    continue
                trust = (row.get("trust") or {}).get("status") or row.get("status")
                pred = (row.get("predict") or {}).get("rel_l2_mean") if row.get("predict") else None
                res = (row.get("conserve") or {}).get("residual_rel_l2") if row.get("conserve") else None
                lines.append(
                    _table_row(
                        str(row.get("name")),
                        pred,
                        res,
                        trust,
                        _reason_text(row),
                        color=color,
                    )
                )
    explain = card.get("explain") or {}
    gold = (explain.get("gold") or {}).get("mean_coverage")
    rule = (explain.get("rule_based") or {}).get("mean_coverage")
    dump = (explain.get("metric_dump") or {}).get("mean_coverage")
    llm = (explain.get("llm") or {}).get("status")
    lines.append("")
    lines.append(
        f"Exam 4  n={explain.get('n_items')}  gold={fmt_short(gold)}  "
        f"rule-based={fmt_short(rule)}  metric-dump={fmt_short(dump)}  llm={llm}"
    )
    lines.append("")
    for pde, section in pdes.items():
        for row in _rows(section):
            line = verdict_line(row, pde=pde if multi else None)
            status = (row.get("trust") or {}).get("status") or row.get("status")
            token = str(status)
            needle = f"flag: {token}"
            if color and needle in line:
                line = line.replace(
                    needle,
                    "flag: " + _paint(token, token, color=True),
                    1,
                )
            lines.append(line)
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
    note = card.get("eval_set") or ""
    if note:
        parts.append(f"<p>{escape(note)}</p>")
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
        n_total = section.get("n_total")
        total = f" of {n_total}" if n_total is not None else ""
        parts.append(
            f"<p>Labels: {escape(str(section.get('label_source')))}. "
            f"Split: {escape(str(section.get('split_label') or 'eval'))}, "
            f"n={escape(str(section.get('n_eval')))}{escape(total)}.</p>"
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
    parts.append("<h2>Known limitation</h2>")
    parts.append(f"<p>{escape(residual_limitation_note())}</p>")
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


def _wrap_to_width(text: str, max_chars: int) -> str:
    """Wrap a banner into balanced lines that fit the figure width."""
    clean = " ".join(str(text).split())
    if not clean or len(clean) <= max_chars:
        return clean
    n_lines = max(2, (len(clean) + max_chars - 1) // max_chars)
    target = min(max_chars, max(28, (len(clean) // n_lines) + 6))
    lines = textwrap.wrap(clean, width=target)
    if any(len(line) > max_chars for line in lines):
        lines = textwrap.wrap(clean, width=max_chars)
    return "\n".join(lines)


def _max_title_chars(fig, fontsize: float = 9.0) -> int:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    probe = fig.text(0, 0, "n", fontsize=fontsize, alpha=0)
    fig.canvas.draw()
    char_w = max(probe.get_window_extent(renderer).width, 1.0)
    probe.remove()
    return max(24, int((fig.bbox.width - 24) / char_w))


def _texts_outside(fig, tol: float = 1.0) -> list[str]:
    from matplotlib.text import Text

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.bbox.width, fig.bbox.height
    bad: list[str] = []
    for text in fig.findobj(lambda artist: isinstance(artist, Text)):
        value = text.get_text()
        if not value or not str(value).strip() or not text.get_visible():
            continue
        box = text.get_window_extent(renderer=renderer)
        if box.width <= 0 or box.height <= 0:
            continue
        if box.x0 < -tol or box.y0 < -tol or box.x1 > width + tol or box.y1 > height + tol:
            bad.append(" ".join(str(value).split()))
    return bad


def _field_caption(pde: str, sample: dict, section: dict) -> str:
    """Short measured caption. The trust module's long banner stays in the JSON."""
    name = str(sample.get("name") or "field")
    row = next((item for item in _rows(section) if item.get("name") == name), None)
    if row is None or row.get("status") != "ok":
        return f"{pde} {name} (flag: {_flag_text(row or {'status': 'not_trained'})})"
    pred = (row.get("predict") or {}).get("rel_l2_mean")
    res = (row.get("conserve") or {}).get("residual_rel_l2")
    return (
        f"{pde} {name}: Predict {fmt_short(pred)}, residual {fmt_short(res)} "
        f"(flag: {_flag_text(row)})"
    )


def _place_title(fig, text: str) -> None:
    wrapped = _wrap_to_width(text, _max_title_chars(fig))
    fig.suptitle(wrapped or text, fontsize=9)
    fig.canvas.draw()
    outside = _texts_outside(fig)
    if outside:
        raise RuntimeError("plot text extends past the figure: " + " | ".join(outside))


def _style_field_axis(ax) -> None:
    from matplotlib.ticker import MaxNLocator

    locator = MaxNLocator(nbins=4, integer=True, prune="both")
    ax.xaxis.set_major_locator(locator)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True, prune="both"))
    ax.tick_params(labelsize=8)


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
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), layout="constrained")
        axes[0].bar(labels, preds, color="#3d5a80")
        axes[0].axhline(pred_cut, color="#9b2226", linestyle="--", linewidth=1)
        axes[0].set_title("Predict rel-L2")
        axes[0].set_ylabel("rel-L2")
        axes[1].bar(labels, ress, color="#3d5a80")
        axes[1].axhline(res_cut, color="#9b2226", linestyle="--", linewidth=1)
        axes[1].set_title("residual rel-L2")
        for ax, values in ((axes[0], preds), (axes[1], ress)):
            top = max(values) if values else 1.0
            hi = top * 1.28 + 1e-6
            ax.set_ylim(0, hi)
            ticks = [tick for tick in ax.get_yticks() if 0 <= tick <= hi]
            if ticks:
                ax.set_yticks(ticks)
            for i, val in enumerate(values):
                ax.text(i, val, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
        for ax in axes:
            ax.tick_params(axis="x", labelrotation=15, labelsize=8)
        _place_title(fig, f"{pde}: dashed line is the documented threshold")
        filename = f"bars_{pde}.png"
        fig.savefig(out_dir / filename, dpi=120)
        plt.close(fig)
        names.append(filename)

        sample = arrays.get((pde, "fno")) or arrays.get((pde, "persist_ic"))
        if sample is None:
            continue
        truth = np.asarray(sample["truth"])
        pred = np.asarray(sample["pred"])
        fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.8), layout="constrained")
        if truth.ndim == 2:
            panels = (truth, pred, np.abs(pred - truth))
            titles = ("truth", sample["name"], "|error|")
        else:
            mid = truth.shape[0] // 2
            panels = (truth[mid], pred[mid], np.abs(pred[mid] - truth[mid]))
            titles = (f"truth t={mid}", sample["name"], "|error|")
        for ax, img, title in zip(axes, panels, titles):
            kwargs = {"origin": "lower", "cmap": "viridis"}
            if truth.ndim == 2:
                kwargs["aspect"] = "auto"
            im = ax.imshow(img, **kwargs)
            ax.set_title(title, fontsize=10)
            _style_field_axis(ax)
            colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            colorbar.ax.tick_params(labelsize=7)
        _place_title(fig, _field_caption(pde, sample, section))
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
