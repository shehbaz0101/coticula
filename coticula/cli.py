"""Command line: ``coticula demo`` and ``coticula grade``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from coticula import __version__
from coticula.adapter import load_adapter_spec, repo_root
from coticula.grading import build_scorecard
from coticula.report import terminal_scorecard, write_scorecard

DEMO_MODELS = ["classical_fd", "fno", "pino", "persist_ic"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coticula",
        description="Grade a PDE surrogate on Predict, Conserve, Counterfactual, and Explain.",
    )
    parser.add_argument("--version", action="version", version=f"coticula {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    demo = sub.add_parser(
        "demo",
        help="Score the shipped FNO and PINO plus the toy persist_ic adapter.",
    )
    demo.add_argument(
        "--pde",
        choices=("burgers", "heat2d", "both"),
        default="both",
        help="Which equation (default: both).",
    )
    demo.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: reports/demo).",
    )
    demo.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny Burgers grid and a 1-epoch FNO. For CI. Not the shipped checkpoint.",
    )
    demo.add_argument("--no-ood", action="store_true", help="Skip the param-shift probe.")
    demo.add_argument("--n-eval", type=int, default=None, help="Cap the holdout size.")

    grade = sub.add_parser(
        "grade",
        help="Score one model. classical_fd is included as the reference row.",
    )
    grade.add_argument(
        "--model",
        choices=("fno", "pino", "classical", "persist_ic"),
        default=None,
        help="Shipped checkpoint or the built-in toy adapter.",
    )
    grade.add_argument(
        "--adapter",
        default=None,
        help="User model as path.py:ClassName (see examples/persist_ic.py).",
    )
    grade.add_argument(
        "--pde",
        choices=("burgers", "heat2d"),
        required=True,
    )
    grade.add_argument("--checkpoint", type=Path, default=None)
    grade.add_argument("--out", type=Path, default=None)
    grade.add_argument("--smoke", action="store_true")
    grade.add_argument("--no-ood", action="store_true")
    grade.add_argument("--n-eval", type=int, default=None)
    return parser


def _pdes(name: str, smoke: bool) -> list[str]:
    if smoke:
        return ["burgers"]
    if name == "both":
        return ["burgers", "heat2d"]
    return [name]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = repo_root()
    shown = list(sys.argv[1:] if argv is None else argv)
    command = "coticula " + " ".join(shown)
    if getattr(args, "smoke", False) and args.pde not in {"burgers", "both"}:
        raise SystemExit("smoke is Burgers only")
    if args.cmd == "demo":
        out = args.out or (root / "reports" / "demo")
        names = ["classical_fd", "fno", "persist_ic"] if args.smoke else list(DEMO_MODELS)
        card, arrays = build_scorecard(
            pdes=_pdes(args.pde, args.smoke),
            model_names=names,
            smoke=args.smoke,
            n_eval=args.n_eval,
            include_ood=not args.no_ood,
            command=command.strip(),
            out_dir=out,
        )
    else:
        if args.smoke and args.pde == "heat2d":
            raise SystemExit("smoke is Burgers only")
        if args.model is None and args.adapter is None:
            raise SystemExit("grade needs --model or --adapter")
        if args.model is not None and args.adapter is not None:
            raise SystemExit("pass either --model or --adapter, not both")
        out = args.out or (root / "reports" / "grade")
        user = load_adapter_spec(args.adapter) if args.adapter else None
        names = ["classical_fd"]
        if args.model == "persist_ic":
            names.append("persist_ic")
        elif args.model in {"fno", "pino"}:
            names.append(args.model)
        card, arrays = build_scorecard(
            pdes=_pdes(args.pde, args.smoke),
            model_names=names,
            smoke=args.smoke,
            n_eval=args.n_eval,
            include_ood=not args.no_ood,
            checkpoint=args.checkpoint,
            adapter=user,
            command=command.strip(),
            out_dir=out,
        )
    print(terminal_scorecard(card))
    written = write_scorecard(card, arrays, out)
    print()
    for path in written:
        print(f"wrote {path}")
    if not any(path.suffix == ".png" for path in written):
        print("PNG plots were not written (matplotlib is not installed).")
    return 0
