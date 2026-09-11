"""Train a PINO checkpoint (FNO trunk + data and PDE residual losses).

Equivalent to ``python -m scripts.train_fno --pino``. Loss terms are documented
in ``baselines/pino/losses.py`` and ``docs/DESIGN.md``.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_fno import build_parser, main as train_main


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Default on: PINO residual. Caller may still pass --pino explicitly.
    if "--pino" not in argv:
        argv.append("--pino")
    # Re-parse so --help mentions we are the PINO entry point.
    build_parser().parse_args(argv)
    train_main(argv)


if __name__ == "__main__":
    main()
