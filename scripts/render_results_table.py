"""Print the README results table from reports/latest.json.

The JSON is written by ``python -m scripts.run_eval``. This script does not
recompute metrics and does not invent cells.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coticula.report import render_published_table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report",
        nargs="?",
        type=Path,
        default=ROOT / "reports" / "latest.json",
    )
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    sys.stdout.write(render_published_table(report))


if __name__ == "__main__":
    main()
