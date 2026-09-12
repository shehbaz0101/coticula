"""Verify SHA256 pins. Missing label NPZ is OK unless --require-labels.

Usage::

    python -m scripts.verify_manifests
    python -m scripts.verify_manifests --require-labels
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.pins import verify_all


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Verify Coticula SHA manifests")
    p.add_argument(
        "--require-labels",
        action="store_true",
        help="Fail if a pinned label file is missing (NPZ are gitignored)",
    )
    p.add_argument(
        "--skip-missing-labels",
        action="store_true",
        default=True,
        help=argparse.SUPPRESS,
    )
    args = p.parse_args(argv)
    report = verify_all(skip_missing_labels=not args.require_labels)
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        print("MANIFEST MISMATCH", file=sys.stderr)
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
