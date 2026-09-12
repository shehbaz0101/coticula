"""Refresh SHA manifests for labels, OOD generators, and Exam 4.

Usage::

    python -m scripts.pin_datasets
    python -m scripts.pin_datasets --require-labels
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.pins import pin_all


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Write SHA manifests for VU-Bench pins")
    p.add_argument(
        "--require-labels",
        action="store_true",
        help="Fail if burgers_v0.npz / heat2d_v0.npz are missing",
    )
    args = p.parse_args(argv)
    out = pin_all(skip_missing_labels=not args.require_labels)
    for kind, man in out.items():
        print(f"{kind}: {man.get('n_files', 0)} file(s)")
        for e in man.get("files") or []:
            print(f"  {e['path']}: {e['sha256'][:16]}... ({e['bytes']} bytes)")
        missing = man.get("missing_on_disk") or man.get("missing") or []
        for m in missing:
            print(f"  missing on disk (pin kept): {m}")


if __name__ == "__main__":
    main()
