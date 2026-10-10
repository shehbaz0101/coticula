"""Regenerate every table and figure in paper_artifacts/.

Refuses to run unless research/protocols/protocol.json matches the frozen SHA-256.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Reproduce the weak-form residual study")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "paper_artifacts",
        help="Directory for CSV, JSON, and figures",
    )
    args = parser.parse_args(argv)

    from research.protocol import PROTOCOL_PATH, verify_frozen
    from research.pipeline import run_all
    from research.report import write_artifacts

    protocol, digest = verify_frozen()
    threads = int(protocol["torch_threads"])
    try:
        import torch

        torch.set_num_threads(threads)
    except ImportError as exc:
        raise SystemExit("torch is required to rescore checkpoints and train the model study") from exc

    print(f"Protocol SHA-256 {digest}", flush=True)
    result = run_all(protocol)
    write_artifacts(
        result,
        protocol,
        digest,
        args.out,
        ROOT / "docs" / "research" / "weak_form_study.md",
        PROTOCOL_PATH,
    )
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
