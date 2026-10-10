"""Score the protocol v2 amendment. Refuses to run if either digest mismatches.

Does not rewrite v1 tables. Appends the amendment section to the study note.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Reproduce the protocol v2 amendment")
    parser.add_argument("--out", type=Path, default=ROOT / "paper_artifacts")
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "paper_artifacts" / "summary.json",
        help="v1 summary used only to reuse selected hyperparameters",
    )
    args = parser.parse_args(argv)

    from research.amendment import run_amendment
    from research.amendment_report import write_amendment
    from research.protocol import PROTOCOL_V2_PATH, verify_amendment

    amendment, digest, protocol, _v1_digest = verify_amendment()
    threads = int(amendment["torch_threads"])
    try:
        import torch

        torch.set_num_threads(threads)
    except ImportError as exc:
        raise SystemExit("torch is required for the amendment retrain") from exc

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    print(f"Protocol v2 SHA-256 {digest}", flush=True)
    result = run_amendment(protocol, amendment, summary)
    write_amendment(result, amendment, digest, args.out, ROOT / "docs" / "research" / "weak_form_study.md", PROTOCOL_V2_PATH)
    print(f"Wrote v2 artifacts under {args.out}", flush=True)


if __name__ == "__main__":
    main()
