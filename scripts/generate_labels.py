"""Generate classical Burgers + heat2d labels and SHA256 manifests."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.classical.burgers1d import generate_dataset as gen_burgers
from baselines.classical.heat2d import generate_dataset as gen_heat


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_npz(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def write_manifest(entries: list[dict], out_path: Path) -> dict:
    payload = {
        "version": "vu-bench-v0",
        "n_files": len(entries),
        "files": entries,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    data_root = ROOT / "datasets"
    burgers_dir = data_root / "burgers"
    heat_dir = data_root / "heat2d"
    man_dir = data_root / "manifests"

    print("Generating Burgers 1D labels...")
    burgers = gen_burgers(n_traj=48, nx=64, nt=80, seed=0)
    b_path = burgers_dir / "burgers_v0.npz"
    _save_npz(
        b_path,
        {
            "u": burgers["u"],
            "x": burgers["x"],
            "t": burgers["t"],
            "nu": burgers["nu"],
        },
    )
    meta_b = burgers_dir / "burgers_v0_meta.json"
    meta_b.write_text(json.dumps(burgers["meta"], indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {b_path} shape={burgers['u'].shape}")

    print("Generating heat 2D labels...")
    heat = gen_heat(n_traj=32, n=32, nt=40, seed=1)
    h_path = heat_dir / "heat2d_v0.npz"
    _save_npz(
        h_path,
        {
            "u": heat["u"],
            "x": heat["x"],
            "y": heat["y"],
            "t": heat["t"],
            "alpha": heat["alpha"],
        },
    )
    meta_h = heat_dir / "heat2d_v0_meta.json"
    meta_h.write_text(json.dumps(heat["meta"], indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {h_path} shape={heat['u'].shape}")

    entries = []
    for p in [b_path, meta_b, h_path, meta_h]:
        entries.append(
            {
                "path": str(p.relative_to(ROOT)),
                "sha256": _sha256_file(p),
                "bytes": p.stat().st_size,
            }
        )
    man_path = man_dir / "labels_v0.json"
    write_manifest(entries, man_path)
    print(f"Manifest: {man_path}")
    for e in entries:
        print(f"  {e['path']}: {e['sha256'][:16]}... ({e['bytes']} bytes)")


if __name__ == "__main__":
    main()
