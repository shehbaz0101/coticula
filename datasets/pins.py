"""SHA256 dataset pins for labels, OOD generators, and Exam 4.

Regenerate / refresh pins::

    python -m scripts.generate_labels   # IID NPZ + labels_v0.json (when you need new labels)
    python -m scripts.pin_datasets      # refresh OOD + Exam 4 (+ label SHAs if NPZ exist)
    python -m scripts.verify_manifests  # check pinned files (skip missing NPZ)

OOD trajectories are **not** stored as NPZ. The pin is the generator recipe
(``datasets/ood/probe_plan_v0.json``): knobs, seeds, and ``default_probe_plan()``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]

LABELS_MANIFEST = ROOT / "datasets" / "manifests" / "labels_v0.json"
OOD_MANIFEST = ROOT / "datasets" / "manifests" / "ood_generators_v0.json"
EXAM4_MANIFEST = ROOT / "datasets" / "manifests" / "exam4_v0.json"
OOD_PLAN_PATH = ROOT / "datasets" / "ood" / "probe_plan_v0.json"
EXAM4_ITEMS_PATH = ROOT / "datasets" / "exam4" / "items_v0.json"

LABEL_FILES = (
    ROOT / "datasets" / "burgers" / "burgers_v0.npz",
    ROOT / "datasets" / "burgers" / "burgers_v0_meta.json",
    ROOT / "datasets" / "heat2d" / "heat2d_v0.npz",
    ROOT / "datasets" / "heat2d" / "heat2d_v0_meta.json",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dump_canonical(payload: dict) -> str:
    """Stable JSON (sorted keys, trailing newline) for pin hashes."""
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def write_manifest(entries: list[dict], out_path: Path, **extra) -> dict:
    payload = {
        "version": "coticula-v0",
        "n_files": len(entries),
        "files": entries,
        **extra,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def file_entry(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _jsonable_support(support: dict) -> dict:
    out = {}
    for k, v in support.items():
        if isinstance(v, tuple):
            out[k] = [float(v[0]), float(v[1])]
        elif isinstance(v, (float, int, str)):
            out[k] = v
        else:
            out[k] = v
    return out


def ood_generator_contract() -> dict:
    """Frozen OOD recipe: knobs + default probe plan (not stored trajectories)."""
    from metrics.ood import (
        BURGERS_OOD_IC,
        BURGERS_OOD_NU,
        BURGERS_OOD_NX,
        HEAT_OOD_ALPHA,
        HEAT_OOD_IC,
        HEAT_OOD_N,
        default_probe_plan,
    )
    from metrics.trust import BURGERS_TRAIN_SUPPORT, HEAT_TRAIN_SUPPORT

    return {
        "version": "coticula-v0",
        "kind": "ood_generators",
        "note": (
            "OOD trajectories are generated on the fly from this recipe. "
            "Do not mix these probes into IID Exam 1 tables."
        ),
        "knobs": {
            "burgers_ood_nu": dict(BURGERS_OOD_NU),
            "heat_ood_alpha": dict(HEAT_OOD_ALPHA),
            "burgers_ood_nx": list(BURGERS_OOD_NX),
            "heat_ood_n": list(HEAT_OOD_N),
            "burgers_ood_ic": list(BURGERS_OOD_IC),
            "heat_ood_ic": list(HEAT_OOD_IC),
        },
        "train_support": {
            "burgers": _jsonable_support(BURGERS_TRAIN_SUPPORT),
            "heat2d": _jsonable_support(HEAT_TRAIN_SUPPORT),
        },
        "entry_points": {
            "make_burgers_ic": "metrics.ood.make_burgers_ic",
            "make_heat_ic": "metrics.ood.make_heat_ic",
            "generate_burgers_cases": "metrics.ood.generate_burgers_cases",
            "generate_heat_cases": "metrics.ood.generate_heat_cases",
            "default_probe_plan": "metrics.ood.default_probe_plan",
            "materialize_probe": "metrics.ood.materialize_probe",
        },
        "probe_plan": default_probe_plan(),
        "regenerate": [
            "python -m scripts.generate_labels",
            "python -m scripts.pin_datasets",
            "python -m scripts.verify_manifests",
        ],
    }


def write_ood_plan(path: Path | None = None) -> dict:
    path = path or OOD_PLAN_PATH
    payload = ood_generator_contract()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_canonical(payload), encoding="utf-8")
    return payload


def pin_labels(skip_missing: bool = True) -> dict:
    """Refresh label SHAs. Missing NPZ keep their last published pin."""
    previous: dict[str, dict] = {}
    if LABELS_MANIFEST.exists():
        old = json.loads(LABELS_MANIFEST.read_text(encoding="utf-8"))
        previous = {e["path"]: e for e in old.get("files") or []}
    entries = []
    missing = []
    for p in LABEL_FILES:
        rel = str(p.relative_to(ROOT))
        if p.exists():
            entries.append(file_entry(p))
        elif rel in previous:
            entries.append(previous[rel])
            missing.append(rel)
        else:
            missing.append(rel)
    extra = {
        "kind": "labels",
        "regenerate": "python -m scripts.generate_labels",
        "note": (
            "NPZ labels are gitignored. Re-run generate_labels to recreate them. "
            "pin_datasets keeps last published SHAs when NPZ are absent."
        ),
    }
    if missing and not skip_missing:
        raise FileNotFoundError("missing label files: " + ", ".join(missing))
    return write_manifest(entries, LABELS_MANIFEST, **extra)


def pin_ood() -> dict:
    write_ood_plan()
    return write_manifest(
        [file_entry(OOD_PLAN_PATH)],
        OOD_MANIFEST,
        kind="ood_generators",
        regenerate="python -m scripts.pin_datasets",
        note=(
            "Pins the OOD generator recipe (knobs + probe plan + seeds). "
            "Trajectories are rematerialized by metrics.ood.materialize_probe."
        ),
    )


def pin_exam4() -> dict:
    if not EXAM4_ITEMS_PATH.exists():
        raise FileNotFoundError(EXAM4_ITEMS_PATH)
    return write_manifest(
        [file_entry(EXAM4_ITEMS_PATH)],
        EXAM4_MANIFEST,
        kind="exam4",
        regenerate="python -m scripts.pin_datasets",
        note="Fixed Exam 4 explanation item set (expected law keywords).",
    )


def pin_all(*, skip_missing_labels: bool = True) -> dict:
    return {
        "labels": pin_labels(skip_missing=skip_missing_labels),
        "ood": pin_ood(),
        "exam4": pin_exam4(),
    }


def verify_one(manifest_path: Path, *, skip_missing: bool = False) -> dict:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = []
    ok = True
    for entry in data.get("files") or []:
        rel = entry["path"]
        path = ROOT / rel
        row = {"path": rel, "expected": entry.get("sha256")}
        if not path.exists():
            row["status"] = "missing"
            row["ok"] = skip_missing
            if not skip_missing:
                ok = False
        else:
            got = sha256_file(path)
            row["got"] = got
            row["ok"] = got == entry.get("sha256")
            row["status"] = "ok" if row["ok"] else "mismatch"
            if not row["ok"]:
                ok = False
        results.append(row)
    return {"manifest": str(manifest_path.relative_to(ROOT)), "ok": ok, "files": results}


def verify_all(*, skip_missing_labels: bool = True) -> dict:
    reports = [
        verify_one(EXAM4_MANIFEST, skip_missing=False),
        verify_one(OOD_MANIFEST, skip_missing=False),
        verify_one(LABELS_MANIFEST, skip_missing=skip_missing_labels),
    ]
    return {
        "ok": all(r["ok"] for r in reports),
        "reports": reports,
        "skip_missing_labels": skip_missing_labels,
    }


def load_exam4_sha() -> str | None:
    if not EXAM4_ITEMS_PATH.exists():
        return None
    return sha256_file(EXAM4_ITEMS_PATH)
