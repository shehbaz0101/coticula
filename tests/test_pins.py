"""CPU tests for SHA dataset pins (labels, OOD generators, Exam 4)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.pins import (
    EXAM4_ITEMS_PATH,
    EXAM4_MANIFEST,
    OOD_MANIFEST,
    OOD_PLAN_PATH,
    ood_generator_contract,
    sha256_file,
    verify_all,
)
from metrics.ood import default_probe_plan


def test_exam4_and_ood_manifests_match():
    report = verify_all(skip_missing_labels=True)
    assert report["ok"], report
    kinds = {r["manifest"] for r in report["reports"]}
    assert "datasets/manifests/exam4_v0.json" in kinds
    assert "datasets/manifests/ood_generators_v0.json" in kinds


def test_ood_plan_matches_default_probe_plan():
    assert OOD_PLAN_PATH.exists()
    pinned = json.loads(OOD_PLAN_PATH.read_text(encoding="utf-8"))
    live = default_probe_plan()
    assert pinned["probe_plan"] == live
    # Contract helper must stay in lockstep with the pin file.
    contract = ood_generator_contract()
    assert contract["probe_plan"] == live
    assert contract["knobs"]["burgers_ood_nu"]["below"] == 0.001


def test_exam4_manifest_sha():
    man = json.loads(EXAM4_MANIFEST.read_text(encoding="utf-8"))
    assert man["files"][0]["sha256"] == sha256_file(EXAM4_ITEMS_PATH)
    ood = json.loads(OOD_MANIFEST.read_text(encoding="utf-8"))
    assert ood["files"][0]["sha256"] == sha256_file(OOD_PLAN_PATH)


def test_labels_manifest_skips_missing_npz():
    report = verify_all(skip_missing_labels=True)
    labels = next(r for r in report["reports"] if r["manifest"].endswith("labels_v0.json"))
    for row in labels["files"]:
        if row["status"] == "missing":
            assert row["path"].endswith(".npz")
            assert row["ok"] is True
