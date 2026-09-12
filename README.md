# Vermithor — VU-Bench v0

**VU = Vermithor Understanding Bench** (not the `uv` Python packager).

Public repo: [github.com/shehbaz0101/vermithor](https://github.com/shehbaz0101/vermithor)

**Thesis.** VU-Bench is Vermithor’s *understanding harness* for PDE surrogates:
four exams (Predict, Conserve, Counterfactual, Explain) that score whether a
model has grasped the dynamics—not just interpolated training trajectories.

Reports contain **only measured numbers**. Missing checkpoints stay
`not_trained`. The optional LLM judge is env-gated and **never invents scores**.

Writeup draft (tables + non-claims): [docs/WRITEUP.md](docs/WRITEUP.md).
Design / train support / trust rules: [docs/DESIGN.md](docs/DESIGN.md).

## Quickstart

Python 3.11+ and NumPy / SciPy. Torch is optional for classical-only eval
and is required to train or score FNO / PINO.

```bash
pip install -e ".[dev]"          # classical + pytest
pip install -e ".[dev,torch]"    # + PyTorch (CPU is enough)

python -m scripts.generate_labels   # IID NPZ + refresh SHA pins
python -m scripts.verify_manifests  # Exam 4 + OOD recipe; skip missing NPZ

# Tiny FNO / PINO (laptop / single-GPU friendly)
python -m scripts.train_fno --pde burgers
python -m scripts.train_fno --pde heat2d
python -m scripts.train_pino --pde burgers

python -m scripts.run_eval          # IID exams 1–4 + OOD + trust flags
python -m scripts.run_ood           # OOD only → reports/ood.json
python -m scripts.run_eval --skip-ood
pytest -q
```

CPU CI / smoke (no labels needed; this is what GitHub Actions runs):

```bash
python -m scripts.train_fno --smoke
python -m scripts.train_pino --smoke --pde burgers
python -m scripts.verify_manifests
```

CI workflow: [`.github/workflows/ci.yml`](.github/workflows/ci.yml) — `pytest -q`
plus the `--smoke` trains above on `ubuntu-latest`. Full `generate_labels` /
`run_eval` is **not** in CI (NPZ labels are gitignored; a full train is heavier
than a PR check).

## Week 1–4 status

| Week | Status | What shipped |
|------|--------|----------------|
| 1 | done | Classical FD labels, exam APIs, eval harness, smoke tests |
| 2 | done | Laptop-scale FNO + PINO residual; Exams 1–3 measured when a `*.pt` loads |
| 3 | done | OOD probes (param / resolution / IC), fail-closed trust, [WRITEUP.md](docs/WRITEUP.md) |
| 4 | this | GitHub Actions CI, Exam 4 **fixed item set** (36 law-keyword items), SHA pins for labels + OOD generators, README polish |

## How to read trust flags

Fail-closed layer: `metrics/trust.py`. A pretty field is never a silent success.

| Flag | Meaning |
|------|---------|
| `trusted` | In declared train support **and** residual rel-L2 ≤ 1.0 **and** Predict rel-L2 ≤ 0.5 |
| `ood` | Left train support (param / resolution / IC family); residual and L2 still below those cuts |
| `untrusted` | Residual or Predict L2 exceeds a cut (wins over `ood` if both apply) |
| `reference` | Classical FD labeler row — not a learned-transfer claim |
| `not_trained` | Checkpoint, torch, or LLM judge absent — **not** a placeholder accuracy |

OOD / untrusted fields must carry `metrics.trust.refuse_silent_heatmap()`.
Thresholds are documented defaults, not tuned after seeing scores.
`trusted` is **not** a claim that the model understands the PDE.

IID Exam 1–3 tables and the OOD section are **not mixed**. Transfer scores
live under `ood` in `reports/latest.json`.

## Latest measured results (this repo)

Copied from [`reports/latest.md`](reports/latest.md) (Week 3 eval run
`2026-09-12T00:18:43Z` on the shipped checkpoints). **Not invented.**
`—` / `not_trained` means that checkpoint or judge was absent.

| Baseline | Burgers rel-L2 | Heat2D rel-L2 | Burgers residual | Trust |
|----------|---------------:|--------------:|-----------------:|-------|
| classical_fd | 0 | 0 | 1.44e-2 | `reference` |
| FNO | 1.57e-1 | 3.22e-1 | 4.85 | `untrusted` |
| PINO | 1.88e-1 | — | 1.73 | `untrusted` / `not_trained` |
| LLM | — | — | — | `not_trained` |

Exam 4 gold-reference keyword coverage is computed from the pinned item set
(authored gold texts vs expected law keywords) — see the Exam 4 section of
`reports/latest.md` after `run_eval`. That number is a rubric ceiling, **not**
an LLM or FNO explanation score.

Full tables + failure analysis: [docs/WRITEUP.md](docs/WRITEUP.md).

## Layout

```
vermithor/
  datasets/{burgers,heat2d,manifests,ood,exam4}/
  datasets/pins.py              # SHA helpers
  datasets/exam4/items_v0.json  # Exam 4 item set (36)
  datasets/ood/probe_plan_v0.json
  metrics/{predict,conserve,counterfactual,explain,ood,trust}.py
  baselines/classical|fno|pino|llm/
  scripts/generate_labels.py    # IID labels + refresh pins
  scripts/pin_datasets.py       # refresh SHA manifests
  scripts/verify_manifests.py
  scripts/train_fno.py / train_pino.py
  scripts/run_eval.py / run_ood.py
  .github/workflows/ci.yml
  docs/DESIGN.md
  docs/WRITEUP.md
  reports/latest.{json,md}
```

## Dataset pins (regenerate path)

SHA manifests live in `datasets/manifests/`:

| Manifest | Pins |
|----------|------|
| `labels_v0.json` | `burgers_v0.npz` + meta, `heat2d_v0.npz` + meta |
| `ood_generators_v0.json` | `datasets/ood/probe_plan_v0.json` (knobs, seeds, `default_probe_plan()`) |
| `exam4_v0.json` | `datasets/exam4/items_v0.json` |

OOD trajectories are **generated on the fly** from the pinned recipe — they
are not stored as NPZ. `*.npz` is gitignored; after a fresh clone:

```bash
python -m scripts.generate_labels      # rewrite IID NPZ + all three manifests
python -m scripts.pin_datasets         # refresh OOD + Exam 4 (+ label SHAs if NPZ exist)
python -m scripts.verify_manifests     # CI default: skip missing label NPZ
python -m scripts.verify_manifests --require-labels
```

If `default_probe_plan()` or Exam 4 items change, re-run `pin_datasets` and
commit the updated plan + manifests. Tests check that the live probe plan
matches the pin.

## Exams

| Exam | What is measured |
|------|------------------|
| 1 Predict | Rel-L2 and NMSE vs classical labels |
| 2 Conserve | FD PDE residual + discrete energy drift of the predicted field |
| 3 Counterfactual | Re-solve / re-predict at ν′ or α′; learned models graded vs the classical solver |
| 4 Explain | **36-item** pinned set; keyword coverage vs expected law keywords. LLM hook stays `not_trained` |
| Trust (W3) | Fail-closed `trusted` / `ood` / `untrusted` / `reference` |
| OOD (W3) | Param shift, spatial resolution transfer, IC-family shift |

Useful knobs (all have small defaults): `--epochs`, `--width`, `--modes`,
`--n-layers`, `--batch-size`, `--lambda-pde`, `--out`.

`run_eval` always scores the classical solver. It loads FNO / PINO only when a
checkpoint is present. It does **not** invent metrics.

## Non-goals (unchanged)

- No chip-cooling / industrial multiphysics demos
- No AU-scale (astronomical-unit) or cosmology workloads
- No fabricated SOTA numbers for FNO / PINO / LLM
- No claim that keyword rubrics equal true explanation quality

## License

Internal Vermithor research scaffold.
