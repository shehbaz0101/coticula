# Coticula v0.1.0

Cite-ready packaging of **Coticula** (Coticula Understanding Bench) after
Weeks 1–4 on `main` (PRs #1–#3). Software citation: `CITATION.cff`
(Shehbaz Pathan; v0.1.0; https://github.com/shehbaz0101/coticula).

This release **does not invent metrics.** Measured IID / OOD / Exam 4 tables
are in `docs/WRITEUP.md` and `reports/latest.md`, produced by
`python -m scripts.run_eval`. Missing checkpoints stay `not_trained`.

## What’s in the tag

- **Week 1** — Classical FD labels (Burgers 1D, heat 2D), four exam APIs, eval harness.
- **Week 2** — Laptop-scale FNO + PINO-style residual; Exams 1–3 when a `*.pt` loads.
- **Week 3** — OOD probes (param / resolution / IC), fail-closed trust, writeup draft.
- **Week 4** — CPU CI, 36-item Exam 4 keyword set, SHA pins for labels + OOD generators.
- **Week 5** — `CITATION.cff`, `CHANGELOG.md`, `docs/RELEASE_v0.1.0.md`.

## Reproduce

```bash
pip install -e ".[dev,torch]"
python -m scripts.generate_labels
python -m scripts.train_fno --pde burgers
python -m scripts.train_fno --pde heat2d
python -m scripts.train_pino --pde burgers
python -m scripts.run_eval
pytest -q
```

Smoke (CI): `train_fno --smoke`, `train_pino --smoke --pde burgers`, `verify_manifests`.

## Non-claims

No chip-cooling / industrial multiphysics. No AU-scale / cosmology / foundation-model
workloads. No fabricated SOTA. Keyword rubrics are not explanation quality.
A `trusted` flag is not a claim that the model understands the PDE.

Full how-to + WRITEUP link: `docs/RELEASE_v0.1.0.md`.
