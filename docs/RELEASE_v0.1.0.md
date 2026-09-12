# Coticula v0.1.0 — what’s in the tag

**VU = Coticula Understanding Bench.** This note describes the intended
`v0.1.0` GitHub Release of [shehbaz0101/coticula](https://github.com/shehbaz0101/coticula).
It does **not** invent metrics. Measured tables live in
[WRITEUP.md](WRITEUP.md) (filled from `reports/latest.json`).

The annotated tag and GitHub Release are created **after** this packaging
merges to `main` (parent: `gh release create`). This PR does not cut the tag.

## What’s in v0.1.0

Weeks 1–4 of the understanding harness, already on `main` via PRs #1–#3:

| Week | What shipped |
|------|----------------|
| 1 | Classical FD labels, four exam APIs, eval harness, smoke tests |
| 2 | Laptop-scale FNO + PINO-style residual; Exams 1–3 when a `*.pt` loads |
| 3 | OOD probes (param / resolution / IC), fail-closed trust, writeup draft |
| 4 | CPU CI, 36-item Exam 4 set, SHA pins for labels + OOD generators |

Plus this Week 5 packaging: `CITATION.cff`, `CHANGELOG.md`, these notes.

Package version in `pyproject.toml` is already `0.1.0`. Checkpoints in
`checkpoints/` are optional laptop-scale weights; missing files stay
`not_trained`.

## How to reproduce

Requires Python 3.11+ . Torch is required to train or score FNO / PINO.

```bash
pip install -e ".[dev,torch]"          # CPU torch is enough
python -m scripts.generate_labels      # IID NPZ + refresh SHA pins
python -m scripts.verify_manifests

python -m scripts.train_fno --pde burgers
python -m scripts.train_fno --pde heat2d
python -m scripts.train_pino --pde burgers

python -m scripts.run_eval             # IID exams 1–4 + OOD + trust
pytest -q
```

Shipped checkpoints may be reused instead of retraining; `run_eval` never
invents weights or scores.

CPU smoke (what GitHub Actions runs; no labels needed):

```bash
python -m scripts.train_fno --smoke
python -m scripts.train_pino --smoke --pde burgers
python -m scripts.verify_manifests
pytest -q
```

OOD-only: `python -m scripts.run_ood`. Skip OOD: `python -m scripts.run_eval --skip-ood`.

## Where the numbers are

- Full tables + failure analysis: [WRITEUP.md](WRITEUP.md)
- Raw eval: `reports/latest.json` / `reports/latest.md`
  (`generated_at_utc`: 2026-09-12T00:18:43Z on the Week 3 checkpoints;
  Exam 4 gold-reference coverage graded on the Week 4 revision)
- Design / train support / trust rules: [DESIGN.md](DESIGN.md)
- Changelog (no duplicated tables): [../CHANGELOG.md](../CHANGELOG.md)

Do not copy digits from memory. If you re-run `generate_labels` + train +
`run_eval`, treat the new report as a new measurement.

## Non-claims

Unchanged from the writeup:

- No chip-cooling / industrial multiphysics results.
- No AU-scale (astronomical-unit) or cosmology / foundation-model workloads.
- No fabricated SOTA for FNO, PINO, or LLM.
- No claim that keyword rubrics equal explanation quality.
- No claim that a `trusted` flag means the model “understands” the PDE — only
  that it stayed inside declared support and below documented residual / L2
  cuts on this harness.

Exam 4 gold-reference coverage is the keyword rubric on authored gold texts
(a ceiling for the item set), **not** an LLM or FNO explanation score.

## Cite

Software citation: [`CITATION.cff`](../CITATION.cff) (author: Shehbaz Pathan;
version 0.1.0; url https://github.com/shehbaz0101/coticula).

## Parent: cut the tag after merge

This PR does not create `v0.1.0`. After merge to `main`:

```bash
git checkout main && git pull
git tag -a v0.1.0 -m "v0.1.0 — cite-ready Coticula (Weeks 1–4)"
git push origin v0.1.0
gh release create v0.1.0 --title "v0.1.0" --notes-file RELEASE_NOTES.md
```

Pasteable notes: [`RELEASE_NOTES.md`](../RELEASE_NOTES.md).
