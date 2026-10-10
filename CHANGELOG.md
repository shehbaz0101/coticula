# Changelog

Factual record of what shipped. Numeric tables are not copied here. They live in
[`docs/WRITEUP.md`](docs/WRITEUP.md), [`reports/latest.md`](reports/latest.md),
and the README, and were produced by `python -m scripts.run_eval`,
`python -m scripts.render_results_table`, and `coticula demo`.
This file does not invent metrics.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The package version in `pyproject.toml` is `0.3.2`. The 0.3.2 entry does not
create a tag.

## [0.3.2] - 2026-10-10

CLI and report consistency. No new measurements. The residual metric and the
trust rule are unchanged. No tag.

### Changed

- `coticula demo`, `scorecard.md`, and `report.html` open with a counterfactual
  table (model, PDE, model trajectory change, classical trajectory change,
  model vs classical at the new coefficient, and one plain-English line).
  The cells are the Exam 3 values already on the scorecard.
- `report.html` Reading uses the same short flag lines as the terminal
  (`fno burgers: Predict 0.157, residual 4.849 (flag: untrusted, residual_too_high)`).
  The Known limitation section stays.
- The README and the CLI footer state why Exam 4 rule-based and metric-dump
  differ between `grade` and `demo` on the same holdout: those answers append
  observations copied only from the FNO and PINO rows on the card. `grade`
  scores one model, so the other cells are written `not_trained`.
- The README notes again that PINO heat2d has no checkpoint shipped.

## [0.3.1] - 2026-10-10

Interview-demo polish. No new measurements and no new checkpoints.

### Changed

- Field-plot titles are a short measured caption (Predict, residual, flag)
  wrapped so they fit the figure. `field_burgers_fno.png` and
  `field_heat2d_fno.png` were checked, and so were the bar charts. The long
  trust-module banner stays in the JSON and is not the plot headline.
- `coticula demo` and `coticula grade` print an aligned scorecard. Color is
  used when stdout is a TTY, and plain text otherwise. Terminal numbers are
  short (`0.157`). JSON and Markdown keep full precision. Each model ends
  with the measured values and the flag, for example
  `fno: Predict 0.157, residual 4.849 (flag: untrusted, residual_too_high)`.
- The residual flag is described as a roughness indicator. A noise probe,
  `python -m scripts.probe_residual_noise`, measured residual rel-L2
  1.134730e+00 after 1% white noise on one exact Burgers trajectory. The
  trust rule is unchanged. The README leads with the reproduced
  counterfactual comparison.
- The terminal output and the README name the evaluation set. `demo` scores
  the checkpoint holdout (Burgers 12, heat 8). `grade` scores that holdout
  when a shipped checkpoint is loaded, and every trajectory otherwise
  (Burgers 48, heat 32). `grade --split holdout` matches `demo`.
- The README labels the PINO heat2d `not_trained` row: no checkpoint shipped,
  to keep the repo small.

## [0.3.0] - 2026-10-10

CLI and a documented way to grade a user surrogate. No new training run of the
shipped checkpoints. `python -m scripts.run_eval` was re-run on this date.
Learned Predict, Conserve, OOD, Exam 4, and diagnostic cells match the v0.2
report. The previous `reports/latest.md` note that IID cells were copied from
an earlier run is removed, because this run measured them again.

### Added

- `coticula demo` and `coticula grade` (`python -m coticula` works the same).
- Surrogate protocol: batch `predict(u0, coeff, n_steps=...)`.
  Example: `examples/persist_ic.py`.
- Scorecard written as JSON, Markdown, HTML, and PNG plots.
- `scripts/render_results_table.py` prints the README harness table from
  `reports/latest.json`.
- CI runs `coticula demo` on CPU after the existing pytest and smoke trains.
- Tests for the adapter and the CLI, including the shipped Burgers FNO
  fail-closed result.

### Measured, not a new model

- On the Burgers holdout, shipped FNO is inside the Predict cut and outside
  the Conserve cut (`untrusted`). Shipped PINO has a higher Predict rel-L2
  and a lower residual, and is still `untrusted`.
- `persist_ic` on that holdout is inside both cuts, so the flag is `trusted`,
  while its counterfactual trajectory change is 0. The README states that
  limitation next to the numbers.

## [0.2.0] — 2026-09-12

Research increment: Exam 4, diagnostics, paper-style writeup, Coticula
branding. **No fabricated SOTA.** IID Exam 1–3 and OOD cells are the
measured Week 3 run; Exam 4 and diagnostics were re-measured on this
revision.

### Exam 4

- Item set expanded **36 → 48** with clearer prompts (Coticula, not VU-Bench).
- Structured **rule-based** facet templates + measured Exam 1–3 observations.
- **Metric-dump** negative control (numbers only).
- LLM judge still env-gated (`COTICULA_LLM_JUDGE` / `VU_LLM_JUDGE`) and
  never invents scores.

### Eval rigor

- Residual-vs-L2 scatter from measured IID + OOD points
  (`metrics/diagnostics.py`, `reports/diagnostics.json`).
- Classical horizon residual split (early vs late half of the stored grid).
- Classical counterfactual coefficient-sensitivity sweep.

### Docs / brand

- `docs/WRITEUP.md` tightened toward a short paper draft.
- README points at `docs/ONEPAGER.md` and v0.2 status.
- Product name is **Coticula** (historical Vermithor / VU-Bench mentions stay
  in this changelog).

## [0.1.0] — 2026-09-12

Cite-ready packaging of Weeks 1–4. Tag `v0.1.0` is intended after this
changelog lands on `main` (this increment does not create the tag).

Merged PRs: [#1](https://github.com/shehbaz0101/coticula/pull/1) (Week 2),
[#2](https://github.com/shehbaz0101/coticula/pull/2) (Week 3),
[#3](https://github.com/shehbaz0101/coticula/pull/3) (Week 4).
Week 1 is the initial commit on this repo.

### Week 1 — classical harness

- Classical finite-difference label generators for periodic Burgers 1D and
  Dirichlet heat 2D (`scripts/generate_labels.py`).
- Exam APIs: Predict, Conserve, Counterfactual, Explain
  (`metrics/{predict,conserve,counterfactual,explain}.py`).
- Eval harness writes `reports/latest.{json,md}` with **measured** classical
  numbers only. Missing FNO / PINO / LLM stay `not_trained`.
- Smoke tests for metrics and report keys.

### Week 2 — FNO / PINO baselines ([#1](https://github.com/shehbaz0101/coticula/pull/1))

- Laptop-scale Fourier neural operator (Li et al., ICLR 2021) for Burgers and
  heat; time as output channels. See `docs/DESIGN.md` for widths / modes.
- PINO-style path: same trunk plus a light PDE residual term
  (`L = L_data + λ_pde L_pde`; default `λ_pde = 1e-3`).
- `scripts/train_fno.py` / `scripts/train_pino.py` (`--smoke` for CPU CI).
- `run_eval` scores Exams 1–3 for loaded `checkpoints/*.pt`. Exam 4 stayed a
  keyword-rubric stub. **No fabricated metrics.**
- Repo branded **Coticula**; VU = Coticula Understanding Bench.

### Week 3 — OOD, trust, writeup ([#2](https://github.com/shehbaz0101/coticula/pull/2))

- OOD / transfer probes reported **separately from IID**: parameter shift,
  spatial resolution transfer, IC-family shift (`metrics/ood.py`,
  `scripts/run_ood.py`).
- Fail-closed trust flags (`metrics/trust.py`): `trusted` / `ood` /
  `untrusted` / `reference`. OOD and untrusted fields must carry
  `refuse_silent_heatmap()`.
- Failure-analysis notes compare measured L2 vs residual and at least one
  counterfactual / OOD case (copied into `docs/WRITEUP.md` from that eval).
- `docs/WRITEUP.md` arXiv-style draft filled from `reports/latest.json`.

### Week 4 — CI, Exam 4 set, pins ([#3](https://github.com/shehbaz0101/coticula/pull/3))

- GitHub Actions CPU CI (`.github/workflows/ci.yml`): `pytest -q`,
  `--smoke` trains, `verify_manifests`. Full `generate_labels` / `run_eval`
  is **not** in CI (NPZ labels are gitignored).
- Exam 4 is a **fixed 36-item** pinned set (`datasets/exam4/items_v0.json`)
  graded by expected law keywords. LLM judge stays env-gated and never
  invents scores.
- SHA pins for IID labels, OOD generators (recipe, not stored NPZ), and the
  Exam 4 item set. Refresh: `generate_labels` / `pin_datasets` /
  `verify_manifests`.
- README quickstart, Week 1–4 status, trust-flag table, WRITEUP link.

### Week 5 — citation packaging (this increment)

- `CITATION.cff` for software citation (author: Shehbaz Pathan; v0.1.0).
- This changelog, `docs/RELEASE_v0.1.0.md`, and pasteable `RELEASE_NOTES.md`.
- README Cite / Release section. **No new eval numbers.**

### Non-claims (unchanged)

- No chip-cooling / industrial multiphysics demos.
- No AU-scale or cosmology / foundation-model workloads.
- No fabricated SOTA for FNO / PINO / LLM.
- No claim that keyword rubrics equal explanation quality.
- No claim that a `trusted` flag means the model understands the PDE.

[0.3.2]: https://github.com/shehbaz0101/coticula/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/shehbaz0101/coticula/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/shehbaz0101/coticula/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/shehbaz0101/coticula/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/shehbaz0101/coticula/releases/tag/v0.1.0
