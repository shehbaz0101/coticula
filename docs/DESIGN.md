# Coticula v0 Design

**VU = Coticula Understanding Bench.** Project branding: **Coticula**.
This harness is not related to the `uv` Python packager.

Public repo: https://github.com/shehbaz0101/coticula

## Four exams

| Exam | Question | Signal |
|------|----------|--------|
| **Predict** | Can the model match future fields? | Relative L2 + NMSE vs classical labels |
| **Conserve** | Does the rollout respect the PDE / energy? | FD residual relative L2; discrete energy drift |
| **Counterfactual** | What if a coefficient changes? | Classical: re-solve IC at ν′ / α′. FNO/PINO: predict at ν′ / α′ and **grade vs the classical solver** at the new coefficient |
| **Explain** | Can a rationale cite the right physics? | Pinned Exam 4 item set (36); keyword coverage vs expected law keywords. LLM hook env-gated, never fabricated |

Classical FD solvers both *generate labels* and act as the first baseline.
Self-consistency (re-solve IC → compare to stored traj) should yield near-zero
predict error; residual/energy audits are diagnostic of the discrete scheme.

## Success criteria

### Week 1 (shipped)

1. `python -m scripts.generate_labels` writes Burgers + heat2d NPZ + SHA256 manifest.
2. `python -m scripts.run_eval` writes `reports/latest.{json,md}` with all four
   exam sections and **real** classical metrics (no placeholders / invented SOTA).
3. FNO / PINO / LLM sections explicitly `not_trained` when untrained.
4. `pytest` smoke tests pass (metrics + import stubs + report keys).

### Week 2 (shipped)

1. A tiny PyTorch FNO trains on Burgers and/or heat2d labels and writes a checkpoint.
2. A PINO-style path adds a PDE residual term; loss terms are documented below.
3. Exams 1–3 report **measured** FNO (and classical) numbers when a checkpoint
   loads. Missing checkpoints stay `not_trained`.
4. Exam 4 remains a keyword stub. The LLM hook does not emit a fake score.
5. `scripts/run_eval.py` writes JSON+MD with only measured numbers.
6. `pytest` passes on CPU (no GPU required). FNO tests `importorskip("torch")`.

### Week 3 (shipped)

1. OOD / transfer probes for Burgers and heat2d, **reported separately from IID**:
   param shift (ν / α outside the label range), spatial resolution transfer
   (FNO spectral conv), optional IC-family shift.
2. Fail-closed trust flags: `ood` or `untrusted` when a case leaves train
   support or the residual / Predict L2 exceeds documented thresholds.
   OOD / untrusted fields must not be shown as silent pretty heatmaps
   (`metrics.trust.refuse_silent_heatmap`).
3. Failure-analysis notes compare measured L2 vs residual (FNO vs PINO) and
   at least one counterfactual / OOD case. No fabricated metrics.
4. `scripts/run_eval.py` (and `scripts/run_ood.py`) write only measured IID + OOD
   numbers into `reports/latest.{json,md}`.
5. `docs/WRITEUP.md` is an arXiv-style draft filled from that run.
6. `pytest` covers OOD helpers on CPU.

### Week 4 (this increment)

1. GitHub Actions CI (`.github/workflows/ci.yml`) on push/PR: `ubuntu-latest`,
   `pytest -q`, and CPU `--smoke` trains (`train_fno --smoke`,
   `train_pino --smoke --pde burgers`). Full label generation / `run_eval` is
   **not** in CI (NPZ gitignored; heavier than a PR check) — documented in README.
2. Exam 4 is a **fixed item set** (≥15, v0 = 36) with expected law keywords,
   graded in `metrics/explain.py` and written by `run_eval`. Optional LLM judge
   stays env-gated (`VU_LLM_JUDGE`) and **never invents scores**.
3. SHA manifests cover labels **and** OOD generators (pinned probe plan /
   knobs / seeds). Regenerate path: `generate_labels` / `pin_datasets` /
   `verify_manifests`.
4. README: quickstart, Week 1–4 status, how to read trust flags, WRITEUP link,
   optional results table from `reports/latest.md`.
5. `pytest` covers Exam 4 + pins on CPU.

Later: a real LLM/rationale judge (still must not fabricate); larger
counterfactual suites.

## 4-week plan (summary)

| Week | Focus |
|------|--------|
| **1** | Classical labels, metrics APIs, eval harness, smoke tests |
| **2** | Train small FNO + PINO residual; wire Predict / Conserve / Counterfactual |
| **3** | OOD / transfer probes, fail-closed trust, failure analysis, writeup draft |
| **4** (this) | CI, Exam 4 item set, dataset pins, README polish |

## Equations

- **Burgers 1D (periodic):** \(u_t + u u_x = \nu u_{xx}\)
- **Heat 2D (Dirichlet):** \(u_t = \alpha (u_{xx} + u_{yy})\)

## FNO baseline (Week 2)

Citation: Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart, Anandkumar.
*Fourier Neural Operator for Parametric Partial Differential Equations*, ICLR 2021
(https://arxiv.org/abs/2010.08895).

Laptop-scale defaults (not a SOTA claim):

| PDE | Arch | Default width / modes / layers | Input channels | Output |
|-----|------|--------------------------------|----------------|--------|
| Burgers | 1D FNO, time as output channels | 16 / 8 / 3 | \(u_0\), \(\nu\), \(x\)-grid | \(u(t,x)\) on the label grid |
| Heat | 2D FNO, time as output channels | 12 / 6 / 2 | \(u_0\), \(\alpha\), \(x,y\)-grids | \(u(t,x,y)\) on the label grid |

Inductive biases (documented, not hidden):

- \(t=0\) of the prediction is locked to the given IC (the operator predicts the evolution).
- Heat Dirichlet walls are zeroed after the projection.

## PINO loss terms (Week 2)

Citation: Li et al., *Physics-Informed Neural Operator for Learning Partial
Differential Equations*, arXiv:2111.03794.

```
L = L_data + λ_pde * L_pde + λ_ic * L_ic
```

| Term | Definition |
|------|------------|
| **L_data** | \(\mathrm{MSE}(\hat u,\, u_{\mathrm{label}})\) over the full trajectory (supervised fit to classical FD labels) |
| **L_pde** (Burgers) | Mean squared FD residual \(R = \hat u_t + \partial_x(\hat u^2)/2 - \nu \hat u_{xx}\) (periodic, midpoint in time — same stencil as `metrics.conserve`) |
| **L_pde** (Heat) | Mean squared FD residual \(R = \hat u_t - \alpha \nabla^2 \hat u\) on Dirichlet interior (same stencil as `metrics.conserve`) |
| **L_ic** | \(\mathrm{MSE}(\hat u(t=0),\, u_0)\). Default \(\lambda_{\mathrm{ic}}=0\) because the operators lock the IC |

FNO-only training is \(\lambda_{\mathrm{pde}}=\lambda_{\mathrm{ic}}=0\).
Default PINO weights: \(\lambda_{\mathrm{pde}}=10^{-3}\), \(\lambda_{\mathrm{ic}}=0\).
The small \(\lambda_{\mathrm{pde}}\) is intentional: raw FD residuals are
O(1)–O(100) while data MSE is O(10^{-2}), so \(\lambda=0.1\) drowns \(L_{\mathrm{data}}\).

Code: `baselines/pino/losses.py`. Entry points: `scripts/train_fno.py --pino`
or `scripts/train_pino.py`.

## Train support (v0 labels)

| PDE | Param range | Grid | IC family |
|-----|-------------|------|-----------|
| Burgers | ν ∈ [0.005, 0.05] | nx=64, nt=80 | `fourier_modes` |
| Heat-2D | α ∈ [0.05, 0.2] | n=32, nt=40 | `gaussian_bump` |

OOD knobs (isolated; see `metrics/ood.py`): ν ∈ {0.001, 0.15}, α ∈ {0.01, 0.50},
Burgers nx ∈ {32, 128}, heat n ∈ {16, 48}, Burgers ICs `gaussian_pulse` /
`tanh_front`, heat ICs `two_bump` / `sinusoid`. Time is **not** transferred:
FNO/PINO emit a fixed number of time channels.

## Fail-closed trust (`metrics/trust.py`)

| Status | Meaning |
|--------|---------|
| `trusted` | In train support **and** residual_rel_l2 ≤ 1.0 **and** Predict rel-L2 ≤ 0.5 |
| `ood` | Leaves train support (param / resolution / IC family); residual still below threshold |
| `untrusted` | Residual or Predict L2 exceeds a threshold (wins over `ood` if both apply) |
| `reference` | Classical FD labeler row (not a transfer claim) |

Thresholds are documented defaults, not tuned after seeing scores. A field
with `ood` or `untrusted` must carry `refuse_silent_heatmap()` — no silent
pretty heatmaps.

## Reporting rules

- Every numeric field in `reports/latest.*` is computed in that eval run.
- `not_trained` means the checkpoint (or torch, or LLM judge) was absent — not
  a placeholder accuracy.
- Exam 4 grades a pinned item set (`status: keyword_rubric`). The legacy
  single-text helper remains `stub_keyword_rubric`. LLM judge is `not_trained`
  with `score: null` until a real judge is wired.
- OOD metrics live under `ood` (and the markdown OOD section), never mixed
  into Exam 1–3 IID tables.
- Trust flags travel with the measured numbers; they are not a substitute
  for the numbers.

## Non-goals (unchanged)

- No chip-cooling / industrial multiphysics demos
- No AU-scale (astronomical-unit) or cosmology / foundation-model workloads
- No fabricated SOTA
- No claim that keyword rubrics equal true explanation quality
