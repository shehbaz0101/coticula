# VU-Bench v0 Design

**VU = Vermithor Understanding.** This harness belongs to the Vermithor
project. It is not related to the `uv` Python packager.

## Four exams

| Exam | Question | Signal |
|------|----------|--------|
| **Predict** | Can the model match future fields? | Relative L2 + NMSE vs classical labels |
| **Conserve** | Does the rollout respect the PDE / energy? | FD residual relative L2; discrete energy drift |
| **Counterfactual** | What if a coefficient changes? | Classical: re-solve IC at ν′ / α′. FNO/PINO: predict at ν′ / α′ and **grade vs the classical solver** at the new coefficient |
| **Explain** | Can a rationale cite the right physics? | Keyword rubric stub (LLM not scored; no fabricated judge) |

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

### Week 2 (this increment)

1. A tiny PyTorch FNO trains on Burgers and/or heat2d labels and writes a checkpoint.
2. A PINO-style path adds a PDE residual term; loss terms are documented below.
3. Exams 1–3 report **measured** FNO (and classical) numbers when a checkpoint
   loads. Missing checkpoints stay `not_trained`.
4. Exam 4 remains a keyword stub. The LLM hook does not emit a fake score.
5. `scripts/run_eval.py` writes JSON+MD with only measured numbers.
6. `pytest` passes on CPU (no GPU required). FNO tests `importorskip("torch")`.

## 3-week plan (summary)

| Week | Focus |
|------|--------|
| **1** | Classical labels, metrics APIs, eval harness, smoke tests |
| **2** (this) | Train small FNO + PINO residual; wire Predict / Conserve / Counterfactual |
| **3** | Counterfactual suites at scale; LLM/rationale judge beyond keyword stub; freeze v0 report format |

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

## Reporting rules

- Every numeric field in `reports/latest.*` is computed in that eval run.
- `not_trained` means the checkpoint (or torch, or LLM judge) was absent — not
  a placeholder accuracy.
- Exam 4 keyword coverage is a stub (`status: stub_keyword_rubric`).

## Non-goals (unchanged)

- No chip-cooling / industrial multiphysics demos
- No AU-scale (astronomical-unit) or cosmology / foundation-model workloads
- No fabricated SOTA
- No claim that keyword rubrics equal true explanation quality
