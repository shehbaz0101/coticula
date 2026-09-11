# VU-Bench v0 Design

## Four exams

| Exam | Question | Week-1 signal |
|------|----------|---------------|
| **Predict** | Can the model match future fields? | Relative L2 + NMSE vs classical labels |
| **Conserve** | Does the rollout respect the PDE / energy? | FD residual relative L2; discrete energy drift |
| **Counterfactual** | What if a coefficient changes? | Re-solve from same IC with ν′ / α′; traj Δ |
| **Explain** | Can a rationale cite the right physics? | Keyword rubric stub (LLM not wired) |

Classical FD solvers both *generate labels* and act as the first baseline.
Self-consistency (re-solve IC → compare to stored traj) should yield near-zero
predict error; residual/energy audits are diagnostic of the discrete scheme.

## Success criteria (Week 1)

1. `python -m scripts.generate_labels` writes Burgers + heat2d NPZ + SHA256 manifest.
2. `python -m scripts.run_eval` writes `reports/latest.{json,md}` with all four
   exam sections and **real** classical metrics (no placeholders / invented SOTA).
3. FNO / PINO / LLM sections explicitly `not_trained`.
4. `pytest` smoke tests pass (metrics + import stubs + report keys).

## 3-week plan (summary)

| Week | Focus |
|------|--------|
| **1** (this scaffold) | Classical labels, metrics APIs, eval harness, smoke tests |
| **2** | Train small FNO (and optional PINO) on Burgers/heat; wire predict+conserve |
| **3** | Counterfactual suites at scale; LLM/rationale judge beyond keyword stub; freeze v0 report format |

## Equations

- **Burgers 1D (periodic):** \(u_t + u u_x = \nu u_{xx}\)
- **Heat 2D (Dirichlet):** \(u_t = \alpha (u_{xx} + u_{yy})\)

Prefer pure NumPy/SciPy for Week 1; torch optional for later baselines.
