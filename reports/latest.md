# VU-Bench v0 — latest eval

_Generated (UTC): 2026-09-11T23:28:47.071103+00:00_

## Exam 1 — Predict

| Baseline | Burgers rel-L2 mean | Heat2D rel-L2 mean | NMSE (B / H) | Status |
|---|---:|---:|---:|---|
| classical_fd | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 / 0.000000e+00 | ok |
| FNO | — | — | — | not_trained |
| PINO | — | — | — | not_trained |
| LLM | — | — | — | not_trained |

## Exam 2 — Conserve

- Burgers residual_rel_l2: `1.438256e-02`
- Burgers energy max |rel drift|: `4.794753e-01`
- Heat residual_rel_l2: `3.033189e-02`
- Heat energy max |rel drift|: `9.895394e-01`
- FNO/PINO: `not_trained`

## Exam 3 — Counterfactual

- Burgers ν→2ν traj Δ rel-L2: `1.189840e-01` (ν=0.03366→0.06733)
- Heat α→1.5α traj Δ rel-L2: `2.080764e-01` (α=0.1268→0.1902)
- FNO/PINO: `not_trained`

## Exam 4 — Explain

- Keyword rubric overall: `0.5357` (status: stub_keyword_rubric)
- LLM: `not_trained`

## Notes

- Classical numbers are from re-solving stored ICs (self-consistency / label check).
- FNO, PINO, LLM are Week-1 stubs — marked `not_trained`; no fabricated SOTA.
