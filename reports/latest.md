# Vermithor VU-Bench v0 — latest eval

_Generated (UTC): 2026-09-11T23:46:14.734192+00:00_

**VU = Vermithor Understanding Bench** (not the `uv` Python packager).
Project: **Vermithor**. Repo: https://github.com/shehbaz0101/vermithor

## Exam 1 — Predict

| Baseline | Burgers rel-L2 mean | Heat2D rel-L2 mean | NMSE (B / H) | Status |
|---|---:|---:|---:|---|
| classical_fd | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 / 0.000000e+00 | ok |
| FNO | 1.573143e-01 | 3.221335e-01 | 2.400286e-02 / 1.203401e-01 | ok (burgers=ok, heat2d=ok) |
| PINO | 1.880157e-01 | — | 3.432543e-02 / — | ok (burgers=ok, heat2d=not_trained) |
| LLM | — | — | — | not_trained |

## Exam 2 — Conserve

- classical Burgers residual_rel_l2: `1.438256e-02`
- classical Burgers energy max |rel drift|: `4.794753e-01`
- classical Heat residual_rel_l2: `3.033189e-02`
- classical Heat energy max |rel drift|: `9.895394e-01`
- FNO Burgers residual_rel_l2: `4.848762e+00`; energy max |rel drift|: `5.392740e-01`
- FNO Heat residual_rel_l2: `1.303385e+01`; energy max |rel drift|: `9.646470e-01`
- PINO Burgers residual_rel_l2: `1.732850e+00`; energy max |rel drift|: `4.406458e-01`
- PINO Heat: `not_trained`

## Exam 3 — Counterfactual

- classical Burgers ν→2ν traj Δ rel-L2: `1.189840e-01` (ν=0.03366→0.06733)
- classical Heat α→1.5α traj Δ rel-L2: `2.080764e-01` (α=0.1268→0.1902)
- FNO Burgers vs classical at θ′ rel-L2: `1.793023e-01`
- FNO Heat vs classical at θ′ rel-L2: `9.553082e-01`
- PINO Burgers vs classical at θ′ rel-L2: `2.397134e-01`
- PINO Heat: `not_trained`

## Exam 4 — Explain

- Keyword rubric overall: `0.5357` (status: stub_keyword_rubric)
- LLM: `not_trained` — No API key (VU_LLM_API_KEY / OPENAI_API_KEY); keyword rubric only. No fabricated LLM scores.

## Notes

- Classical numbers are from re-solving stored ICs (self-consistency / label check).
- FNO / PINO numbers appear only when a checkpoint loads; otherwise `not_trained`.
- No fabricated SOTA. Exam 4 is a keyword stub, not a trained judge.
- VU = Vermithor Understanding Bench. Non-goals: no chip cooling, no AU-scale FM.
