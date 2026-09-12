# Coticula v0.2 — latest eval

_Generated (UTC): 2026-09-12T20:04:10.654852+00:00_

**Coticula** (Latin *coticula*, touchstone). Formerly Vermithor / VU-Bench. Repo: https://github.com/shehbaz0101/coticula
_IID Exam 1–3 / OOD measured at: 2026-09-12T00:18:43.378079+00:00_
_Exam 4 and diagnostics were re-measured on this revision. IID Exam 1–3 and OOD cells are copied from the prior measured run (see iid_ood_measured_at_utc). No FNO/PINO numbers were invented._

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

## Trust (fail-closed)

- Policy: `coticula-v0-fail-closed` — residual > 1.0 or Predict rel-L2 > 0.5 → `untrusted`; outside train support → `ood`.
- classical Burgers: `reference`
- classical Heat: `reference`
- FNO Burgers: `untrusted`
- FNO Heat: `untrusted`
- PINO Burgers: `untrusted`
- PINO Heat: `not_trained`
- OOD / untrusted fields must not be shown as silent pretty heatmaps (see `metrics.trust.refuse_silent_heatmap`).

## Exam 4 — Explain

- Item set: `datasets/exam4/items_v0.json` (n=48, sha256 `48323ca8ea9d8277…`)
- Gold-reference keyword coverage: `1.0000` (status: keyword_rubric) — authored gold texts vs expected law keywords, not a model score.
- Perfect / zero items: `48` / `0`
- Rule-based (facet templates + measured observations): `0.8785` (status: keyword_rubric_rule_based; perfect/zero `33` / `1`) — not an LLM score.
- Metric-dump (numbers only): `0.0903` (status: keyword_rubric_metric_dump) — rubric negative control.
- Legacy single-rationale stub overall: `0.5357` (status: stub_keyword_rubric)
- LLM: `not_trained` — LLM judge off (no COTICULA_LLM_JUDGE / VU_LLM_JUDGE / API key). Keyword rubric only. No fabricated LLM scores. (score=None)

| PDE / slice | n | mean keyword coverage |
|---|---:|---:|
| burgers1d | 14 | 1.0000 |
| harness | 20 | 1.0000 |
| heat2d | 14 | 1.0000 |

## OOD / transfer (distinct from IID)

Transfer probes distinct from IID exams 1–3. Ground truth is a fresh classical FD solve. Learned rows are fail-closed (ood / untrusted).

- Resolution note: FNO spatial conv is resolution-agnostic; time is locked as output channels (Burgers nt=80, heat nt=40). No temporal transfer is claimed.
- Trust rollup: `n=30 (untrusted=18, reference=12)`

| Probe | param / grid / IC | classical residual | FNO L2 / residual [trust] | PINO L2 / residual [trust] |
|---|---|---:|---|---|
| burgers_param_below | ν=0.001, nx=64, IC=fourier_modes | 4.186e-02 | 2.697e-01 / 5.274e+00 [untrusted] | 2.952e-01 / 1.890e+00 [untrusted] |
| burgers_param_above | ν=0.15, nx=64, IC=fourier_modes | 9.858e-03 | 2.936e-01 / 4.827e+00 [untrusted] | 3.022e-01 / 1.934e+00 [untrusted] |
| heat2d_param_below | α=0.01, n=32, IC=gaussian_bump | 1.048e-02 | 5.685e-01 / 6.353e+00 [untrusted] | not_trained |
| heat2d_param_above | α=0.5, n=32, IC=gaussian_bump | 1.879e-01 | 1.133e+00 / 4.327e+01 [untrusted] | not_trained |
| burgers_resolution_nx32 | ν=0.0275, nx=32, IC=fourier_modes | 8.786e-03 | 1.397e-01 / 4.529e+00 [untrusted] | 1.718e-01 / 1.612e+00 [untrusted] |
| burgers_resolution_nx128 | ν=0.0275, nx=128, IC=fourier_modes | 1.124e-02 | 1.347e-01 / 3.872e+00 [untrusted] | 1.582e-01 / 1.312e+00 [untrusted] |
| heat2d_resolution_n16 | α=0.125, n=16, IC=gaussian_bump | 9.738e-02 | 1.867e-01 / 6.571e+00 [untrusted] | not_trained |
| heat2d_resolution_n48 | α=0.125, n=48, IC=gaussian_bump | 4.184e-02 | 1.971e-01 / 2.103e+01 [untrusted] | not_trained |
| burgers_ic_gaussian_pulse | ν=0.0275, nx=64, IC=gaussian_pulse | 7.487e-03 | 5.200e-01 / 4.142e+00 [untrusted] | 5.695e-01 / 2.283e+00 [untrusted] |
| burgers_ic_tanh_front | ν=0.0275, nx=64, IC=tanh_front | 8.757e-03 | 2.446e-01 / 4.353e+00 [untrusted] | 2.622e-01 / 2.052e+00 [untrusted] |
| heat2d_ic_two_bump | α=0.125, n=32, IC=two_bump | 2.638e-02 | 2.420e-01 / 1.406e+01 [untrusted] | not_trained |
| heat2d_ic_sinusoid | α=0.125, n=32, IC=sinusoid | 3.853e-01 | 8.186e-01 / 2.576e+01 [untrusted] | not_trained |

## Failure analysis

Qualitative comparisons using only measured fields from this run. No fabricated metrics.

### FNO vs PINO on Burgers (IID): L2 vs residual

On IID Burgers, FNO has a lower Predict rel-L2 than PINO but a larger FD residual. A silent heatmap of the FNO field would look 'closer' to the label while Exam 2 and the fail-closed trust flag show a worse PDE violation. PINO's residual term reduces that violation at a modest L2 cost — not a SOTA claim, a measured tradeoff.

- `fno_rel_l2`: `1.573143e-01`
- `fno_residual_rel_l2`: `4.848762e+00`
- `pino_rel_l2`: `1.880157e-01`
- `pino_residual_rel_l2`: `1.732850e+00`
- `fno_trust`: `untrusted`
- `pino_trust`: `untrusted`

### FNO heat2d counterfactual insensitivity

Classical heat at α′ differs from α by a large trajectory Δ, but the FNO field barely moves when α is scaled. Exam 3 vs the classical solve at α′ is therefore large — a failure of coefficient response, not just interpolation error.

- `alpha_orig`: `1.942486e-01`
- `alpha_cf`: `2.913729e-01`
- `rel_l2_traj_delta_classical`: `2.338120e-01`
- `rel_l2_traj_delta_model`: `1.703234e-02`
- `rel_l2_vs_classical_cf`: `9.553082e-01`

### OOD Burgers ν below train range

ν=0.001 is below the v0 train range [0.005, 0.05]. Numbers below are transfer scores vs a fresh classical solve — not IID Exam 1. Trust flags are fail-closed.

- `nu`: `1.000000e-03`
- `fno_rel_l2`: `2.696545e-01`
- `fno_residual_rel_l2`: `5.274467e+00`
- `fno_trust`: `untrusted`
- `pino_rel_l2`: `2.951888e-01`
- `pino_residual_rel_l2`: `1.890248e+00`
- `pino_trust`: `untrusted`

## Diagnostics (v0.2)

v0.2 diagnostics. Residual-vs-L2 scatter re-uses measured IID + OOD cells. Horizon split and CF sensitivity are fresh classical solves on stored labels. No fabricated SOTA.

### Residual vs L2 scatter (measured IID + OOD points)

- n points: `21` (status: ok)
- Pearson residual vs L2: `0.7411755514841815`
- Spearman residual vs L2: `0.34155844155844156`
- residual median / max: `4.529278778747326` / `43.26958343971169`
- Predict L2 median / max: `0.26217477728954186` / `1.1330011201186059`
- untrusted / ood among points: `21` / `0`

| model | split | probe | rel-L2 | residual | trust |
|---|---|---|---:|---:|---|
| fno | iid | iid | 1.573e-01 | 4.849e+00 | untrusted |
| fno | iid | iid | 3.221e-01 | 1.303e+01 | untrusted |
| pino | iid | iid | 1.880e-01 | 1.733e+00 | untrusted |
| fno | ood | burgers_param_below | 2.697e-01 | 5.274e+00 | untrusted |
| pino | ood | burgers_param_below | 2.952e-01 | 1.890e+00 | untrusted |
| fno | ood | burgers_param_above | 2.936e-01 | 4.827e+00 | untrusted |
| pino | ood | burgers_param_above | 3.022e-01 | 1.934e+00 | untrusted |
| fno | ood | heat2d_param_below | 5.685e-01 | 6.353e+00 | untrusted |
| fno | ood | heat2d_param_above | 1.133e+00 | 4.327e+01 | untrusted |
| fno | ood | burgers_resolution_nx32 | 1.397e-01 | 4.529e+00 | untrusted |
| pino | ood | burgers_resolution_nx32 | 1.718e-01 | 1.612e+00 | untrusted |
| fno | ood | burgers_resolution_nx128 | 1.347e-01 | 3.872e+00 | untrusted |
| pino | ood | burgers_resolution_nx128 | 1.582e-01 | 1.312e+00 | untrusted |
| fno | ood | heat2d_resolution_n16 | 1.867e-01 | 6.571e+00 | untrusted |
| fno | ood | heat2d_resolution_n48 | 1.971e-01 | 2.103e+01 | untrusted |
| fno | ood | burgers_ic_gaussian_pulse | 5.200e-01 | 4.142e+00 | untrusted |
| pino | ood | burgers_ic_gaussian_pulse | 5.695e-01 | 2.283e+00 | untrusted |
| fno | ood | burgers_ic_tanh_front | 2.446e-01 | 4.353e+00 | untrusted |
| pino | ood | burgers_ic_tanh_front | 2.622e-01 | 2.052e+00 | untrusted |
| fno | ood | heat2d_ic_two_bump | 2.420e-01 | 1.406e+01 | untrusted |
| fno | ood | heat2d_ic_sinusoid | 8.186e-01 | 2.576e+01 | untrusted |

### Horizon residual split (classical labels)

- Burgers early / late / full: `1.870e-02` / `7.068e-03` / `1.288e-02` (late−early `-1.163e-02`)
- Heat early / late / full: `2.936e-02` / `5.381e-03` / `1.737e-02` (late−early `-2.397e-02`)

### Burgers CF sensitivity (classical)

| scale | θ′ | classical traj Δ | model traj Δ | model vs classical |
|---:|---:|---:|---:|---:|
| 0.50 | 0.01683 | 8.864e-02 | — | — |
| 1.00 | 0.03366 | 0.000e+00 | — | — |
| 1.50 | 0.05049 | 6.601e-02 | — | — |
| 2.00 | 0.06733 | 1.190e-01 | — | — |
| 3.00 | 0.101 | 2.003e-01 | — | — |

### Heat CF sensitivity (classical)

| scale | θ′ | classical traj Δ | model traj Δ | model vs classical |
|---:|---:|---:|---:|---:|
| 0.50 | 0.06339 | 3.304e-01 | — | — |
| 1.00 | 0.1268 | 0.000e+00 | — | — |
| 1.50 | 0.1902 | 2.081e-01 | — | — |
| 2.00 | 0.2535 | 3.495e-01 | — | — |

## Notes

- Classical IID numbers are from re-solving stored ICs (self-consistency / label check).
- OOD numbers are a separate section: fresh classical solves outside train support.
- FNO / PINO numbers appear only when a checkpoint loads; otherwise `not_trained`.
- No fabricated SOTA. Exam 4 is a pinned keyword item set, not a trained judge.
- Rule-based Exam 4 is structured templates + measured observations, not an LLM.
- Coticula. Non-goals: no chip cooling, no AU-scale FM.
