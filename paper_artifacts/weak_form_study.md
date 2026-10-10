# Weak-form noise-calibrated residual study

Experimental backbone for **Small Errors, Large Residuals**. Every number below is written by `scripts/reproduce_paper.py` from the frozen protocol. The default Conserve metric, the trust rule, and the README are unchanged.

Protocol SHA-256: `8b9110746356564fb3cf1ac6e8cd1b6cd223c4b022b366e99c93c85c5520d477`

## Definitions

Strong residual ρ_s is the per-step finite-difference residual in `metrics/conserve.py`: mean over time of ||R||_2 / ||u_mid||_2. Heat uses zero Dirichlet boundaries and drops the boundary nodes.

Weak residual ρ_w = sqrt(sum_m r_m^2) / (||u_ref||_2 sqrt(tr G)), with G the Gram matrix of ψ_m = L* φ_m. Test functions are compact polynomial bumps in time times Fourier modes (Burgers) or Dirichlet sine modes (heat). Bandwidth K_φ and the bump geometry are in `paper_artifacts/protocol.json`. The denominator uses the reference field, as in the draft.

Smooth errors are time-varying trigonometric polynomials of degree 2 on spatial modes 1..4. They are not frozen in time. White heat noise is i.i.d. on the interior and exactly zero on the Dirichlet boundary, then rescaled.

Scaled score S(M) = 100 (M_0 - M) / (M_0 - M_ref), not clipped. M_ref is 0 for relative L2 and the solver residual for residual metrics. M_0 is the best (lowest) trivial baseline. The zero field is excluded from M_0 for residual metrics. S is undefined when M_0 ≤ M_ref.

The legacy untrusted flag is strong residual > 1, the same constant as `metrics.trust.RESIDUAL_UNTRUSTED`. It survives the weak form only when ρ_w is strictly above the white-noise weak envelope at the model's own relative L2. Fields whose relative L2 lies below the noise grid do not receive that call.

## Datasets and budget

| PDE | grid | h | dt | Nt | train/val/test | coefficient |
|---|---|---|---|---|---|---|
| Burgers | nx=32 | 0.1963 | 0.0625 | 16 | 12/4/4 | ν in [0.005, 0.05] |
| Heat | n=16 Dirichlet | 0.06667 | 0.04167 | 12 | 8/4/4 | α in [0.05, 0.2] |

Equal tuning budget: 2 trials per architecture, 40 epochs on the validation split (seed 7), then 120 epochs from scratch for seeds [0, 1, 2, 3, 4]. Optimizer Adam, lr=0.001, weight decay=0.0001, batch size 4, data MSE only. No trial sees the test split. DeepONet trunks use fixed Fourier features and a fixed coefficient scale (0.05 for ν, 0.2 for α).

## Noise injection

Mean ± sample standard deviation across noise seeds of the per-seed trajectory mean.

| PDE | error | ε | realized rel-L2 | keff·h | ρ_s | ρ_w | ρ_s/ρ_w |
|---|---|---|---|---|---|---|---|
| burgers | white | 0.001 | 0.001 ± 2.87e-19 | 1.56459 ± 0.0049 | 0.0635845 ± 7.46e-05 | 0.000257793 ± 1.99e-06 | 246.649 |
| burgers | smooth | 0.001 | 0.001 ± 3.76e-19 | 0.585321 ± 0.00948 | 0.0597919 ± 4.19e-05 | 0.000262472 ± 2.37e-06 | 227.802 |
| burgers | white | 0.01 | 0.01 ± 8.67e-19 | 1.56125 ± 0.00375 | 0.233954 ± 0.00102 | 0.000512212 ± 2.61e-05 | 456.752 |
| burgers | smooth | 0.01 | 0.01 ± 8.67e-19 | 0.583694 ± 0.00642 | 0.0998502 ± 0.00105 | 0.000660949 ± 3e-05 | 151.071 |
| burgers | white | 0.05 | 0.05 ± 4.91e-18 | 1.56434 ± 0.00474 | 1.12934 ± 0.00443 | 0.00211827 ± 4.86e-05 | 533.142 |
| burgers | smooth | 0.05 | 0.05 ± 3.47e-18 | 0.582182 ± 0.00809 | 0.401232 ± 0.0092 | 0.00299811 ± 0.000121 | 133.828 |
| burgers | white | 0.1 | 0.1 ± 6.94e-18 | 1.56635 ± 0.00323 | 2.24554 ± 0.00926 | 0.00434133 ± 0.000201 | 517.247 |
| burgers | smooth | 0.1 | 0.1 ± 6.94e-18 | 0.58301 ± 0.00212 | 0.786808 ± 0.0118 | 0.00594486 ± 0.000194 | 132.351 |
| burgers | white | 0.2 | 0.2 ± 2.4e-17 | 1.56629 ± 0.00346 | 4.48315 ± 0.0205 | 0.00869414 ± 0.00047 | 515.652 |
| burgers | smooth | 0.2 | 0.2 ± 2.4e-17 | 0.591809 ± 0.00733 | 1.53862 ± 0.0537 | 0.0111788 ± 0.000637 | 137.637 |
| burgers | white | 0.5 | 0.5 ± 4.81e-17 | 1.56402 ± 0.00705 | 10.7505 ± 0.0476 | 0.0213141 ± 0.00121 | 504.382 |
| burgers | smooth | 0.5 | 0.5 ± 5.55e-17 | 0.584831 ± 0.00739 | 3.74152 ± 0.0905 | 0.030183 ± 0.0011 | 123.961 |
| burgers | reference | 0 | 0 | — | 0.0592059 | 0.000255196 | — |
| heat | white | 0.001 | 0.001 ± 2.42e-19 | 2.10884 ± 0.00278 | 0.236256 ± 0.00034 | 0.000457815 ± 1.28e-06 | 516.051 |
| heat | smooth | 0.001 | 0.001 ± 1.08e-19 | 0.843842 ± 0.00968 | 0.138134 ± 0.000737 | 0.000459799 ± 2.01e-06 | 300.422 |
| heat | white | 0.01 | 0.01 ± 8.67e-19 | 2.10768 ± 0.00333 | 1.55046 ± 0.00415 | 0.000508959 ± 9.54e-06 | 3046.33 |
| heat | smooth | 0.01 | 0.01 ± 1.5e-18 | 0.842541 ± 0.0067 | 0.4532 ± 0.00766 | 0.000715314 ± 1.32e-05 | 633.568 |
| heat | white | 0.05 | 0.05 ± 0 | 2.10555 ± 0.00204 | 7.48255 ± 0.0143 | 0.00120299 ± 5.58e-05 | 6219.94 |
| heat | smooth | 0.05 | 0.05 ± 3.47e-18 | 0.844054 ± 0.00916 | 1.94203 ± 0.038 | 0.00284287 ± 8.21e-05 | 683.124 |
| heat | white | 0.1 | 0.1 ± 6.94e-18 | 2.10889 ± 0.00301 | 14.8547 ± 0.0876 | 0.00215088 ± 6.13e-05 | 6906.34 |
| heat | smooth | 0.1 | 0.1 ± 6.94e-18 | 0.843649 ± 0.00493 | 3.73564 ± 0.0362 | 0.00538827 ± 0.000138 | 693.29 |
| heat | white | 0.2 | 0.2 ± 2.78e-17 | 2.10702 ± 0.00325 | 28.7363 ± 0.0497 | 0.00435585 ± 0.000157 | 6597.18 |
| heat | smooth | 0.2 | 0.2 ± 0 | 0.840505 ± 0.00515 | 7.12477 ± 0.0962 | 0.0107857 ± 0.000428 | 660.578 |
| heat | white | 0.5 | 0.5 ± 4.81e-17 | 2.10736 ± 0.00365 | 61.7185 ± 0.42 | 0.0105155 ± 0.000534 | 5869.29 |
| heat | smooth | 0.5 | 0.5 ± 4.81e-17 | 0.842302 ± 0.00764 | 14.7077 ± 0.379 | 0.0277436 ± 0.00109 | 530.128 |
| heat | reference | 0 | 0 | — | 0.118298 | 0.00045577 | — |

## Resolution and time-step sweep

Fixed ε = 0.01. Theory for heat white noise is C_h ε from the second-moment formula. Theory for Burgers is the same per-step normalization applied to the pointwise Burgers second-moment formula, and it ignores the discretization floor.

| PDE | sweep | N | dt | r | ρ_s white | theory | ρ_s smooth | ρ_w white | ρ_w smooth |
|---|---|---|---|---|---|---|---|---|---|
| burgers | spatial | 32 | 0.05 | 0.044 | 0.283174 | 0.284404 | 0.0891637 | 0.000432156 | 0.000627587 |
| burgers | spatial | 64 | 0.05 | 0.131 | 0.294682 | 0.288021 | 0.110964 | 0.000320429 | 0.000664955 |
| burgers | spatial | 128 | 0.05 | 0.546 | 0.457756 | 0.363354 | 0.212345 | 0.000222255 | 0.000542373 |
| burgers | temporal | 64 | 0.1 | 0.194 | 0.328921 | 0.148829 | 0.299257 | 0.00147329 | 0.00167077 |
| burgers | temporal | 64 | 0.05 | 0.131 | 0.294682 | 0.288021 | 0.110964 | 0.000320429 | 0.000664955 |
| burgers | temporal | 64 | 0.025 | 0.0925 | 0.569629 | 0.572952 | 0.0882712 | 0.000199483 | 0.00060585 |
| heat | spatial | 16 | 0.04167 | 0.82 | 0.874681 | 0.710728 | 0.264707 | 0.000595504 | 0.000778563 |
| heat | spatial | 32 | 0.04167 | 4.31 | 4.76409 | 3.29145 | 0.49332 | 0.000489177 | 0.000709435 |
| heat | spatial | 48 | 0.04167 | 12.1 | 14.4238 | 9.20076 | 0.643306 | 0.0004465 | 0.000723155 |
| heat | temporal | 32 | 0.0625 | 9.43 | 9.05127 | 4.77476 | 0.989408 | 0.0161292 | 0.016164 |
| heat | temporal | 32 | 0.04167 | 4.31 | 4.76409 | 3.29145 | 0.49332 | 0.000489177 | 0.000709435 |
| heat | temporal | 32 | 0.02083 | 1.96 | 3.78521 | 3.05423 | 0.280749 | 7.92063e-05 | 0.000550507 |

## Symbol check

Heat synthetic i.i.d. check, mean measured/formula = 1.00111. Log-log fit of measured vs formula: slope 1.00001, R² 1.

Burgers synthetic check ratio = 0.992316 (empirical 0.0511349, formula 0.0515308).

Static heat error, ||R[e]||/||e|| over α keff(e)² = 1.

## Models

| PDE | model | rel-L2 | S(rel-L2) | ρ_s | S(ρ_s) | ρ_w | S(ρ_w) |
|---|---|---|---|---|---|---|---|
| burgers | zeros | 1 ± 0 | -169.43 | 0 ± 0 | — | 0 ± 0 | — |
| burgers | training_mean | 0.996807 ± 0 | -168.57 | 1.01465 ± 0 | -38.4233 | 0.000842696 ± 0 | 0 |
| burgers | frozen_initial | 0.371154 ± 0 | 0 | 0.747975 ± 0 | 0 | 0.00484133 ± 0 | -709.109 |
| burgers | reference | 0 ± 0 | 100 | 0.0539355 ± 0 | 100 | 0.000278801 ± 0 | 100 |
| burgers | fno | 0.0754256 ± 0.00103 | 79.6781 | 0.446257 ± 0.0123 | 43.4728 | 0.00162939 ± 0.000235 | -139.511 |
| burgers | unet | 0.108056 ± 0.00558 | 70.8865 | 0.993192 ± 0.114 | -35.3317 | 0.00341545 ± 0.00125 | -456.247 |
| burgers | deeponet | 0.374961 ± 0.00816 | -1.02565 | 0.895786 ± 0.0248 | -21.2971 | 0.00245084 ± 0.000293 | -285.184 |
| heat | zeros | 1 ± 0 | -100.345 | 0 ± 0 | — | 0 ± 0 | — |
| heat | training_mean | 0.499138 ± 0 | 0 | 0.961087 ± 0 | 0 | 0.00273929 ± 0 | 0 |
| heat | frozen_initial | 1.38554 ± 0 | -177.586 | 11.5265 ± 0 | -1218.56 | 0.051183 ± 0 | -2135.02 |
| heat | reference | 0 ± 0 | 100 | 0.0940405 ± 0 | 100 | 0.000470282 ± 0 | 100 |
| heat | fno | 0.181637 ± 0.00295 | 63.6098 | 4.90286 ± 0.268 | -454.621 | 0.0053979 ± 0.000487 | -117.17 |
| heat | unet | 0.140632 ± 0.00948 | 71.8251 | 4.49754 ± 0.291 | -407.874 | 0.00512489 ± 0.000713 | -105.138 |
| heat | deeponet | 0.303569 ± 0.0218 | 39.1814 | 8.58461 ± 1.52 | -879.251 | 0.00814971 ± 0.000464 | -238.448 |

S is undefined (—) when the best trivial baseline does no worse than the reference. Zeros are excluded from the residual anchor and their residual score is blank.

### Coefficient response

γ = 1.5. S_coef = ||G(γc) - G(c)|| / ||G*(γc) - G*(c)||. Cosine is undefined when either difference vanishes.

| PDE | model | S_coef | cosine | rel-L2 at γc |
|---|---|---|---|---|
| burgers | fno | 0.0355132 ± 0.0255 | -0.138706 ± 0.262 | 0.0775992 ± 0.00201 |
| burgers | unet | 0.118804 ± 0.0493 | 0.30449 ± 0.0625 | 0.101028 ± 0.00486 |
| burgers | deeponet | 1.32791 ± 0.124 | -0.0560803 ± 0.067 | 0.369994 ± 0.0146 |
| heat | fno | 0.0799616 ± 0.0478 | 0.165469 ± 0.695 | 0.273762 ± 0.0127 |
| heat | unet | 0.36433 ± 0.0749 | 0.700468 ± 0.0506 | 0.221303 ± 0.0148 |
| heat | deeponet | 0.845619 ± 0.087 | 0.709866 ± 0.0387 | 0.356007 ± 0.0347 |

### Associations

burgers: ρ_s vs keff²·rel-L2 slope 0.0716763, intercept 0.484484, R² 0.433241 (n=60). Kendall τ(ρ_s, ρ_w) = 0.420339 (p=2.08e-06, n=60).
heat: ρ_s vs keff²·rel-L2 slope 0.225563, intercept -1.94324, R² 0.756604 (n=60). Kendall τ(ρ_s, ρ_w) = 0.59548 (p=1.79e-11, n=60).
Heat ρ_s vs α keff²·rel-L2 slope 1.31205, R² 0.918348.

## Coefficient mismatch on exact labels

| PDE | scale | ρ_s | ρ_w |
|---|---|---|---|
| burgers | 0 | 0.179479 ± 0.105 | 0.000770174 ± 0.000516 |
| burgers | 0.5 | 0.0891532 ± 0.0412 | 0.000380361 ± 0.00022 |
| burgers | 1 | 0.0592059 ± 0.0141 | 0.000255196 ± 8.81e-05 |
| burgers | 2 | 0.249727 ± 0.113 | 0.00109465 ± 0.000511 |
| heat | 0 | 3.92039 ± 0.895 | 0.0124034 ± 0.00198 |
| heat | 0.5 | 1.96553 ± 0.425 | 0.00615261 ± 0.0009 |
| heat | 1 | 0.118298 ± 0.0803 | 0.00045577 ± 7.49e-05 |
| heat | 2 | 3.92246 ± 1.02 | 0.00945917 ± 0.000755 |

## Shipped checkpoints

These are the v0 checkpoints on the original grids, not the five-seed study.

| PDE | model | status | rel-L2 | ρ_s | ρ_w | legacy untrusted | weak survival |
|---|---|---|---|---|---|---|---|
| burgers | fno | ok | 0.157314 | 4.84876 | 0.00373441 | true | true |
| burgers | frozen_initial | ok | 0.377195 | 0.809169 | 0.00526196 | false | true |
| burgers | zeros | ok | 1 | 0 | 0 | false | false |
| burgers | reference | ok | 0 | 0.0131944 | 5.46756e-05 | false | — |
| burgers | pino | ok | 0.188016 | 1.73285 | 0.00482244 | true | true |
| heat | fno | ok | 0.322134 | 13.0339 | 0.00897895 | true | true |
| heat | frozen_initial | ok | 1.4243 | 12.8013 | 0.0621702 | true | true |
| heat | zeros | ok | 1 | 0 | 0 | false | false |
| heat | reference | ok | 0 | 0.0298339 | 2.27398e-05 | false | — |
| heat | pino | not_trained | — | — | — | — | — |

burgers FNO legacy untrusted=true, weak-form survival=true (strong=4.84876, weak=0.00373441, rel-L2=0.157314) heat FNO legacy untrusted=true, weak-form survival=true (strong=13.0339, weak=0.00897895, rel-L2=0.322134)

## Bandwidth ablation

| PDE | K_φ | ρ_w white | ρ_w smooth | ρ_w frozen | ρ_w clean |
|---|---|---|---|---|---|
| burgers | 1 | 0.000426538 | 0.000482177 | 0.00181292 | 4.84754e-05 |
| heat | 1 | 0.00136742 | 0.00148402 | 0.0435134 | 0.00137791 |
| burgers | 2 | 0.000480518 | 0.000560654 | 0.00390674 | 0.000133331 |
| heat | 2 | 0.00073874 | 0.000877137 | 0.053347 | 0.000714735 |
| burgers | 3 | 0.000512842 | 0.000663111 | 0.00561369 | 0.000255196 |
| heat | 3 | 0.000514843 | 0.000733055 | 0.0555188 | 0.00045577 |
| burgers | 4 | 0.000616993 | 0.000761561 | 0.00601605 | 0.000432177 |
| heat | 4 | 0.0003839 | 0.000748095 | 0.0547468 | 0.000312488 |

## Inverse probe

Burgers viscosity from evenly spaced sensors. The FNO-conditioned inverse was not run.

| split | method | rel err in ν | failures |
|---|---|---|---|
| id | strong_form_ls | 0.510897 ± 0.109 | 4 |
| id | weak_form_ls | 0.15499 ± 0.111 | 0 |
| id | solver_in_the_loop | 0.0621494 ± 0.0179 | 0 |
| hard_ood | strong_form_ls | 0.667938 ± 0.586 | 2 |
| hard_ood | weak_form_ls | 0.6225 ± 0.769 | 1 |
| hard_ood | solver_in_the_loop | 0.286445 ± 0.131 | 0 |

## Pre-registered checks

| check | supported | statistic |
|---|---|---|
| H1_burgers_strong_white_gt_smooth | yes | 0.134104 |
| H1_heat_strong_white_gt_smooth | yes | 1.09726 |
| H2_heat_second_moment | yes | 1.00111 |
| H3_burgers_second_moment | yes | 0.992316 |
| H4_burgers_strong_grows_with_resolution | yes | 1.61652 |
| H5_burgers_weak_grows_less_than_strong | yes | 0.318149 |
| H4_heat_strong_grows_with_resolution | yes | 16.4903 |
| H5_heat_weak_grows_less_than_strong | yes | 0.0454682 |
| H6_burgers_weak_detects_zero_coefficient | yes | 3.01797 |
| H6_heat_weak_detects_zero_coefficient | yes | 27.2142 |

## Wall clock (seconds)

| stage | seconds |
|---|---|
| noise | 1.933 |
| resolution | 1.547 |
| symbol | 0.029 |
| mismatch | 0.097 |
| kphi | 0.843 |
| models | 31.168 |
| shipped | 3.185 |
| inverse | 0.050 |
| total | 38.889 |

## Limitations

- The exact spectral identity is for linear constant-coefficient operators. Burgers is checked through the white-noise second-moment formula, which matched on this run to the ratio reported above.
- ρ_w uses a finite test space. High-frequency errors are nearly invisible to it. The zero field has a zero residual and is excluded from the residual score anchor for that reason.
- Heat white noise in the injection study is zero on the boundary, so the nodewise formula is checked on a fully i.i.d. array, separately from the Dirichlet metric.
- The model study uses small grids, a two-trial search, and CPU training. DeepONet's trunk uses a fixed Fourier feature map. Results can disagree with a larger search.
- The weak-form survival rule compares a model with the white envelope at its own relative L2. A large weak residual means the error is visible to the test functions; it is not by itself a certificate of correctness.
- The inverse probe interpolates sparse sensors before the strong and weak least-squares fits. Solver-in-the-loop sees only the sensors. No learned inverse was trained.
- Shipped FNO/PINO checkpoints were trained under the original one-seed recipe. They are a rescore, not part of the five-seed budget.

