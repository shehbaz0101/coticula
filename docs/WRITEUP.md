# Coticula: a touchstone harness for small PDE surrogates

**Draft (v0.2).** Project branding: **Coticula** (Latin *coticula*, touchstone).
Formerly Vermithor / VU-Bench. This is not a claim of SOTA, chip-cooling
impact, or AU-scale foundation-model physics. Every numeric table below is
copied from `reports/latest.json` (IID Exam 1–3 / OOD: measured run
`2026-09-12T00:18:43.378079+00:00`; Exam 4 + diagnostics: re-measured on
this revision). Empty / `not_trained` cells mean a checkpoint or judge was
absent — not a placeholder accuracy.

Repo: https://github.com/shehbaz0101/coticula
One-pager: [ONEPAGER.md](ONEPAGER.md)

---

## Abstract

We describe Coticula, a small **understanding harness** for PDE surrogates
on two classical equations (viscous Burgers in 1D; heat in 2D). Four exams
score (1) field match, (2) discrete PDE residual and energy drift, (3)
response to a changed coefficient, and (4) a pinned explanation item set
graded by expected law keywords. Laptop-scale Fourier neural operators
(FNO) and a PINO-style residual regularizer are optional baselines.
Out-of-distribution probes (parameter, spatial resolution, initial-condition
family) are reported **separately from IID**, and a **fail-closed** trust
layer flags `ood` or `untrusted` instead of presenting a silent heatmap.
v0.2 expands Exam 4 to 48 items, grades structured (non-LLM) rule-based
and metric-dump rationales, and adds residual-versus-$L^2$ scatter plus a
classical counterfactual-sensitivity sweep. We record only measured numbers
and state the non-claims explicitly.

---

## 1. Introduction

Operator-learning papers often quote a single relative $L^2$ on a held-out
draw from the same generator. That number does not say whether the surrogate
respects the PDE, responds to a coefficient, or survives a shift the
training corpus never showed. Coticula is a *harness*, not a leaderboard:
classical finite-difference (FD) solvers both write labels and grade
counterfactuals; missing models stay `not_trained`.

The name is literal. A *coticula* is a touchstone: a stone used to test
whether gold is pure. The harness is meant to test whether a PDE surrogate
is doing the physics, not whether a heatmap looks close.

---

## 2. Related work

Coticula does **not** reproduce the scales or claimed accuracies of the
papers below. They are pointers for the methods we implement or sit next to.

- **PINNs.** Raissi, Perdikaris, Karniadakis. *Physics-informed neural
  networks*, J. Comput. Phys. 2019. https://doi.org/10.1016/j.jcp.2018.10.045
- **DeepONet.** Lu, Jin, Pang, Zhang, Karniadakis. *Learning nonlinear
  operators with DeepONet*, Nat. Mach. Intell. 2021.
  https://arxiv.org/abs/1910.03193
- **FNO.** Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart,
  Anandkumar. *Fourier Neural Operator for Parametric Partial Differential
  Equations*, ICLR 2021. https://arxiv.org/abs/2010.08895
- **PINO.** Li, Zheng, Kovachki, Liu, Liu, Yu, … Anandkumar.
  *Physics-Informed Neural Operator for Learning Partial Differential
  Equations*, arXiv:2111.03794. https://arxiv.org/abs/2111.03794
- **PDEBench.** Takamoto, Praditia, Leiteritz, MacKinlay, Alesiani,
  Pflüger, Niepert. *PDEBench: An Extensive Benchmark for Scientific
  Machine Learning*, NeurIPS 2022. https://arxiv.org/abs/2210.07182

PDEBench scores data fit and some physics residuals at dataset scale.
Coticula is complementary and much smaller: four named exams, fail-closed
trust, and an explanation item set, on two laptop-scale PDEs. We do not
claim a PDEBench comparison.

We implement a *tiny* FNO trunk (time as output channels) and a light
$\lambda_{\mathrm{pde}}$ residual term.

---

## 3. Method

**Equations.** Periodic Burgers $u_t + u u_x = \nu u_{xx}$; Dirichlet heat
$u_t = \alpha(u_{xx}+u_{yy})$.

**Labels.** `scripts/generate_labels.py` writes `burgers_v0.npz` (48 traj,
$n_x=64$, $n_t=80$, $\nu\in[0.005,0.05]$, Fourier-mode ICs) and
`heat2d_v0.npz` (32 traj, $n=32$, $n_t=40$, $\alpha\in[0.05,0.2]$, Gaussian
bumps), plus SHA256 manifests (`datasets/manifests/labels_v0.json`).
OOD generators are pinned by recipe (`datasets/ood/probe_plan_v0.json`),
not by stored NPZ.

**Baselines.** Classical FD (labeler + Exam 3 grader). FNO / PINO when a
`checkpoints/*.pt` loads. LLM judge is not wired.

**Fail-closed trust** (`metrics/trust.py`). Thresholds are documented
defaults (residual rel-$L^2>1$, Predict rel-$L^2>0.5$), not tuned after
seeing scores. Statuses: `trusted`, `ood`, `untrusted`, `reference`
(classical labeler). OOD/untrusted fields must carry
`refuse_silent_heatmap()`.

**OOD probes.** Each probe isolates one shift: $\nu\in\{0.001,0.15\}$,
$\alpha\in\{0.01,0.50\}$, Burgers $n_x\in\{32,128\}$, heat $n\in\{16,48\}$,
Burgers ICs `gaussian_pulse` / `tanh_front`, heat ICs `two_bump` /
`sinusoid`. Ground truth is a *fresh* classical solve. **Time is locked**
as FNO output channels (no temporal transfer claim).

**Exam 4 (v0.2).** Pinned 48-item set (`datasets/exam4/items_v0.json`).
Three structured grades, none of them an LLM score:

| Slice | What is scored |
|-------|----------------|
| gold reference | Authored rationales vs expected law keywords (rubric ceiling) |
| rule-based | Facet-level templates + measured Exam 1–3 observations |
| metric-dump | Numbers-only dump (negative control) |
| LLM | `not_trained` unless a real judge is wired (`COTICULA_LLM_JUDGE`) |

**Diagnostics (v0.2).** Residual-versus-$L^2$ scatter from already-measured
IID + OOD cells; early/late residual split on the stored horizon; classical
coefficient-sensitivity sweep. Exported to `reports/diagnostics.json`.

---

## 4. Four exams

| Exam | Question | IID metric |
|------|----------|------------|
| 1 Predict | Match future fields? | Rel-$L^2$, NMSE vs stored labels |
| 2 Conserve | Respect the discrete PDE / energy? | FD residual rel-$L^2$; energy drift |
| 3 Counterfactual | What if $\nu$ or $\alpha$ changes? | Learned pred at $\theta'$ vs classical at $\theta'$ |
| 4 Explain | Does a rationale cite the physics? | 48-item pinned set; keyword coverage |

OOD is **not** Exam 1. Transfer scores live in `reports/latest.json` → `ood`.

---

## 5. Experiments

Source: `reports/latest.json` / `reports/latest.md`.
IID + OOD digits: `python -m scripts.run_eval` on the Week 3 revision
(`iid_ood_measured_at_utc`: **2026-09-12T00:18:43.378079+00:00**; same
Week 2 checkpoints). Exam 4 and diagnostics were graded/solved on this
v0.2 revision. No FNO/PINO cell was rewritten.

No number below is invented. `not_trained` means the checkpoint or judge
was absent.

### 5.1 IID Exam 1 — Predict

| Baseline | Burgers rel-$L^2$ | Heat2D rel-$L^2$ | NMSE (B / H) | Trust |
|----------|------------------:|-----------------:|-------------:|-------|
| classical_fd | $0$ | $0$ | $0$ / $0$ | `reference` |
| FNO | $1.573\times10^{-1}$ | $3.221\times10^{-1}$ | $2.400\times10^{-2}$ / $1.203\times10^{-1}$ | `untrusted` |
| PINO | $1.880\times10^{-1}$ | — | $3.433\times10^{-2}$ / — | `untrusted` / `not_trained` |
| LLM | — | — | — | `not_trained` |

FNO/PINO are `untrusted` on IID because Exam 2 residual rel-$L^2$ exceeds $1$
(see §5.2), not because the holdout $L^2$ is above $0.5$.

### 5.2 IID Exam 2 — Conserve

| Baseline | Burgers residual | Heat residual | Burgers energy max \|rel drift\| |
|----------|-----------------:|--------------:|---------------------------------:|
| classical_fd | $1.438\times10^{-2}$ | $3.033\times10^{-2}$ | $4.795\times10^{-1}$ |
| FNO | $4.849$ | $1.303\times10^{1}$ | $5.393\times10^{-1}$ |
| PINO | $1.733$ | — (`not_trained`) | $4.406\times10^{-1}$ |

Classical residuals are the discrete scheme's own audit (not zero). Learned
residuals are field-scale — the fail-closed layer refuses a silent heatmap.

### 5.3 IID Exam 3 — Counterfactual

| Baseline | Burgers vs classical at $\nu'$ | Heat vs classical at $\alpha'$ |
|----------|-------------------------------:|-------------------------------:|
| classical $\Delta$ (re-solve) | $1.190\times10^{-1}$ ($\nu$: $0.03366\to0.06733$) | $2.081\times10^{-1}$ ($\alpha$: $0.1268\to0.1902$) |
| FNO | $1.793\times10^{-1}$ | $9.553\times10^{-1}$ |
| PINO | $2.397\times10^{-1}$ | `not_trained` |

FNO heat at $\alpha'$ is almost a miss ($0.955$ rel-$L^2$ vs the classical
solve). The model's own trajectory $\Delta$ when $\alpha$ is scaled is only
$1.70\times10^{-2}$ against a classical $\Delta$ of $2.34\times10^{-1}$
(§5.6).

### 5.4 Exam 4 — Explain (v0.2 item set)

Source: `datasets/exam4/items_v0.json` (n=48, sha256 `48323ca8ea9d8277…`)
graded by `metrics.explain`. **Not** an LLM score. **Not** an FNO/PINO
explanation metric.

| Slice | n | mean keyword coverage | perfect / zero |
|-------|--:|----------------------:|----------------|
| gold reference | 48 | $1.000$ | $48$ / $0$ |
| rule-based (facet templates + measured obs.) | 48 | $0.8785$ | $33$ / $1$ |
| metric-dump (numbers only) | 48 | $0.0903$ | $0$ / $35$ |
| LLM judge | — | `not_trained` (`score: null`) | — |

Gold-reference coverage is the keyword rubric applied to the authored gold
rationales (a ceiling for this item set). Rule-based answers are
facet-level templates, so they miss item-specific terms (e.g. `cfl`,
`touchstone`); that gap is the point. The metric-dump shows the rubric is
not a free lunch: dumping measured $L^2$ / residual numbers scores $0.090$.
Empty / missing answers score $0$. CI and `tests/test_exam4.py` lock this.

Rule-based by PDE (measured): burgers $0.893$ / heat $0.905$ / harness
$0.850$. By facet: counterfactual $1.000$, physics $0.958$, pde_terms
$0.905$, ood $0.889$, trust $0.875$, exams $0.867$, numerics $0.708$,
explain $0.667$.

### 5.5 OOD / transfer (distinct from IID)

Cells are **Predict rel-$L^2$ / residual [trust]** vs a *fresh* classical
solve — not Exam 1. Classical residual of the labeler is shown for scale.
Trust rollup: $n=30$ scored rows (`untrusted=18`, `reference=12`).

| Probe | classical residual | FNO | PINO |
|-------|-------------------:|-----|------|
| Burgers $\nu=0.001$ (below) | $4.19\times10^{-2}$ | $2.70\times10^{-1}$ / $5.27$ [`untrusted`] | $2.95\times10^{-1}$ / $1.89$ [`untrusted`] |
| Burgers $\nu=0.15$ (above) | $9.86\times10^{-3}$ | $2.94\times10^{-1}$ / $4.83$ [`untrusted`] | $3.02\times10^{-1}$ / $1.93$ [`untrusted`] |
| Heat $\alpha=0.01$ (below) | $1.05\times10^{-2}$ | $5.69\times10^{-1}$ / $6.35$ [`untrusted`] | `not_trained` |
| Heat $\alpha=0.50$ (above) | $1.88\times10^{-1}$ | $1.13$ / $43.3$ [`untrusted`] | `not_trained` |
| Burgers $n_x=32$ | $8.79\times10^{-3}$ | $1.40\times10^{-1}$ / $4.53$ [`untrusted`] | $1.72\times10^{-1}$ / $1.61$ [`untrusted`] |
| Burgers $n_x=128$ | $1.12\times10^{-2}$ | $1.35\times10^{-1}$ / $3.87$ [`untrusted`] | $1.58\times10^{-1}$ / $1.31$ [`untrusted`] |
| Heat $n=16$ | $9.74\times10^{-2}$ | $1.87\times10^{-1}$ / $6.57$ [`untrusted`] | `not_trained` |
| Heat $n=48$ | $4.18\times10^{-2}$ | $1.97\times10^{-1}$ / $21.0$ [`untrusted`] | `not_trained` |
| Burgers IC `gaussian_pulse` | $7.49\times10^{-3}$ | $5.20\times10^{-1}$ / $4.14$ [`untrusted`] | $5.70\times10^{-1}$ / $2.28$ [`untrusted`] |
| Burgers IC `tanh_front` | $8.76\times10^{-3}$ | $2.45\times10^{-1}$ / $4.35$ [`untrusted`] | $2.62\times10^{-1}$ / $2.05$ [`untrusted`] |
| Heat IC `two_bump` | $2.64\times10^{-2}$ | $2.42\times10^{-1}$ / $14.1$ [`untrusted`] | `not_trained` |
| Heat IC `sinusoid` | $3.85\times10^{-1}$ | $8.19\times10^{-1}$ / $25.8$ [`untrusted`] | `not_trained` |

Spatial resolution transfer on Burgers does **not** blow up Predict $L^2$
relative to IID ($0.13$–$0.14$ vs $0.16$), which is consistent with FNO's
spectral conv — but residuals stay $O(1)$–$O(5)$, so the trust layer still
refuses the plot. Heat $\alpha=0.5$ and the sinusoid IC are the harshest
Predict misses.

### 5.6 Failure analysis

Three measured comparisons from `failure_analysis.notes`:

1. **FNO vs PINO on IID Burgers (L2 vs residual).** FNO Predict rel-$L^2$
   $0.157$ beats PINO $0.188$, but FNO residual $4.85$ is worse than PINO
   $1.73$. A silent FNO heatmap would look “closer” to the label while Exam 2
   flags a larger PDE violation. Both are `untrusted`. This is a measured
   tradeoff from $\lambda_{\mathrm{pde}}=10^{-3}$, not a SOTA claim.
2. **FNO heat counterfactual insensitivity.** Classical $\Delta$ at
   $\alpha'=1.5\alpha$ is $0.234$; the FNO field moves by only $0.017$.
   Exam 3 vs the classical solve at $\alpha'$ is $0.955$.
3. **OOD Burgers $\nu=0.001$** (below $[0.005,0.05]$). Transfer $L^2$ is
   $0.270$ (FNO) / $0.295$ (PINO) with residuals $5.27$ / $1.89$. These are
   **not** Exam 1 numbers. Flags: `untrusted` (residual) and `ood` (param).

### 5.7 Diagnostics (v0.2)

#### Residual vs $L^2$ scatter

$n=21$ measured (model, split) points: 3 IID learned cells + 18 OOD learned
cells from the Week 3 report. Pearson $r=0.741$ (residual vs Predict
$L^2$); Spearman $\rho=0.342$. Residual median $4.53$ (min $1.31$, max
$43.3$); Predict $L^2$ median $0.262$ (min $0.135$, max $1.13$). All $21$
points are `untrusted`. Pearson is pulled by the heat $\alpha=0.5$ and
sinusoid tails; the rank correlation is weaker, so a single ranking by
$L^2$ still disagrees with residual order on several Burgers pairs (the
IID FNO/PINO tradeoff is one of them).

#### Horizon residual split (classical labels)

Early vs late half of the *stored* horizon — not a longer-time transfer
claim (FNO time is locked as output channels).

| PDE | early | late | full | late − early |
|-----|------:|-----:|-----:|-------------:|
| Burgers (traj 0) | $1.870\times10^{-2}$ | $7.068\times10^{-3}$ | $1.288\times10^{-2}$ | $-1.163\times10^{-2}$ |
| Heat (traj 0) | $2.936\times10^{-2}$ | $5.381\times10^{-3}$ | $1.737\times10^{-2}$ | $-2.397\times10^{-2}$ |

Late residual is smaller than early on both labelers (fields smooth). This
is a discrete-scheme diagnostic, not a learned-model score.

#### Counterfactual sensitivity (classical)

Same IC as Exam 3 classical rows. Scale $2.0$ (Burgers) and $1.5$ (heat)
reproduce the Exam 3 trajectory $\Delta$ exactly ($1.190\times10^{-1}$,
$2.081\times10^{-1}$).

| scale | Burgers $\nu'$ | Burgers traj $\Delta$ | Heat $\alpha'$ | Heat traj $\Delta$ |
|------:|---------------:|----------------------:|---------------:|-------------------:|
| $0.5$ | $0.01683$ | $8.864\times10^{-2}$ | $0.06339$ | $3.304\times10^{-1}$ |
| $1.0$ | $0.03366$ | $0$ | $0.1268$ | $0$ |
| $1.5$ | $0.05049$ | $6.601\times10^{-2}$ | $0.1902$ | $2.081\times10^{-1}$ |
| $2.0$ | $0.06733$ | $1.190\times10^{-1}$ | $0.2535$ | $3.495\times10^{-1}$ |
| $3.0$ | $0.1010$ | $2.003\times10^{-1}$ | — | — |

Learned columns are `—`: this revision did not re-forward FNO/PINO (no
new checkpoint train). The sweep is the classical reference a later
learned row can be graded against.

---

## 6. Limitations

- Laptop-scale operators (~20k–45k parameters), short training, two PDEs.
- Time-as-channels FNO does not transfer in $t$; only spatial resolution is probed.
- Exam 4 is a keyword item set, not a trained explanation judge and not an
  LLM score. Gold-reference coverage is the rubric ceiling on authored texts.
  Rule-based coverage is a structured template, not model-generated language.
- Residual thresholds are conservative and documented; they are not a
  calibrated posterior.
- Classical energy drift on dissipative PDEs is expected (not a conservation
  law of the continuous energy for inviscid Burgers / heat).
- Residual-vs-$L^2$ correlations are descriptive on $n=21$ already-measured
  points; they are not a new train/eval split.

---

## 7. Non-claims

- No chip-cooling / industrial multiphysics results.
- No AU-scale (astronomical-unit) or cosmology / foundation-model workloads.
- No fabricated SOTA for FNO, PINO, or LLM.
- No claim that keyword rubrics equal explanation quality.
- No claim that a trusted flag means the model “understands” the PDE — only
  that it stayed inside declared support and below documented residual/L2
  cuts on this harness.

---

## How to reproduce

```bash
pip install -e ".[dev,torch]"
python -m scripts.generate_labels
python -m scripts.verify_manifests
python -m scripts.train_fno --pde burgers
python -m scripts.train_fno --pde heat2d
python -m scripts.train_pino --pde burgers
python -m scripts.run_eval
# Re-grade Exam 4 + diagnostics without retraining (keeps measured FNO/PINO cells):
python -m scripts.refresh_v02_report
pytest -q
# CI-equivalent smoke (no labels):
python -m scripts.train_fno --smoke
python -m scripts.train_pino --smoke --pde burgers
```

Checkpoints already in `checkpoints/` may be reused; eval never invents
weights. CI: `.github/workflows/ci.yml`. Design notes: [DESIGN.md](DESIGN.md).
One-pager: [ONEPAGER.md](ONEPAGER.md).
