# Coticula

Coticula (Latin *coticula*, touchstone) is a small harness for PDE surrogates
on 1D viscous Burgers and 2D heat. On the shipped FNO, the counterfactual
exam is the result that holds up: the heat model barely moves when
diffusivity changes, and the classical solver does not. Predict error and a
strong-form residual are reported beside that. The residual flag is a
roughness indicator, not evidence that the physics is violated (see Known
limitation). Flags are `trusted`, `ood`, `untrusted`, or `reference`.
Missing checkpoints stay `not_trained`.

Repo: https://github.com/shehbaz0101/coticula

## Quickstart

Python 3.11+, NumPy, SciPy. PyTorch is required to score the shipped FNO and
PINO checkpoints. Matplotlib writes the PNG plots (`pip install -e ".[dev]"`
or `".[plot]"`).

```bash
pip install -e ".[dev]"
pip install torch --index-url https://download.pytorch.org/whl/cpu

coticula demo --out reports/demo
# same entry point: python -m coticula demo --out reports/demo
```

That command grades the shipped checkpoints on the checkpoint holdout, plus
the toy adapter in `examples/persist_ic.py`. It prints a scorecard and writes
`scorecard.json`, `scorecard.md`, `report.html`, and PNG plots under
`reports/demo/`. Open `report.html` in a browser. CPU, about a few seconds.
If the label NPZ files are missing, the demo builds them in memory with the
same recipe as `python -m scripts.generate_labels` (Burgers seed 0, heat seed 1).

`coticula demo` scores the checkpoint holdout stored on the shipped FNO
weights: 12 Burgers trajectories (indices 36-47 of 48) and 8 heat
trajectories (indices 24-31 of 32). The terminal scorecard names that set.
Numbers there are short (`0.157`). `scorecard.json` and `scorecard.md` keep
the full precision.

`coticula grade` does not always use that holdout. `--model fno` or
`--model pino` loads the checkpoint and scores the same indices as demo.
`--adapter`, or `--model persist_ic` with no checkpoint, scores every
trajectory (Burgers 48, heat 32). That is why a `persist_ic` number from
`grade` can differ from the demo table. `--split holdout` forces the demo
indices.

```bash
coticula grade --model fno --pde burgers --out reports/grade
coticula grade --model pino --pde burgers
coticula grade --adapter examples/persist_ic.py:PersistIC --pde burgers --split holdout
```

The older scripts still work:

```bash
python -m scripts.generate_labels
python -m scripts.run_eval
python -m scripts.run_ood
python -m scripts.train_fno --pde burgers
pytest -q
```

`coticula demo --smoke` trains a 1-epoch FNO on a 16-point Burgers grid. That
path is for a tiny check. It is not the shipped checkpoint. CI runs the real
`coticula demo` plus `pytest` and the existing `--smoke` trainers.

## Results

Two commands, both run on 2026-10-10. Learned cells match across them.
Classical residuals do not, because the commands do not use the same slice.
`scripts.run_eval` re-solves the first trajectories. `coticula demo` re-solves
the checkpoint holdout (Burgers indices 36-47, 12 trajectories; heat indices
24-31, 8 trajectories). `coticula grade --model fno` or `--model pino` uses
that same holdout. `coticula grade` of an adapter, with no shipped checkpoint
loaded, scores every trajectory (Burgers 48, heat 32) unless you pass
`--split holdout`.

Harness table, stdout of `python -m scripts.render_results_table` after
`python -m scripts.run_eval` (generated 2026-10-10T04:03:32Z). The PINO
heat2d cell is annotated; the script prints `not_trained`.

| model | pde | Predict rel-L2 | residual rel-L2 | trust |
|---|---|---:|---:|---|
| classical_fd | burgers | 0.000000e+00 | 1.438256e-02 | reference |
| classical_fd | heat2d | 0.000000e+00 | 3.033189e-02 | reference |
| fno | burgers | 1.573143e-01 | 4.848762e+00 | untrusted |
| fno | heat2d | 3.221335e-01 | 1.303385e+01 | untrusted |
| pino | burgers | 1.880157e-01 | 1.732850e+00 | untrusted |
| pino | heat2d | - | - | not_trained (no checkpoint shipped, to keep the repo small) |

Exam 4 n=48 sha256=48323ca8ea9d8277 gold=1.000000 rule-based=0.878472 metric-dump=0.090278 llm=not_trained

Residual-vs-L2 points n=21 pearson=0.7411755514841815 spearman=0.34155844155844156 untrusted=21 ood_flag=0

PINO heat2d is `not_trained`: no checkpoint shipped, to keep the repo small.
All 21 learned IID and OOD points in the scatter carry `untrusted`, because
the residual is also over the cut, and that flag wins over `ood`. That
residual flag is the roughness indicator in Known limitation, not a separate
physics verdict.

Demo holdout, copied from `reports/demo/scorecard.md` written by
`coticula demo --out reports/demo` at 2026-10-10T04:04:17Z. FNO and PINO
Predict and residual cells match the harness table.

| model | pde | Predict rel-L2 | residual rel-L2 | trust |
|---|---|---:|---:|---|
| classical_fd | burgers | 0.000000e+00 | 1.319436e-02 | reference |
| fno | burgers | 1.573143e-01 | 4.848762e+00 | untrusted |
| pino | burgers | 1.880157e-01 | 1.732850e+00 | untrusted |
| persist_ic | burgers | 3.771952e-01 | 8.091685e-01 | trusted |
| classical_fd | heat2d | 0.000000e+00 | 2.983392e-02 | reference |
| fno | heat2d | 3.221335e-01 | 1.303385e+01 | untrusted |
| pino | heat2d | - | - | not_trained (no checkpoint shipped, to keep the repo small) |
| persist_ic | heat2d | 1.424304e+00 | 1.280133e+01 | untrusted |

One OOD probe per equation (viscosity or diffusivity below the train range,
n=4). Full probe list remains `python -m scripts.run_ood`.

| model | probe | Predict rel-L2 | residual rel-L2 | trust |
|---|---|---:|---:|---|
| fno | burgers_param_below | 2.696545e-01 | 5.274467e+00 | untrusted |
| pino | burgers_param_below | 2.951888e-01 | 1.890248e+00 | untrusted |
| persist_ic | burgers_param_below | 4.691619e-01 | 8.935663e-01 | ood |
| fno | heat2d_param_below | 5.684832e-01 | 6.353075e+00 | untrusted |
| persist_ic | heat2d_param_below | 2.483906e-01 | 1.060790e+00 | untrusted |

Cuts, fixed in `metrics/trust.py` before these scores: Predict rel-L2 above
0.5, or residual rel-L2 above 1.0, is `untrusted`. Leaving the declared train
support is `ood` only when both cuts pass. `trusted` is not a claim that the
model understands the PDE.

What this demo run measured, from `coticula demo --out reports/demo`
(reproduced 2026-10-10; counterfactual cells below match that scorecard):

- FNO heat barely moves when diffusivity is scaled by 1.5. Model trajectory
  change 1.703234e-02, classical change 2.338120e-01, model vs the classical
  solve at the new diffusivity 9.553082e-01.
- FNO Burgers barely moves when viscosity is doubled. Model trajectory change
  3.954555e-03, classical change 1.473826e-01, model vs the classical solve
  at the new viscosity 1.793023e-01.
- `persist_ic` does not move on either equation (trajectory change 0). The
  trust flag does not look at this exam.
- On Burgers, FNO Predict rel-L2 is 1.573143e-01 and the residual rel-L2 is
  4.848762e+00, so the flag is `untrusted` with reason `residual_too_high`.
  PINO Predict rel-L2 is 1.880157e-01 and the residual rel-L2 is 1.732850e+00,
  flag `untrusted`, same reason. These are the measured flag values. They are
  not a claim that either model violates the PDE.
- `persist_ic` on the Burgers holdout has Predict rel-L2 3.771952e-01 and
  residual rel-L2 8.091685e-01, flag `trusted`. On the Burgers param-shift
  probe the flag is `ood` (`param_shift` only).

Exam 4 on this demo matched the harness numbers above: gold 1.000000,
rule-based 0.878472, metric-dump 0.090278, llm `not_trained`. Gold is the
authored answer key. Rule-based text is templates plus the measured
observations. Metric-dump is numbers only. None of those is an LLM score.

Longer tables and the failure notes live in `reports/latest.md` and
`docs/WRITEUP.md`.

## Grade your own model

Implement `name` and `predict`. The harness always passes a batch.
`n_steps` is the number of time intervals. Return shape is
`(batch, n_steps + 1, ...)`, including the initial condition at time 0.
Burgers: `u0` is `(batch, nx)` and `coeff` is viscosity.
Heat: `u0` is `(batch, n, n)` and `coeff` is diffusivity.

`examples/persist_ic.py` is a complete file. It repeats the initial condition
and ignores the coefficient. It is an adapter example, not a baseline I am
claiming as a solver.

```bash
coticula grade --adapter examples/persist_ic.py:PersistIC --pde burgers
```

A class with those two attributes is enough. You do not have to subclass
anything. The shipped FNO and PINO weights go through the same `predict`
call (`coticula.adapter.TorchOperatorAdapter`).

## Known limitation

The strong-form finite-difference residual amplifies high-frequency error, so
a noisy exact solution can cross the residual cut. `python -m scripts.probe_residual_noise`
(run 2026-10-10) takes one exact Burgers
trajectory from the label recipe (nx=64, nt=80, seed 0). The clean residual
rel-L2 is 1.597894e-02. Adding 1% white noise, with standard deviation
`0.01 * std(u)` and noise seed 0, raises it to 1.134730e+00, which is over
the residual cut of 1. Treat the residual flag as a roughness indicator for
now. A weak-form, noise-calibrated residual is planned. The trust rule itself
is unchanged in this release.

## Limitations

- The shipped checkpoints are the laptop-scale weights already in `checkpoints/`. This is not a SOTA claim.
- The residual flag is a roughness indicator. See Known limitation. On Burgers,
  a model that never moves can land `trusted` while a lower Predict error
  lands `untrusted`. Read the counterfactual exam next to the flag.
- `trusted` does not mean the model understands the PDE.
- PINO heat2d is `not_trained`: no checkpoint shipped, to keep the repo small. PINO Burgers is the shipped checkpoint.
- Exam 4 is keyword coverage on a pinned 48-item set. The LLM judge is off
  unless a real client is wired, and this repo does not invent that score.
- OOD trajectories are generated from `datasets/ood/probe_plan_v0.json`.
  They are not stored as NPZ. Label NPZ files are gitignored.
- No chip-cooling demo, and no astronomical-unit or cosmology workloads.

## Citation

Software citation is `CITATION.cff` (Shehbaz Pathan, version 0.3.1).

```bibtex
@software{pathan_coticula_2026,
  author = {Pathan, Shehbaz},
  title = {Coticula},
  year = {2026},
  version = {0.3.1},
  url = {https://github.com/shehbaz0101/coticula}
}
```

## Layout

```
coticula/          CLI, adapter, scorecard writers
examples/persist_ic.py
metrics/           predict, conserve, counterfactual, explain, ood, trust
baselines/         classical FD, FNO, PINO loss, LLM stub
checkpoints/       shipped fno_burgers, fno_heat2d, pino_burgers
scripts/           generate_labels, run_eval, run_ood, train_*, render_results_table, probe_residual_noise
datasets/exam4/    48 pinned Exam 4 items
reports/           latest.json from run_eval; demo/ is generated and gitignored
```

## License

Internal Coticula research scaffold.
