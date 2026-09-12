# Checkpoints

Optional weights written by `python -m scripts.train_fno` / `train_pino`.

| File | Meaning |
|------|---------|
| `fno_burgers.pt` | Data-only FNO on Burgers labels |
| `fno_heat2d.pt` | Data-only FNO on heat-2D labels |
| `pino_burgers.pt` | FNO trunk + PDE residual (`--pino`) |
| `pino_heat2d.pt` | Same for heat-2D |

`scripts/run_eval.py` loads these if present. If a file is missing the report
says `not_trained` — it does not invent a score.

The `*.pt` files in this folder (when committed) are laptop-scale CPU
weights (~20k–45k params), not a SOTA claim. Retrain anytime with
`python -m scripts.train_fno` / `train_pino`.

`--smoke` writes `checkpoints/_smoke_{fno|pino}_{pde}.pt` (gitignored) so a
CI / laptop smoke run does not overwrite the published eval weights.

Format: `vu-bench-fno-v0` dict with `config`, `state_dict`, `training` (includes
the PINO loss-term docstring), and `extra.split` so eval can reuse the holdout.
