# Vermithor VU-Bench v0 — Understanding Harness

**VU = Vermithor Understanding** (not the `uv` Python packager).

**Thesis.** VU-Bench is an *understanding harness* for PDE surrogates under the
Vermithor project: four exams (Predict, Conserve, Counterfactual, Explain) that
score whether a model has grasped the dynamics—not just interpolated training
trajectories.

Week 1 shipped classical finite-difference label generators and the exam APIs.
Week 2 adds a laptop-scale **FNO** (Li et al., ICLR 2021) and a **PINO-style**
path (data loss + PDE residual). Reports contain **only measured numbers**.
Missing checkpoints stay `not_trained`. The LLM exam is a keyword rubric stub.

## Layout

```
vu-bench/
  datasets/{burgers,heat2d,manifests}/
  metrics/{predict,conserve,counterfactual,explain}.py
  baselines/classical/          # NumPy FD solvers (labels + Exam 3 grader)
  baselines/fno/                # tiny PyTorch FNO (Burgers 1D, heat 2D)
  baselines/pino/               # PINO losses (same trunk + residual term)
  baselines/llm/                # explain stub (no fabricated judge scores)
  scripts/generate_labels.py
  scripts/train_fno.py
  scripts/train_pino.py
  scripts/run_eval.py
  checkpoints/                  # optional *.pt written by train_* 
  docs/DESIGN.md
  tests/
  reports/                      # latest.json + latest.md
```

## How to run

Requires Python 3.11+ and NumPy / SciPy. Torch is optional for classical-only
eval and is required to train or score FNO / PINO.

```bash
pip install -e ".[dev]"          # classical + pytest
pip install -e ".[dev,torch]"    # + PyTorch (CPU is enough)

python -m scripts.generate_labels

# Tiny FNO on Burgers (defaults are laptop / single-GPU friendly)
python -m scripts.train_fno --pde burgers
python -m scripts.train_fno --pde heat2d

# PINO = same trunk + λ_pde * PDE residual (see docs/DESIGN.md)
python -m scripts.train_pino --pde burgers

python -m scripts.run_eval
pytest -q
```

CPU CI / smoke (no labels needed):

```bash
python -m scripts.train_fno --smoke
python -m scripts.train_pino --smoke --pde burgers
```

Useful knobs (all have small defaults): `--epochs`, `--width`, `--modes`,
`--n-layers`, `--batch-size`, `--lambda-pde`, `--out`.

Artifacts:

- `datasets/burgers/burgers_v0.npz`, `datasets/heat2d/heat2d_v0.npz`
- `datasets/manifests/labels_v0.json` (SHA256)
- `checkpoints/fno_burgers.pt`, `checkpoints/fno_heat2d.pt`, `checkpoints/pino_*.pt`
- `reports/latest.json`, `reports/latest.md` — four exam sections

`run_eval` always scores the classical solver. It loads FNO / PINO only when a
checkpoint is present. It does **not** invent metrics.

## Exams

| Exam | What is measured |
|------|------------------|
| 1 Predict | Rel-L2 and NMSE vs classical labels |
| 2 Conserve | FD PDE residual + discrete energy drift of the predicted field |
| 3 Counterfactual | Re-solve / re-predict at ν′ or α′; learned models graded vs the classical solver |
| 4 Explain | Keyword-rubric stub. LLM hook stays `not_trained` without a wired judge |

## Non-goals (unchanged)

- No chip-cooling / industrial multiphysics demos
- No AU-scale (astronomical-unit) or cosmology workloads
- No fabricated SOTA numbers for FNO / PINO / LLM
- No claim that keyword rubrics equal true explanation quality

## License

Internal Vermithor research scaffold.
