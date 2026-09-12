# Vermithor — VU-Bench v0

**VU = Vermithor Understanding Bench** (not the `uv` Python packager).

Public repo: [github.com/shehbaz0101/vermithor](https://github.com/shehbaz0101/vermithor)
(renamed from `shehbaz0101/vu-bench`).

**Thesis.** VU-Bench is Vermithor’s *understanding harness* for PDE surrogates:
four exams (Predict, Conserve, Counterfactual, Explain) that score whether a
model has grasped the dynamics—not just interpolated training trajectories.

Week 1 shipped classical finite-difference label generators and the exam APIs.
Week 2 adds a laptop-scale **FNO** (Li et al., ICLR 2021) and a **PINO-style**
path (data loss + PDE residual). Week 3 adds an **OOD / transfer** layer
(param, resolution, IC family), **fail-closed** `ood` / `untrusted` flags, and
a writeup draft. Reports contain **only measured numbers**. Missing checkpoints
stay `not_trained`. The LLM exam is a keyword rubric stub.

## Layout

```
vermithor/
  datasets/{burgers,heat2d,manifests}/
  metrics/{predict,conserve,counterfactual,explain}.py
  baselines/classical/          # NumPy FD solvers (labels + Exam 3 grader)
  baselines/fno/                # tiny PyTorch FNO (Burgers 1D, heat 2D)
  baselines/pino/               # PINO losses (same trunk + residual term)
  baselines/llm/                # explain stub (no fabricated judge scores)
  scripts/generate_labels.py
  scripts/train_fno.py
  scripts/train_pino.py
  scripts/run_eval.py           # IID exams + OOD (default)
  scripts/run_ood.py            # OOD / transfer probes only
  metrics/trust.py              # fail-closed ood / untrusted
  metrics/ood.py                # probe generators + scorers
  checkpoints/                  # optional *.pt written by train_*
  docs/DESIGN.md
  docs/WRITEUP.md               # arXiv-style draft (measured tables)
  tests/
  reports/                      # latest.json + latest.md (IID + OOD)
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

# PINO = same trunk + light λ_pde * PDE residual (default 1e-3; see docs/DESIGN.md)
python -m scripts.train_pino --pde burgers

python -m scripts.run_eval          # IID exams 1–4 + OOD probes + trust flags
python -m scripts.run_ood           # OOD only → reports/ood.json
python -m scripts.run_eval --skip-ood
pytest -q
```

Week 3 how-to (trust layer):

```bash
# After labels + optional checkpoints:
python -m scripts.run_eval
# Read IID tables vs the OOD section — they are not mixed.
# Trust banners: metrics.trust.refuse_silent_heatmap
# Writeup draft filled from that run:
#   docs/WRITEUP.md
```

Roadmap: [docs/DESIGN.md](docs/DESIGN.md) (3-week plan + train support +
fail-closed rules). Writeup: [docs/WRITEUP.md](docs/WRITEUP.md).

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
checkpoint is present. It does **not** invent metrics. OOD probes generate
fresh classical solves outside the v0 train ranges and score transfer
separately. A case that is OOD or has a residual / L2 above the documented
threshold is flagged `ood` or `untrusted` — not a silent pretty heatmap.

## Exams

| Exam | What is measured |
|------|------------------|
| 1 Predict | Rel-L2 and NMSE vs classical labels |
| 2 Conserve | FD PDE residual + discrete energy drift of the predicted field |
| 3 Counterfactual | Re-solve / re-predict at ν′ or α′; learned models graded vs the classical solver |
| 4 Explain | Keyword-rubric stub. LLM hook stays `not_trained` without a wired judge |
| Trust (W3) | Fail-closed `trusted` / `ood` / `untrusted` / `reference` |
| OOD (W3) | Param shift, spatial resolution transfer, IC-family shift |

## Non-goals (unchanged)

- No chip-cooling / industrial multiphysics demos
- No AU-scale (astronomical-unit) or cosmology workloads
- No fabricated SOTA numbers for FNO / PINO / LLM
- No claim that keyword rubrics equal true explanation quality

## License

Internal Vermithor research scaffold.
