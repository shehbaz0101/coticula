# Vermithor VU-Bench v0 — Understanding Harness

**Thesis.** VU-Bench is an *understanding harness* for PDE surrogates: four exams
(Predict, Conserve, Counterfactual, Explain) that score whether a model has
grasped the dynamics—not just interpolated training trajectories. Week-1 ships
classical finite-difference label generators, metric APIs, and an eval that
writes real classical numbers. Learned baselines (FNO / PINO / LLM) are
importable stubs marked `not_trained`.

## Layout

```
vu-bench/
  datasets/{burgers,heat2d,manifests}/
  metrics/{predict,conserve,counterfactual,explain}.py
  baselines/classical/          # NumPy FD solvers (labels)
  baselines/{fno,pino,llm}/     # stubs
  scripts/generate_labels.py
  scripts/run_eval.py
  docs/DESIGN.md
  tests/test_metrics_smoke.py
  reports/                      # latest.json + latest.md
```

## How to run

Requires Python 3.11+ and NumPy / SciPy (see `pyproject.toml`).

```bash
cd /workspace/vu-bench
pip install -e ".[dev]"          # or: pip install numpy scipy pytest

python -m scripts.generate_labels
python -m scripts.run_eval
pytest -q
```

Artifacts:

- `datasets/burgers/burgers_v0.npz`, `datasets/heat2d/heat2d_v0.npz`
- `datasets/manifests/labels_v0.json` (SHA256)
- `reports/latest.json`, `reports/latest.md` — four exam sections

## Non-goals (Week 1)

- No chip-cooling / industrial multiphysics demos
- No AU-scale (astronomical-unit) or cosmology workloads
- No fabricated SOTA numbers for FNO / PINO / LLM — stubs report `not_trained`
- No claim that keyword rubrics equal true explanation quality

## License

Internal Vermithor research scaffold.
