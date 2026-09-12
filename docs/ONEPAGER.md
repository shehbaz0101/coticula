# Coticula

**A touchstone for physics models.**

Latin *coticula*: a stone used to test whether gold is pure.  
Coticula tests whether a PDE surrogate actually understands the physics, not just whether it looks close on a heatmap.

**Repo:** https://github.com/shehbaz0101/coticula  
**Brand:** coticula.ai (domain TBD)  
**Author:** Shehbaz Pathan · v0.1.0

---

## The problem

Language models can talk about physics. Surrogate models can fit simulation snapshots.  
Neither automatically means the system **understands** the dynamics.

A model can score low error and still:

- violate conservation / PDE residuals  
- ignore coefficient changes (bad counterfactuals)  
- look fine in-distribution and fail under mild OOD shift  

Most SciML papers report MSE. That is not enough.

---

## What Coticula is

A small, reproducible **understanding harness** for continuum PDE surrogates.

Four exams, scored separately:

| Exam | Question |
|------|----------|
| **1. Predict** | Can you match the field trajectory? |
| **2. Conserve** | Do residuals / energy drift stay honest? |
| **3. Counterfactual** | If I change viscosity / diffusivity, do you respond? |
| **4. Explain** | Can short answers cite the right law? (keyword rubric in v0.1) |

Fail-closed trust flags: `ood` / `untrusted`. Pretty plots are not allowed to hide bad physics.

---

## Scope (v0.1)

- Equations: viscous **Burgers (1D)** and **heat (2D)**  
- Baselines: classical finite differences · tiny **FNO** · **PINO**-style physics loss  
- OOD probes: parameter shift · resolution transfer · IC family  
- CI: pytest + CPU smoke trains  
- Cite package: `CITATION.cff`, changelog, release notes  

**Not in v0.1:** foundation models, chip cooling, industrial multiphysics, fabricated SOTA.

---

## Measured outcome (honest)

On Burgers IID, FNO can look better on relative L2 than PINO while carrying a **worse** residual.  
The harness flags both as `untrusted` when residuals blow up. That tradeoff is the point.

Heat counterfactuals expose another failure mode: the field barely moves when the coefficient changes, while the classical solver moves a lot.

Full tables: `reports/latest.md` and `docs/WRITEUP.md`.

---

## Why it matters

Accelerated Understanding / neural operators showed that physics AI is not “another ChatGPT.”  
Coticula is the complementary layer: **a public exam** so claims of understanding can be checked.

Use it to:

- compare surrogates fairly  
- refuse silent heatmaps  
- publish negative results without shame  
- grow into stronger PDEs and domains later  

---

## How to run

```bash
pip install -e ".[dev,torch]"
python -m scripts.generate_labels
python -m scripts.train_fno --pde burgers
python -m scripts.run_eval
pytest -q
```

---

## Roadmap

| Version | Focus |
|---------|--------|
| **v0.1** | Harness + FNO/PINO + OOD + trust flags (shipped) |
| **v0.2** | Stronger Exam 4, tighter writeup, more rigorous eval story |
| **Later** | Optional domain wedge or paper submission |

---

*Coticula tests purity. If the model is gold, it will survive the stone.*
