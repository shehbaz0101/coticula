"""Adapter protocol for grading a user surrogate.

Coticula calls ``predict`` with NumPy arrays and does not import the user's
trainer. A class with ``name`` and ``predict`` is enough. Subclassing is
optional.

See ``examples/persist_ic.py`` for a complete file, and ``TorchOperatorAdapter``
for how the shipped FNO checkpoints are wrapped.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Surrogate(Protocol):
    """Batch predictor used by every exam except Explain.

    ``n_steps`` is the number of time intervals. The returned trajectory has
    length ``n_steps + 1`` and includes the initial condition at index 0.
    """

    name: str

    def predict(self, u0: np.ndarray, coeff: np.ndarray, *, n_steps: int) -> np.ndarray:
        """Return a batched trajectory with the same batch axis as ``u0``."""


def repo_root() -> Path:
    """Directory that contains this checkout (pyproject + metrics/)."""
    here = Path(__file__).resolve()
    seeds = [Path.cwd(), *here.parents]
    for path in seeds:
        if (path / "pyproject.toml").is_file() and (path / "metrics").is_dir():
            return path
    return here.parents[1]


def load_adapter_spec(spec: str):
    """Load ``path/to/module.py:ClassName`` and construct it with no arguments."""
    if ":" not in spec:
        raise ValueError(
            f"adapter spec {spec!r} must look like examples/persist_ic.py:PersistIC"
        )
    path_s, qual = spec.split(":", 1)
    path = Path(path_s)
    if not path.is_file():
        path = repo_root() / path_s
    if not path.is_file():
        raise FileNotFoundError(f"adapter module not found: {path_s}")
    module_name = "coticula_user_" + path.stem
    loaded = importlib.util.spec_from_file_location(module_name, path)
    if loaded is None or loaded.loader is None:
        raise ImportError(f"cannot import adapter module {path}")
    module = importlib.util.module_from_spec(loaded)
    loaded.loader.exec_module(module)
    obj = module
    for part in qual.split("."):
        obj = getattr(obj, part)
    instance = obj() if isinstance(obj, type) else obj
    if not hasattr(instance, "predict"):
        raise TypeError(f"{qual} has no predict() method")
    if not getattr(instance, "name", None):
        instance.name = qual.split(".")[-1]
    return instance


class ClassicalFD:
    """Reference labeler. Re-solves the PDE. Not a learned model."""

    def __init__(self, pde: str, *, L: float, T: float):
        self.name = "classical_fd"
        self.pde = pde
        self.L = float(L)
        self.T = float(T)
        self.role = "labeler"

    def predict(self, u0: np.ndarray, coeff: np.ndarray, *, n_steps: int) -> np.ndarray:
        u0 = np.asarray(u0, dtype=np.float64)
        coeff = np.asarray(coeff, dtype=np.float64).reshape(-1)
        preds = []
        if self.pde == "burgers":
            from baselines.classical.burgers1d import solve_burgers

            for i in range(u0.shape[0]):
                out = solve_burgers(
                    nu=float(coeff[i]),
                    nx=int(u0.shape[-1]),
                    nt=int(n_steps),
                    L=self.L,
                    T=self.T,
                    u0=u0[i],
                )
                preds.append(out["u"])
        elif self.pde == "heat2d":
            from baselines.classical.heat2d import solve_heat2d

            for i in range(u0.shape[0]):
                out = solve_heat2d(
                    alpha=float(coeff[i]),
                    n=int(u0.shape[-1]),
                    nt=int(n_steps),
                    L=self.L,
                    T=self.T,
                    u0=u0[i],
                )
                preds.append(out["u"])
        else:
            raise ValueError(f"unknown pde {self.pde!r}")
        return np.stack(preds, axis=0)


class TorchOperatorAdapter:
    """Wrap a BurgersFNO / HeatFNO (or the same trunk trained as PINO)."""

    def __init__(self, model, *, name: str, checkpoint: str | None, device: str = "cpu"):
        self.model = model
        self.name = name
        self.checkpoint = checkpoint
        self.device = device
        self.role = "surrogate"
        self.model.eval()

    def predict(self, u0: np.ndarray, coeff: np.ndarray, *, n_steps: int) -> np.ndarray:
        import torch

        expected = int(getattr(self.model, "nt"))
        if int(n_steps) != expected:
            raise ValueError(
                f"{self.name} was built for nt={expected}, got n_steps={n_steps}"
            )
        u0_t = torch.from_numpy(np.asarray(u0, dtype=np.float32)).to(self.device)
        coeff_t = torch.from_numpy(np.asarray(coeff, dtype=np.float32).reshape(-1)).to(
            self.device
        )
        with torch.no_grad():
            pred = self.model(u0_t, coeff_t).detach().cpu().numpy()
        return pred


def persist_example():
    """The documented toy adapter (``examples/persist_ic.py``)."""
    return load_adapter_spec("examples/persist_ic.py:PersistIC")
