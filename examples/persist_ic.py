"""Toy surrogate: repeat the initial condition at every later time.

This is an adapter example, not a research baseline. It shows the protocol
``coticula grade`` expects from a user model.

Run it from the repo root:

    coticula grade --adapter examples/persist_ic.py:PersistIC --pde burgers

``predict`` always receives a batch.

- Burgers: ``u0`` is ``(B, nx)``, ``coeff`` is viscosity, return ``(B, n_steps+1, nx)``.
- Heat: ``u0`` is ``(B, n, n)``, ``coeff`` is diffusivity, return ``(B, n_steps+1, n, n)``.
"""
from __future__ import annotations

import numpy as np


class PersistIC:
    """Hold the initial field fixed. Coefficient is accepted and ignored."""

    name = "persist_ic"

    def predict(self, u0: np.ndarray, coeff: np.ndarray, *, n_steps: int) -> np.ndarray:
        del coeff  # the protocol passes it; this toy does not use it
        field = np.asarray(u0, dtype=np.float64)
        if field.ndim < 2:
            raise ValueError(
                "PersistIC.predict expects a batch: (B, nx) or (B, n, n), "
                f"got shape {field.shape}"
            )
        steps = int(n_steps) + 1
        return np.repeat(field[:, None, ...], steps, axis=1)
