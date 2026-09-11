"""FNO baseline (Week 2): tiny Fourier neural operator, Li et al. ICLR 2021.

`STATUS` is `implemented` (architecture in-repo). A run is `ok` only when
`load_model` finds a real checkpoint; otherwise `not_trained`. No fabricated
metrics or weights.

Torch is imported lazily so `pytest` still collects this package on CPU-only
or torch-free CI.
"""
from __future__ import annotations

from pathlib import Path

STATUS = "implemented"

__all__ = [
    "STATUS",
    "BurgersFNO",
    "HeatFNO",
    "count_parameters",
    "load_model",
    "predict",
    "default_ckpt",
    "save_checkpoint",
]


def __getattr__(name: str):
    if name in {"BurgersFNO", "HeatFNO", "count_parameters"}:
        from .models import BurgersFNO, HeatFNO, count_parameters

        mapping = {
            "BurgersFNO": BurgersFNO,
            "HeatFNO": HeatFNO,
            "count_parameters": count_parameters,
        }
        return mapping[name]
    if name in {"default_ckpt", "save_checkpoint"}:
        from .io import default_ckpt, save_checkpoint

        return {"default_ckpt": default_ckpt, "save_checkpoint": save_checkpoint}[name]
    raise AttributeError(name)


def load_model(
    checkpoint: str | Path | None = None,
    *,
    pde: str = "burgers",
    objective: str = "fno",
    device: str | None = None,
) -> dict:
    try:
        from .io import default_ckpt, load_checkpoint
    except ImportError as exc:  # torch missing
        return {
            "status": "not_trained",
            "model": None,
            "reason": f"torch_not_installed: {exc}",
        }
    path = checkpoint if checkpoint is not None else default_ckpt(pde, objective)
    return load_checkpoint(path, device=device)


def predict(model, u0, param):
    """Forward an operator; `param` is ν (Burgers) or α (heat)."""
    if model is None:
        raise RuntimeError("no model loaded (not_trained)")
    return model(u0, param)
