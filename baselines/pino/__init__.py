"""PINO-style path: same FNO trunk + data/PDE residual loss.

Li et al., "Physics-Informed Neural Operator for Learning Partial
Differential Equations," arXiv:2111.03794.

`STATUS` is `implemented` (loss + train path in-repo). Metrics appear only
when a PINO checkpoint is loaded. No fabricated scores. Torch is lazy.
"""
from __future__ import annotations

from pathlib import Path

STATUS = "implemented"

LOSS_TERMS_DOC = {
    "L": "L_data + lambda_pde * L_pde + lambda_ic * L_ic",
    "L_data": "MSE(pred, classical_label) over the full trajectory",
    "L_pde_burgers": "MSE of FD residual u_t + d(u^2)/dx/2 - nu u_xx (periodic, midpoint)",
    "L_pde_heat": "MSE of FD residual u_t - alpha laplace(u) on Dirichlet interior",
    "L_ic": "MSE(pred[t=0], u0); default weight 0 because operators lock the IC",
    "citation": "Li et al., Physics-Informed Neural Operator, arXiv:2111.03794",
}

__all__ = [
    "STATUS",
    "LOSS_TERMS_DOC",
    "pino_loss",
    "load_model",
]


def __getattr__(name: str):
    if name == "pino_loss":
        from baselines.pino.losses import pino_loss

        return pino_loss
    raise AttributeError(name)


def load_model(
    checkpoint: str | Path | None = None,
    *,
    pde: str = "burgers",
    device: str | None = None,
) -> dict:
    try:
        from baselines.fno.io import default_ckpt, load_checkpoint
    except ImportError as exc:
        return {
            "status": "not_trained",
            "model": None,
            "reason": f"torch_not_installed: {exc}",
        }
    path = checkpoint if checkpoint is not None else default_ckpt(pde, "pino")
    return load_checkpoint(path, device=device)
