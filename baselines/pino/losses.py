"""PINO-style losses: data fit + PDE residual (Li et al., arXiv:2111.03794).

Total objective
---------------
    L = L_data + λ_pde * L_pde + λ_ic * L_ic

Terms
-----
L_data  Mean squared error vs classical FD labels (supervised operator fit).

L_pde   Mean squared discrete PDE residual of the *prediction* (physics).
        Burgers (periodic, midpoint in time — same stencil as metrics.conserve)::

            R = u_t + ∂x(u²)/2 − ν u_xx
              ≈ (u^{n+1}-u^n)/dt + (u₊² − u₋²)/(4 dx) − ν (u₊ − 2u + u₋)/dx²

        Heat (Dirichlet interior, same stencil as metrics.conserve)::

            R = u_t − α ∇²u
              ≈ (u^{n+1}-u^n)/dt − α (u_{i±1,j} + u_{i,j±1} − 4u)/dx²

        Boundaries of the heat residual are masked to 0.

L_ic    MSE of predicted t=0 vs the given IC. Default λ_ic=0 because BurgersFNO /
        HeatFNO lock t=0 to the IC, so this term is identically zero unless that
        lock is disabled.

FNO-only training is the special case λ_pde = λ_ic = 0.
Default λ_pde = 1e-3 (light regularizer: raw FD residuals are O(1)–O(100)
while data MSE is O(10^{-2}), so λ=0.1 drowns L_data).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def data_mse(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, truth)


def ic_mse(pred: torch.Tensor, u0: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred[:, 0], u0)


def burgers_pde_residual_mse(
    u: torch.Tensor,
    dx: float,
    dt: float,
    nu: torch.Tensor,
) -> torch.Tensor:
    """Mean squared Burgers residual on a predicted trajectory.

    u: (B, nt+1, nx), nu: (B,)
    """
    u_t = (u[:, 1:] - u[:, :-1]) / dt
    u_mid = 0.5 * (u[:, 1:] + u[:, :-1])
    u_p = torch.roll(u_mid, -1, dims=-1)
    u_m = torch.roll(u_mid, 1, dims=-1)
    adv = (u_p**2 - u_m**2) / (4.0 * dx)
    visc = nu.reshape(-1, 1, 1) * (u_p - 2.0 * u_mid + u_m) / (dx**2)
    residual = u_t + adv - visc
    return (residual**2).mean()


def heat_pde_residual_mse(
    u: torch.Tensor,
    dx: float,
    dt: float,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Mean squared heat residual on interior nodes.

    u: (B, nt+1, n, n), alpha: (B,)
    """
    u_t = (u[:, 1:] - u[:, :-1]) / dt
    u_mid = 0.5 * (u[:, 1:] + u[:, :-1])
    lap = (
        torch.roll(u_mid, 1, dims=-2)
        + torch.roll(u_mid, -1, dims=-2)
        + torch.roll(u_mid, 1, dims=-1)
        + torch.roll(u_mid, -1, dims=-1)
        - 4.0 * u_mid
    ) / (dx**2)
    residual = u_t - alpha.reshape(-1, 1, 1, 1) * lap
    residual = residual.clone()
    residual[:, :, 0, :] = 0.0
    residual[:, :, -1, :] = 0.0
    residual[:, :, :, 0] = 0.0
    residual[:, :, :, -1] = 0.0
    return (residual[:, :, 1:-1, 1:-1] ** 2).mean()


def pino_loss(
    pred: torch.Tensor,
    truth: torch.Tensor,
    param: torch.Tensor,
    *,
    pde: str,
    dx: float,
    dt: float,
    lambda_pde: float = 1e-3,
    lambda_ic: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Return a dict of scalar tensors: data, pde, ic, total (plus λ weights)."""
    l_data = data_mse(pred, truth)
    pde = pde.lower()
    if pde in ("burgers", "burgers1d"):
        l_pde = burgers_pde_residual_mse(pred, dx, dt, param)
    elif pde in ("heat", "heat2d"):
        l_pde = heat_pde_residual_mse(pred, dx, dt, param)
    else:
        raise ValueError(f"unknown pde {pde!r}")
    l_ic = ic_mse(pred, truth[:, 0])
    total = l_data + float(lambda_pde) * l_pde + float(lambda_ic) * l_ic
    return {
        "data": l_data,
        "pde": l_pde,
        "ic": l_ic,
        "total": total,
        "lambda_pde": pred.new_tensor(float(lambda_pde)),
        "lambda_ic": pred.new_tensor(float(lambda_ic)),
    }


LOSS_TERMS_DOC = {
    "L": "L_data + lambda_pde * L_pde + lambda_ic * L_ic",
    "L_data": "MSE(pred, classical_label) over the full trajectory",
    "L_pde_burgers": "MSE of FD residual u_t + d(u^2)/dx/2 - nu u_xx (periodic, midpoint)",
    "L_pde_heat": "MSE of FD residual u_t - alpha laplace(u) on Dirichlet interior",
    "L_ic": "MSE(pred[t=0], u0); default weight 0 because operators lock the IC",
    "lambda_pde_default": "1e-3 (light; residual amplitude >> data MSE)",
    "citation": "Li et al., Physics-Informed Neural Operator, arXiv:2111.03794",
}
