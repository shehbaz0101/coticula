"""Tiny laptop-scale FNO operators for Burgers-1D and heat-2D.

Architecture follows Li et al., ICLR 2021: lift → stacked (spectral conv +
pointwise skip) → project. Time is lifted as output channels so one forward
pass yields a full trajectory on the fixed Coticula grids.

The initial condition is copied onto t=0 of the output (the operator predicts
the *evolution*; IC is given). Heat-2D Dirichlet boundaries are zeroed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import SpectralConv1d, SpectralConv2d


class FNO1d(nn.Module):
    """Generic 1D FNO: (B, n, in_ch) → (B, n, out_ch)."""

    def __init__(
        self,
        modes: int,
        width: int,
        in_channels: int,
        out_channels: int,
        n_layers: int = 3,
        fc_dim: int = 64,
    ):
        super().__init__()
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        self.fc0 = nn.Linear(in_channels, width)
        self.spec = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(n_layers)]
        )
        self.w = nn.ModuleList(
            [nn.Conv1d(width, width, kernel_size=1) for _ in range(n_layers)]
        )
        self.fc1 = nn.Linear(width, fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n, in_ch)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        for spec, skip in zip(self.spec, self.w):
            x = F.gelu(spec(x) + skip(x))
        x = x.permute(0, 2, 1)
        return self.fc2(F.gelu(self.fc1(x)))


class FNO2d(nn.Module):
    """Generic 2D FNO: (B, h, w, in_ch) → (B, h, w, out_ch)."""

    def __init__(
        self,
        modes: int,
        width: int,
        in_channels: int,
        out_channels: int,
        n_layers: int = 3,
        fc_dim: int = 64,
    ):
        super().__init__()
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        self.fc0 = nn.Linear(in_channels, width)
        self.spec = nn.ModuleList(
            [SpectralConv2d(width, width, modes) for _ in range(n_layers)]
        )
        self.w = nn.ModuleList(
            [nn.Conv2d(width, width, kernel_size=1) for _ in range(n_layers)]
        )
        self.fc1 = nn.Linear(width, fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        for spec, skip in zip(self.spec, self.w):
            x = F.gelu(spec(x) + skip(x))
        x = x.permute(0, 2, 3, 1)
        return self.fc2(F.gelu(self.fc1(x)))


def pack_burgers_input(u0: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    """Stack IC, viscosity, and a unit-interval x-grid. Shapes: (B, nx), (B,) → (B, nx, 3)."""
    b, nx = u0.shape
    nu_ch = nu.reshape(b, 1).expand(b, nx)
    grid = torch.linspace(0.0, 1.0, nx, device=u0.device, dtype=u0.dtype)
    grid = grid.reshape(1, nx).expand(b, nx)
    return torch.stack((u0, nu_ch, grid), dim=-1)


def pack_heat_input(u0: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Stack IC, diffusivity, x-grid, y-grid. (B, n, n), (B,) → (B, n, n, 4)."""
    b, n, _ = u0.shape
    a_ch = alpha.reshape(b, 1, 1).expand(b, n, n)
    xs = torch.linspace(0.0, 1.0, n, device=u0.device, dtype=u0.dtype)
    xg = xs.reshape(1, n, 1).expand(b, n, n)
    yg = xs.reshape(1, 1, n).expand(b, n, n)
    return torch.stack((u0, a_ch, xg, yg), dim=-1)


class BurgersFNO(nn.Module):
    """IC + ν → full Burgers trajectory (B, nt+1, nx)."""

    in_channels = 3

    def __init__(
        self,
        modes: int = 8,
        width: int = 16,
        n_layers: int = 3,
        nt: int = 80,
        nx: int = 64,
        fc_dim: int = 64,
    ):
        super().__init__()
        self.nt = nt
        self.nx = nx
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        self.fc_dim = fc_dim
        self.fno = FNO1d(
            modes=modes,
            width=width,
            in_channels=self.in_channels,
            out_channels=nt + 1,
            n_layers=n_layers,
            fc_dim=fc_dim,
        )

    def config_dict(self) -> dict:
        return {
            "arch": "fno1d_time_channels",
            "pde": "burgers1d",
            "modes": self.modes,
            "width": self.width,
            "n_layers": self.n_layers,
            "nt": self.nt,
            "nx": self.nx,
            "fc_dim": self.fc_dim,
            "in_channels": self.in_channels,
            "out_channels": self.nt + 1,
        }

    def forward(self, u0: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        x = pack_burgers_input(u0, nu)
        pred = self.fno(x).permute(0, 2, 1)  # (B, nt+1, nx)
        # IC is observed — lock t=0 so exams score the predicted evolution.
        pred = torch.cat([u0.unsqueeze(1), pred[:, 1:]], dim=1)
        return pred


class HeatFNO(nn.Module):
    """IC + α → full heat-2D trajectory (B, nt+1, n, n). Dirichlet walls zeroed."""

    in_channels = 4

    def __init__(
        self,
        modes: int = 6,
        width: int = 12,
        n_layers: int = 2,
        nt: int = 40,
        n: int = 32,
        fc_dim: int = 48,
    ):
        super().__init__()
        self.nt = nt
        self.n = n
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        self.fc_dim = fc_dim
        self.fno = FNO2d(
            modes=modes,
            width=width,
            in_channels=self.in_channels,
            out_channels=nt + 1,
            n_layers=n_layers,
            fc_dim=fc_dim,
        )

    def config_dict(self) -> dict:
        return {
            "arch": "fno2d_time_channels",
            "pde": "heat2d",
            "modes": self.modes,
            "width": self.width,
            "n_layers": self.n_layers,
            "nt": self.nt,
            "n": self.n,
            "fc_dim": self.fc_dim,
            "in_channels": self.in_channels,
            "out_channels": self.nt + 1,
        }

    def forward(self, u0: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        x = pack_heat_input(u0, alpha)
        pred = self.fno(x).permute(0, 3, 1, 2)  # (B, nt+1, n, n)
        pred = torch.cat([u0.unsqueeze(1), pred[:, 1:]], dim=1)
        pred = pred.clone()
        pred[:, :, 0, :] = 0.0
        pred[:, :, -1, :] = 0.0
        pred[:, :, :, 0] = 0.0
        pred[:, :, :, -1] = 0.0
        return pred


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
