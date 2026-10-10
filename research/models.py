"""Small FNO, U-Net, and DeepONet operators for the paper study.

These wrappers are opt-in. They do not replace the shipped checkpoints or the
default training scripts. Every model maps (initial condition, coefficient) to
a full trajectory, locks t = 0 to the initial condition, and, for heat, keeps
homogeneous Dirichlet boundaries.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.fno.models import BurgersFNO, HeatFNO, count_parameters, pack_burgers_input, pack_heat_input


def build_model(arch: str, pde: str, grid: dict, trial: dict) -> nn.Module:
    arch = arch.lower()
    pde = pde.lower()
    if pde in ("burgers", "burgers1d"):
        nt = int(grid["nt"])
        nx = int(grid["nx"])
        if arch == "fno":
            return BurgersFNO(
                modes=int(trial["modes"]),
                width=int(trial["width"]),
                n_layers=int(trial["n_layers"]),
                nt=nt,
                nx=nx,
                fc_dim=int(trial["fc_dim"]),
            )
        if arch == "unet":
            return BurgersUNet(nt=nt, nx=nx, width=int(trial["width"]))
        if arch == "deeponet":
            return BurgersDeepONet(
                nt=nt, nx=nx, rank=int(trial["rank"]), hidden=int(trial["hidden"])
            )
    if pde in ("heat", "heat2d"):
        nt = int(grid["nt"])
        n = int(grid["n"])
        if arch == "fno":
            return HeatFNO(
                modes=int(trial["modes"]),
                width=int(trial["width"]),
                n_layers=int(trial["n_layers"]),
                nt=nt,
                n=n,
                fc_dim=int(trial["fc_dim"]),
            )
        if arch == "unet":
            return HeatUNet(nt=nt, n=n, width=int(trial["width"]))
        if arch == "deeponet":
            return HeatDeepONet(
                nt=nt, n=n, rank=int(trial["rank"]), hidden=int(trial["hidden"])
            )
    raise ValueError(f"unknown arch/pde {arch}/{pde}")


def _circ_conv1d(conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
    pad = conv.kernel_size[0] // 2
    return conv(F.pad(x, (pad, pad), mode="circular"))


def _zero_conv2d(conv: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    pad = conv.kernel_size[0] // 2
    return conv(F.pad(x, (pad, pad, pad, pad), mode="constant"))


class BurgersUNet(nn.Module):
    """One-level 1D U-Net. Circular padding respects the periodic interval."""

    def __init__(self, nt: int, nx: int, width: int):
        super().__init__()
        self.nt = nt
        self.nx = nx
        self.width = width
        self.enc = nn.Conv1d(3, width, 3, padding=0)
        self.down = nn.Conv1d(width, width * 2, 3, padding=0)
        self.dec = nn.Conv1d(width * 2 + width, width, 3, padding=0)
        self.out = nn.Conv1d(width, nt + 1, 1)

    def config_dict(self) -> dict:
        return {"arch": "unet1d", "pde": "burgers1d", "nt": self.nt, "nx": self.nx, "width": self.width}

    def forward(self, u0: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        x = pack_burgers_input(u0, nu).permute(0, 2, 1)
        skip = F.gelu(_circ_conv1d(self.enc, x))
        z = F.avg_pool1d(skip, kernel_size=2)
        z = F.gelu(_circ_conv1d(self.down, z))
        z = F.interpolate(z, size=skip.shape[-1], mode="nearest")
        y = F.gelu(_circ_conv1d(self.dec, torch.cat([z, skip], dim=1)))
        pred = self.out(y)
        return torch.cat([u0.unsqueeze(1), pred[:, 1:]], dim=1)


class HeatUNet(nn.Module):
    """One-level 2D U-Net with zero padding. Output walls are set to zero."""

    def __init__(self, nt: int, n: int, width: int):
        super().__init__()
        self.nt = nt
        self.n = n
        self.width = width
        self.enc = nn.Conv2d(4, width, 3, padding=0)
        self.down = nn.Conv2d(width, width * 2, 3, padding=0)
        self.dec = nn.Conv2d(width * 2 + width, width, 3, padding=0)
        self.out = nn.Conv2d(width, nt + 1, 1)

    def config_dict(self) -> dict:
        return {"arch": "unet2d", "pde": "heat2d", "nt": self.nt, "n": self.n, "width": self.width}

    def forward(self, u0: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        x = pack_heat_input(u0, alpha).permute(0, 3, 1, 2)
        skip = F.gelu(_zero_conv2d(self.enc, x))
        z = F.avg_pool2d(skip, kernel_size=2)
        z = F.gelu(_zero_conv2d(self.down, z))
        z = F.interpolate(z, size=skip.shape[-2:], mode="nearest")
        y = F.gelu(_zero_conv2d(self.dec, torch.cat([z, skip], dim=1)))
        pred = self.out(y)
        pred = torch.cat([u0.unsqueeze(1), pred[:, 1:]], dim=1)
        pred = pred.clone()
        pred[:, :, 0, :] = 0.0
        pred[:, :, -1, :] = 0.0
        pred[:, :, :, 0] = 0.0
        pred[:, :, :, -1] = 0.0
        return pred


class BurgersDeepONet(nn.Module):
    """Branch on the initial condition, trunk on (t, x, nu)."""

    def __init__(self, nt: int, nx: int, rank: int, hidden: int):
        super().__init__()
        self.nt = nt
        self.nx = nx
        self.rank = rank
        self.hidden = hidden
        self.branch = nn.Sequential(
            nn.Linear(nx, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, rank),
        )
        # t, scaled nu, and Fourier features of x (k = 1..4). Fixed, not tuned on the test set.
        self.trunk_in = 2 + 8
        self.trunk = nn.Sequential(
            nn.Linear(self.trunk_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, rank),
        )
        self.bias = nn.Parameter(torch.zeros(()))
        # Fixed scale from the protocol viscosity range, not fit on data.
        self.param_scale = 0.05

    def config_dict(self) -> dict:
        return {
            "arch": "deeponet1d",
            "pde": "burgers1d",
            "nt": self.nt,
            "nx": self.nx,
            "rank": self.rank,
            "hidden": self.hidden,
        }

    def forward(self, u0: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        batch, nx = u0.shape
        branch = self.branch(u0)
        t = torch.linspace(0.0, 1.0, self.nt + 1, device=u0.device, dtype=u0.dtype)
        x = torch.linspace(0.0, 1.0, nx, device=u0.device, dtype=u0.dtype)
        tt, xx = torch.meshgrid(t, x, indexing="ij")
        spatial = [tt]
        for k in range(1, 5):
            spatial.append(torch.sin(2.0 * torch.pi * k * xx))
            spatial.append(torch.cos(2.0 * torch.pi * k * xx))
        base = torch.stack(spatial, dim=-1)  # (T, X, 9)
        nu_feat = (nu[:, None, None] / self.param_scale).unsqueeze(-1)
        coords = torch.cat([base.unsqueeze(0).expand(batch, -1, -1, -1), nu_feat.expand(batch, self.nt + 1, nx, 1)], dim=-1)
        pred = (self.trunk(coords) * branch[:, None, None, :]).sum(dim=-1) + self.bias
        return torch.cat([u0.unsqueeze(1), pred[:, 1:]], dim=1)


class HeatDeepONet(nn.Module):
    """Branch on the flattened initial condition, trunk on (t, x, y, alpha)."""

    def __init__(self, nt: int, n: int, rank: int, hidden: int):
        super().__init__()
        self.nt = nt
        self.n = n
        self.rank = rank
        self.hidden = hidden
        self.branch = nn.Sequential(
            nn.Linear(n * n, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, rank),
        )
        # t, scaled alpha, sine features of x and of y (p = 1..3).
        self.trunk_in = 2 + 6
        self.trunk = nn.Sequential(
            nn.Linear(self.trunk_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, rank),
        )
        self.bias = nn.Parameter(torch.zeros(()))
        self.param_scale = 0.2

    def config_dict(self) -> dict:
        return {
            "arch": "deeponet2d",
            "pde": "heat2d",
            "nt": self.nt,
            "n": self.n,
            "rank": self.rank,
            "hidden": self.hidden,
        }

    def forward(self, u0: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        batch, n, _n2 = u0.shape
        branch = self.branch(u0.reshape(batch, n * n))
        t = torch.linspace(0.0, 1.0, self.nt + 1, device=u0.device, dtype=u0.dtype)
        x = torch.linspace(0.0, 1.0, n, device=u0.device, dtype=u0.dtype)
        tt, xx, yy = torch.meshgrid(t, x, x, indexing="ij")
        spatial = [tt]
        for p in range(1, 4):
            spatial.append(torch.sin(torch.pi * p * xx))
            spatial.append(torch.sin(torch.pi * p * yy))
        base = torch.stack(spatial, dim=-1)
        a_feat = (alpha[:, None, None, None] / self.param_scale).unsqueeze(-1)
        coords = torch.cat(
            [base.unsqueeze(0).expand(batch, -1, -1, -1, -1), a_feat.expand(batch, self.nt + 1, n, n, 1)],
            dim=-1,
        )
        pred = (self.trunk(coords) * branch[:, None, None, None, :]).sum(dim=-1) + self.bias
        pred = torch.cat([u0.unsqueeze(1), pred[:, 1:]], dim=1)
        pred = pred.clone()
        pred[:, :, 0, :] = 0.0
        pred[:, :, -1, :] = 0.0
        pred[:, :, :, 0] = 0.0
        pred[:, :, :, -1] = 0.0
        return pred


def parameter_count(model: nn.Module) -> int:
    return count_parameters(model)
