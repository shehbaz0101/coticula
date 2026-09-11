"""Spectral convolution layers after Li et al., ICLR 2021 (FNO).

Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart, Anandkumar.
"Fourier Neural Operator for Parametric Partial Differential Equations."
https://arxiv.org/abs/2010.08895
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SpectralConv1d(nn.Module):
    """1D Fourier integral operator: keep `modes` rFFT modes, linearly mix channels.

    Input/output: (batch, channels, n).
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        # Real view of complex weights: (in, out, modes, 2)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, 2)
        )

    def _mix(self, x_ft: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # x_ft: (B, in, modes) complex; weights: (in, out, modes, 2)
        w = torch.view_as_complex(weights.contiguous())
        return torch.einsum("bim,iom->bom", x_ft, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, n = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        n_modes = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(
            b, self.out_channels, x_ft.shape[-1], device=x.device, dtype=torch.cfloat
        )
        out_ft[:, :, :n_modes] = self._mix(
            x_ft[:, :, :n_modes], self.weights[:, :, :n_modes]
        )
        return torch.fft.irfft(out_ft, n=n, dim=-1)


class SpectralConv2d(nn.Module):
    """2D Fourier integral operator (Li et al. 2021, §4).

    Keeps a `modes_h × modes_w` corner of rFFT2 plus the conjugate-high strip
    along the first frequency axis. Input/output: (batch, channels, h, w).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_h: int,
        modes_w: int | None = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_h = modes_h
        self.modes_w = modes_h if modes_w is None else modes_w
        scale = 1.0 / (in_channels * out_channels)
        # Two corners: low-low and high-low (negative ky frequencies).
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_h, self.modes_w, 2)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_h, self.modes_w, 2)
        )

    def _mix(self, x_ft: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        w = torch.view_as_complex(weights.contiguous())
        return torch.einsum("bihw,iohw->bohw", x_ft, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        mh = min(self.modes_h, x_ft.shape[-2] // 2)
        mw = min(self.modes_w, x_ft.shape[-1])
        out_ft = torch.zeros(
            b,
            self.out_channels,
            x_ft.shape[-2],
            x_ft.shape[-1],
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, :mh, :mw] = self._mix(
            x_ft[:, :, :mh, :mw], self.weights1[:, :, :mh, :mw]
        )
        if mh > 0:
            out_ft[:, :, -mh:, :mw] = self._mix(
                x_ft[:, :, -mh:, :mw], self.weights2[:, :, :mh, :mw]
            )
        return torch.fft.irfft2(out_ft, s=(h, w), dim=(-2, -1))
