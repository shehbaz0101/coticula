"""White and time-varying low-mode errors at a matched relative L2.

Smooth errors are time-varying: each spatial coefficient is a trigonometric
polynomial in time of degree ``n_time_modes`` (including the constant mode).
Heat errors are zero on the Dirichlet boundary. Burgers errors are periodic.
"""
from __future__ import annotations

import numpy as np

from metrics.predict import relative_l2


def scale_to_rel(error: np.ndarray, reference: np.ndarray, epsilon: float) -> np.ndarray:
    """Rescale ``error`` so ||error||_2 / ||reference||_2 equals ``epsilon``."""
    ref_norm = float(np.linalg.norm(reference))
    err_norm = float(np.linalg.norm(error))
    if ref_norm <= 0 or err_norm <= 0:
        raise ValueError("cannot match rel-L2 of a zero field or a zero error")
    return error * (float(epsilon) * ref_norm / err_norm)


def white_error(
    reference: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    *,
    dirichlet: bool,
) -> np.ndarray:
    error = rng.normal(size=reference.shape)
    if dirichlet:
        error = error.copy()
        error[..., 0, :] = 0.0
        error[..., -1, :] = 0.0
        error[..., :, 0] = 0.0
        error[..., :, -1] = 0.0
    return scale_to_rel(error, reference, epsilon)


def _time_amplitude(rng: np.random.Generator, t: np.ndarray, n_time_modes: int) -> np.ndarray:
    """Trigonometric polynomial of degree n_time_modes, including m = 0."""
    t = np.asarray(t, dtype=np.float64)
    span = float(t[-1] - t[0])
    acc = np.zeros_like(t)
    for m in range(int(n_time_modes) + 1):
        acc += float(rng.normal()) * np.cos(2.0 * np.pi * m * (t - t[0]) / span)
        acc += float(rng.normal()) * np.sin(2.0 * np.pi * m * (t - t[0]) / span)
    return acc


def smooth_error_burgers(
    reference: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    *,
    k_smooth: int,
    n_time_modes: int,
) -> np.ndarray:
    """Time-varying Fourier series with 1 <= |k| <= k_smooth. Real-valued."""
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    error = np.zeros_like(reference, dtype=np.float64)
    xx = x[None, :]
    for k in range(1, int(k_smooth) + 1):
        ak = _time_amplitude(rng, t, n_time_modes)
        bk = _time_amplitude(rng, t, n_time_modes)
        error += ak[:, None] * np.cos(k * xx) + bk[:, None] * np.sin(k * xx)
    return scale_to_rel(error, reference, epsilon)


def smooth_error_heat(
    reference: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    *,
    k_smooth: int,
    n_time_modes: int,
) -> np.ndarray:
    """Time-varying sine series, 1 <= p,q <= k_smooth. Zero on the boundary."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    length = float(x[-1] - x[0])
    error = np.zeros_like(reference, dtype=np.float64)
    for p in range(1, int(k_smooth) + 1):
        sx = np.sin(p * np.pi * x / length)
        for q in range(1, int(k_smooth) + 1):
            sy = np.sin(q * np.pi * y / length)
            ak = _time_amplitude(rng, t, n_time_modes)
            error += ak[:, None, None] * sx[:, None] * sy[None, :]
    return scale_to_rel(error, reference, epsilon)


def realized_rel(error: np.ndarray, reference: np.ndarray) -> float:
    return relative_l2(reference + error, reference)
