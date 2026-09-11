"""Predictive accuracy metrics: relative L2 and NMSE."""
from __future__ import annotations

import numpy as np


def relative_l2(pred: np.ndarray, truth: np.ndarray, eps: float = 1e-12) -> float:
    """||pred - truth||_2 / (||truth||_2 + eps), flattened over all axes."""
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    num = np.linalg.norm(pred - truth)
    den = np.linalg.norm(truth) + eps
    return float(num / den)


def nmse(pred: np.ndarray, truth: np.ndarray, eps: float = 1e-12) -> float:
    """Normalized mean squared error: MSE / (var(truth) + eps)."""
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    mse = np.mean((pred - truth) ** 2)
    var = np.var(truth) + eps
    return float(mse / var)


def batch_relative_l2(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Per-trajectory relative L2 (axis 0 = batch) + mean/std."""
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if pred.shape != truth.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {truth.shape}")
    scores = []
    for i in range(pred.shape[0]):
        scores.append(relative_l2(pred[i], truth[i]))
    arr = np.asarray(scores, dtype=np.float64)
    return {
        "per_traj": arr,
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "nmse": nmse(pred, truth),
    }
