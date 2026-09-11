"""Exam 1–3 evaluation for a trained FNO / PINO operator."""
from __future__ import annotations

import numpy as np
import torch

from metrics.conserve import audit_burgers, audit_heat
from metrics.counterfactual import (
    grade_burgers_cf_prediction,
    grade_heat_cf_prediction,
)
from metrics.predict import batch_relative_l2


@torch.no_grad()
def _predict_burgers(model, u0: np.ndarray, nu: np.ndarray, device) -> np.ndarray:
    u0_t = torch.from_numpy(np.asarray(u0, dtype=np.float32)).to(device)
    nu_t = torch.from_numpy(np.asarray(nu, dtype=np.float32)).to(device)
    if u0_t.ndim == 1:
        u0_t = u0_t.unsqueeze(0)
        nu_t = nu_t.reshape(1)
        pred = model(u0_t, nu_t).cpu().numpy()[0]
        return pred
    return model(u0_t, nu_t).cpu().numpy()


@torch.no_grad()
def _predict_heat(model, u0: np.ndarray, alpha: np.ndarray, device) -> np.ndarray:
    u0_t = torch.from_numpy(np.asarray(u0, dtype=np.float32)).to(device)
    a_t = torch.from_numpy(np.asarray(alpha, dtype=np.float32)).to(device)
    if u0_t.ndim == 2:
        u0_t = u0_t.unsqueeze(0)
        a_t = a_t.reshape(1)
        return model(u0_t, a_t).cpu().numpy()[0]
    return model(u0_t, a_t).cpu().numpy()


def eval_burgers_operator(
    model,
    data: dict,
    *,
    idx: np.ndarray | list[int] | None = None,
    n_eval: int = 16,
    device: str | torch.device = "cpu",
    model_name: str = "fno",
) -> dict:
    u = data["u"]
    nu = data["nu"]
    x = data["x"]
    t = data["t"]
    if idx is None:
        idx = np.arange(max(0, len(u) - n_eval), len(u))
    idx = np.asarray(idx, dtype=int)
    u_e = u[idx]
    nu_e = nu[idx]
    nx = len(x)
    nt = len(t) - 1
    L = float(x[-1] + (x[1] - x[0]))
    T = float(t[-1])
    dx = float(x[1] - x[0])
    dt = float(t[1] - t[0])

    model.eval()
    pred = _predict_burgers(model, u_e[:, 0], nu_e, device)
    pred_metrics = batch_relative_l2(pred, u_e)
    conserve = audit_burgers(pred, dx=dx, dt=dt, nu=nu_e)

    pred_cf = _predict_burgers(model, u_e[0, 0], np.asarray(float(nu_e[0]) * 2.0), device)
    cf = grade_burgers_cf_prediction(
        pred_cf,
        u0=u_e[0, 0],
        nu_orig=float(nu_e[0]),
        nu_cf=float(nu_e[0]) * 2.0,
        nx=nx,
        nt=nt,
        L=L,
        T=T,
    )
    pred_base = pred[0]
    cf["rel_l2_traj_delta_model"] = float(
        np.linalg.norm(pred_cf - pred_base) / (np.linalg.norm(pred_base) + 1e-12)
    )

    return {
        "model": model_name,
        "status": "ok",
        "n_eval": int(len(idx)),
        "eval_idx": idx.tolist(),
        "predict": {
            "rel_l2_mean": pred_metrics["mean"],
            "rel_l2_std": pred_metrics["std"],
            "nmse": pred_metrics["nmse"],
        },
        "conserve": conserve,
        "counterfactual": cf,
    }


def eval_heat_operator(
    model,
    data: dict,
    *,
    idx: np.ndarray | list[int] | None = None,
    n_eval: int = 12,
    device: str | torch.device = "cpu",
    model_name: str = "fno",
) -> dict:
    u = data["u"]
    alpha = data["alpha"]
    x = data["x"]
    t = data["t"]
    if idx is None:
        idx = np.arange(max(0, len(u) - n_eval), len(u))
    idx = np.asarray(idx, dtype=int)
    u_e = u[idx]
    a_e = alpha[idx]
    n = len(x)
    nt = len(t) - 1
    L = float(x[-1])
    T = float(t[-1])
    dx = float(x[1] - x[0])
    dt = float(t[1] - t[0])

    model.eval()
    pred = _predict_heat(model, u_e[:, 0], a_e, device)
    pred_metrics = batch_relative_l2(pred, u_e)
    conserve = audit_heat(pred, dx=dx, dt=dt, alpha=a_e)

    pred_cf = _predict_heat(model, u_e[0, 0], np.asarray(float(a_e[0]) * 1.5), device)
    cf = grade_heat_cf_prediction(
        pred_cf,
        u0=u_e[0, 0],
        alpha_orig=float(a_e[0]),
        alpha_cf=float(a_e[0]) * 1.5,
        n=n,
        nt=nt,
        L=L,
        T=T,
    )
    pred_base = pred[0]
    cf["rel_l2_traj_delta_model"] = float(
        np.linalg.norm(pred_cf - pred_base) / (np.linalg.norm(pred_base) + 1e-12)
    )

    return {
        "model": model_name,
        "status": "ok",
        "n_eval": int(len(idx)),
        "eval_idx": idx.tolist(),
        "predict": {
            "rel_l2_mean": pred_metrics["mean"],
            "rel_l2_std": pred_metrics["std"],
            "nmse": pred_metrics["nmse"],
        },
        "conserve": conserve,
        "counterfactual": cf,
    }
