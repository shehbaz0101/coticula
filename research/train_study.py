"""Fixed-budget training for the paper study. CPU, data loss only.

Hyperparameter trials and the final seeds share this loop. The validation
split is used only to pick a trial. Test trajectories are not read here.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def seed_all(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _rel_l2(pred: torch.Tensor, truth: torch.Tensor) -> float:
    num = torch.linalg.norm((pred - truth).reshape(truth.shape[0], -1), dim=1)
    den = torch.linalg.norm(truth.reshape(truth.shape[0], -1), dim=1) + 1e-12
    return float((num / den).mean())


def train_model(
    model: torch.nn.Module,
    u_train: np.ndarray,
    param_train: np.ndarray,
    u_val: np.ndarray,
    param_val: np.ndarray,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
) -> dict:
    """Adam on the data MSE. Returns the final validation relative L2."""
    seed_all(seed)
    device = torch.device("cpu")
    model.to(device)
    u_tr = torch.from_numpy(np.asarray(u_train, dtype=np.float32))
    p_tr = torch.from_numpy(np.asarray(param_train, dtype=np.float32))
    u_va = torch.from_numpy(np.asarray(u_val, dtype=np.float32))
    p_va = torch.from_numpy(np.asarray(param_val, dtype=np.float32))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(
        TensorDataset(u_tr, p_tr),
        batch_size=min(int(batch_size), len(u_tr)),
        shuffle=True,
        generator=generator,
    )
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    last_train = None
    for _epoch in range(int(epochs)):
        model.train()
        total = 0.0
        seen = 0
        for u_b, p_b in loader:
            pred = model(u_b[:, 0], p_b)
            loss = torch.mean((pred - u_b) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * u_b.shape[0]
            seen += u_b.shape[0]
        last_train = total / max(seen, 1)
    model.eval()
    with torch.no_grad():
        pred = model(u_va[:, 0], p_va)
        val = _rel_l2(pred, u_va)
    return {"val_rel_l2": val, "train_mse": float(last_train)}


@torch.no_grad()
def predict(model: torch.nn.Module, u0: np.ndarray, param: np.ndarray) -> np.ndarray:
    model.eval()
    u0_t = torch.from_numpy(np.asarray(u0, dtype=np.float32))
    p_t = torch.from_numpy(np.asarray(param, dtype=np.float32))
    return model(u0_t, p_t).cpu().numpy().astype(np.float64)
