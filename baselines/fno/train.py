"""Shared FNO / PINO training loop (CPU-friendly defaults)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from baselines.fno.io import save_checkpoint
from baselines.fno.models import BurgersFNO, HeatFNO, count_parameters
from baselines.pino.losses import LOSS_TERMS_DOC, pino_loss

ROOT = Path(__file__).resolve().parents[2]


def resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _as_f32(*arrays):
    out = [torch.from_numpy(np.asarray(a, dtype=np.float32)) for a in arrays]
    return out[0] if len(out) == 1 else out


def load_burgers_npz(path: Path | None = None) -> dict:
    path = path or (ROOT / "datasets" / "burgers" / "burgers_v0.npz")
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run python -m scripts.generate_labels")
    z = np.load(path)
    return {k: z[k] for k in z.files}


def load_heat_npz(path: Path | None = None) -> dict:
    path = path or (ROOT / "datasets" / "heat2d" / "heat2d_v0.npz")
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run python -m scripts.generate_labels")
    z = np.load(path)
    return {k: z[k] for k in z.files}


def _split_indices(n: int, eval_n: int) -> tuple[np.ndarray, np.ndarray]:
    eval_n = min(eval_n, max(1, n // 4))
    train_n = n - eval_n
    if train_n < 1:
        train_n = max(1, n - 1)
        eval_n = n - train_n
    return np.arange(0, train_n), np.arange(train_n, n)


@torch.no_grad()
def _val_rel_l2(model, u, param, device) -> float:
    model.eval()
    pred = model(u[:, 0].to(device), param.to(device)).cpu()
    num = torch.linalg.norm((pred - u).reshape(u.shape[0], -1), dim=1)
    den = torch.linalg.norm(u.reshape(u.shape[0], -1), dim=1) + 1e-12
    return float((num / den).mean())


def train_burgers(
    *,
    data: dict | None = None,
    epochs: int = 40,
    width: int = 16,
    modes: int = 8,
    n_layers: int = 3,
    batch_size: int = 8,
    lr: float = 1e-3,
    pino: bool = False,
    lambda_pde: float = 1e-3,
    lambda_ic: float = 0.0,
    eval_n: int = 16,
    seed: int = 0,
    device: str | None = None,
    out_path: Path | str | None = None,
    log_every: int = 5,
) -> dict:
    device_t = resolve_device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if data is None:
        data = load_burgers_npz()
    u_all = np.asarray(data["u"], dtype=np.float32)
    nu_all = np.asarray(data["nu"], dtype=np.float32)
    x = np.asarray(data["x"])
    t = np.asarray(data["t"])
    n_traj, ntp1, nx = u_all.shape
    nt = ntp1 - 1
    dx = float(x[1] - x[0])
    dt = float(t[1] - t[0])

    train_idx, eval_idx = _split_indices(n_traj, eval_n)
    u_tr, nu_tr = _as_f32(u_all[train_idx], nu_all[train_idx])
    u_va, nu_va = _as_f32(u_all[eval_idx], nu_all[eval_idx])

    model = BurgersFNO(modes=modes, width=width, n_layers=n_layers, nt=nt, nx=nx)
    model.to(device_t)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(u_tr, nu_tr),
        batch_size=min(batch_size, len(u_tr)),
        shuffle=True,
    )

    history: list[dict] = []
    objective = "pino" if pino else "fno"
    for epoch in range(1, epochs + 1):
        model.train()
        running = {"total": 0.0, "data": 0.0, "pde": 0.0}
        n_seen = 0
        for u_b, nu_b in loader:
            u_b = u_b.to(device_t)
            nu_b = nu_b.to(device_t)
            pred = model(u_b[:, 0], nu_b)
            terms = pino_loss(
                pred,
                u_b,
                nu_b,
                pde="burgers1d",
                dx=dx,
                dt=dt,
                lambda_pde=(lambda_pde if pino else 0.0),
                lambda_ic=(lambda_ic if pino else 0.0),
            )
            opt.zero_grad(set_to_none=True)
            terms["total"].backward()
            opt.step()
            bs = u_b.shape[0]
            n_seen += bs
            running["total"] += float(terms["total"].detach()) * bs
            running["data"] += float(terms["data"].detach()) * bs
            running["pde"] += float(terms["pde"].detach()) * bs
        row = {k: v / max(n_seen, 1) for k, v in running.items()}
        row["epoch"] = epoch
        if epoch == 1 or epoch == epochs or epoch % log_every == 0:
            row["val_rel_l2"] = _val_rel_l2(model, u_va, nu_va, device_t)
            print(
                f"[burgers {objective}] epoch {epoch}/{epochs} "
                f"data={row['data']:.4e} pde={row['pde']:.4e} "
                f"val_rel_l2={row['val_rel_l2']:.4e}"
            )
        history.append(row)

    extra = {
        "split": {
            "train_idx": train_idx.tolist(),
            "eval_idx": eval_idx.tolist(),
        },
        "dx": dx,
        "dt": dt,
        "n_parameters": count_parameters(model),
    }
    training = {
        "objective": objective,
        "pde": "burgers1d",
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "seed": seed,
        "lambda_pde": float(lambda_pde if pino else 0.0),
        "lambda_ic": float(lambda_ic if pino else 0.0),
        "loss_terms": LOSS_TERMS_DOC,
        "history_tail": history[-min(5, len(history)) :],
        "final_train_data_mse": history[-1]["data"] if history else None,
        "final_val_rel_l2": history[-1].get("val_rel_l2"),
    }
    path = Path(out_path) if out_path else None
    if path is not None:
        save_checkpoint(path, model, objective=objective, training=training, extra=extra)
        print(f"Wrote {path} ({count_parameters(model)} params)")
    return {
        "model": model,
        "path": str(path) if path else None,
        "training": training,
        "extra": extra,
        "history": history,
    }


def train_heat(
    *,
    data: dict | None = None,
    epochs: int = 25,
    width: int = 12,
    modes: int = 6,
    n_layers: int = 2,
    batch_size: int = 4,
    lr: float = 1e-3,
    pino: bool = False,
    lambda_pde: float = 1e-3,
    lambda_ic: float = 0.0,
    eval_n: int = 12,
    seed: int = 1,
    device: str | None = None,
    out_path: Path | str | None = None,
    log_every: int = 5,
) -> dict:
    device_t = resolve_device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if data is None:
        data = load_heat_npz()
    u_all = np.asarray(data["u"], dtype=np.float32)
    a_all = np.asarray(data["alpha"], dtype=np.float32)
    x = np.asarray(data["x"])
    t = np.asarray(data["t"])
    n_traj, ntp1, n, _ = u_all.shape
    nt = ntp1 - 1
    dx = float(x[1] - x[0])
    dt = float(t[1] - t[0])

    train_idx, eval_idx = _split_indices(n_traj, eval_n)
    u_tr, a_tr = _as_f32(u_all[train_idx], a_all[train_idx])
    u_va, a_va = _as_f32(u_all[eval_idx], a_all[eval_idx])

    model = HeatFNO(modes=modes, width=width, n_layers=n_layers, nt=nt, n=n)
    model.to(device_t)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(u_tr, a_tr),
        batch_size=min(batch_size, len(u_tr)),
        shuffle=True,
    )

    history: list[dict] = []
    objective = "pino" if pino else "fno"
    for epoch in range(1, epochs + 1):
        model.train()
        running = {"total": 0.0, "data": 0.0, "pde": 0.0}
        n_seen = 0
        for u_b, a_b in loader:
            u_b = u_b.to(device_t)
            a_b = a_b.to(device_t)
            pred = model(u_b[:, 0], a_b)
            terms = pino_loss(
                pred,
                u_b,
                a_b,
                pde="heat2d",
                dx=dx,
                dt=dt,
                lambda_pde=(lambda_pde if pino else 0.0),
                lambda_ic=(lambda_ic if pino else 0.0),
            )
            opt.zero_grad(set_to_none=True)
            terms["total"].backward()
            opt.step()
            bs = u_b.shape[0]
            n_seen += bs
            running["total"] += float(terms["total"].detach()) * bs
            running["data"] += float(terms["data"].detach()) * bs
            running["pde"] += float(terms["pde"].detach()) * bs
        row = {k: v / max(n_seen, 1) for k, v in running.items()}
        row["epoch"] = epoch
        if epoch == 1 or epoch == epochs or epoch % log_every == 0:
            row["val_rel_l2"] = _val_rel_l2(model, u_va, a_va, device_t)
            print(
                f"[heat2d {objective}] epoch {epoch}/{epochs} "
                f"data={row['data']:.4e} pde={row['pde']:.4e} "
                f"val_rel_l2={row['val_rel_l2']:.4e}"
            )
        history.append(row)

    extra = {
        "split": {
            "train_idx": train_idx.tolist(),
            "eval_idx": eval_idx.tolist(),
        },
        "dx": dx,
        "dt": dt,
        "n_parameters": count_parameters(model),
    }
    training = {
        "objective": objective,
        "pde": "heat2d",
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "seed": seed,
        "lambda_pde": float(lambda_pde if pino else 0.0),
        "lambda_ic": float(lambda_ic if pino else 0.0),
        "loss_terms": LOSS_TERMS_DOC,
        "history_tail": history[-min(5, len(history)) :],
        "final_train_data_mse": history[-1]["data"] if history else None,
        "final_val_rel_l2": history[-1].get("val_rel_l2"),
    }
    path = Path(out_path) if out_path else None
    if path is not None:
        save_checkpoint(path, model, objective=objective, training=training, extra=extra)
        print(f"Wrote {path} ({count_parameters(model)} params)")
    return {
        "model": model,
        "path": str(path) if path else None,
        "training": training,
        "extra": extra,
        "history": history,
    }
