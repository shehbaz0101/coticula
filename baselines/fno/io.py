"""Checkpoint I/O. Missing files stay `not_trained` — never invent weights."""
from __future__ import annotations

from pathlib import Path

import torch

from .models import BurgersFNO, HeatFNO

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = ROOT / "checkpoints"

FORMAT = "vu-bench-fno-v0"

DEFAULT_BURGERS = CHECKPOINT_DIR / "fno_burgers.pt"
DEFAULT_HEAT = CHECKPOINT_DIR / "fno_heat2d.pt"
DEFAULT_PINO_BURGERS = CHECKPOINT_DIR / "pino_burgers.pt"
DEFAULT_PINO_HEAT = CHECKPOINT_DIR / "pino_heat2d.pt"


def default_ckpt(pde: str, objective: str = "fno") -> Path:
    pde = pde.lower()
    prefix = "pino" if objective == "pino" else "fno"
    if pde in ("burgers", "burgers1d"):
        return CHECKPOINT_DIR / f"{prefix}_burgers.pt"
    if pde in ("heat", "heat2d"):
        return CHECKPOINT_DIR / f"{prefix}_heat2d.pt"
    raise ValueError(f"unknown pde {pde!r}")


def save_checkpoint(
    path: Path | str,
    model,
    *,
    objective: str,
    training: dict,
    extra: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": FORMAT,
        "objective": objective,
        "config": model.config_dict(),
        "state_dict": model.state_dict(),
        "training": training,
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)
    return path


def _build_model(config: dict):
    pde = config.get("pde")
    if pde == "burgers1d":
        return BurgersFNO(
            modes=int(config["modes"]),
            width=int(config["width"]),
            n_layers=int(config["n_layers"]),
            nt=int(config["nt"]),
            nx=int(config["nx"]),
            fc_dim=int(config.get("fc_dim", 64)),
        )
    if pde == "heat2d":
        return HeatFNO(
            modes=int(config["modes"]),
            width=int(config["width"]),
            n_layers=int(config["n_layers"]),
            nt=int(config["nt"]),
            n=int(config["n"]),
            fc_dim=int(config.get("fc_dim", 48)),
        )
    raise ValueError(f"unknown pde in checkpoint config: {pde!r}")


def load_checkpoint(
    path: Path | str | None,
    device: str | torch.device | None = None,
) -> dict:
    """Load a checkpoint. If path is missing/None, status is `not_trained`.

    Never fabricates a model or metrics when the file is absent.
    """
    device = torch.device(device or "cpu")
    if path is None:
        return {"status": "not_trained", "model": None, "path": None, "reason": "no_path"}
    path = Path(path)
    if not path.exists():
        return {
            "status": "not_trained",
            "model": None,
            "path": str(path),
            "reason": "missing_checkpoint",
        }
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        return {
            "status": "not_trained",
            "model": None,
            "path": str(path),
            "reason": "unrecognized_checkpoint",
        }
    model = _build_model(payload["config"])
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return {
        "status": "ok",
        "model": model,
        "path": str(path),
        "config": payload.get("config", {}),
        "training": payload.get("training", {}),
        "objective": payload.get("objective", "fno"),
        "extra": payload.get("extra", {}),
        "device": str(device),
    }
