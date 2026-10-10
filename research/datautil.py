"""Dataset constructors for the frozen protocol. Solvers are the repo defaults."""
from __future__ import annotations

import numpy as np

from baselines.classical.burgers1d import generate_dataset as generate_burgers
from baselines.classical.heat2d import generate_dataset as generate_heat


def split_indices(block: dict) -> dict[str, np.ndarray]:
    n_train = int(block["n_train"])
    n_val = int(block["n_val"])
    n_test = int(block["n_test"])
    n = int(block["n_traj"])
    if n_train + n_val + n_test != n:
        raise ValueError("train/val/test counts must sum to n_traj")
    return {
        "train": np.arange(0, n_train),
        "val": np.arange(n_train, n_train + n_val),
        "test": np.arange(n_train + n_val, n),
    }


def load_burgers(block: dict) -> dict:
    data = generate_burgers(
        n_traj=int(block["n_traj"]),
        nx=int(block["nx"]),
        nt=int(block["nt"]),
        nu_range=(float(block["nu_range"][0]), float(block["nu_range"][1])),
        seed=int(block["seed"]),
    )
    dx = float(data["x"][1] - data["x"][0])
    if not np.isclose(dx * len(data["x"]), float(block["L"])):
        raise AssertionError("Burgers length does not match the protocol")
    if not np.isclose(float(data["t"][-1]), float(block["T"])):
        raise AssertionError("Burgers horizon does not match the protocol")
    return data


def load_heat(block: dict) -> dict:
    data = generate_heat(
        n_traj=int(block["n_traj"]),
        n=int(block["n"]),
        nt=int(block["nt"]),
        alpha_range=(float(block["alpha_range"][0]), float(block["alpha_range"][1])),
        seed=int(block["seed"]),
    )
    if not np.isclose(float(data["x"][-1]), float(block["L"])):
        raise AssertionError("Heat domain does not match the protocol")
    if not np.isclose(float(data["t"][-1]), float(block["T"])):
        raise AssertionError("Heat horizon does not match the protocol")
    return data
