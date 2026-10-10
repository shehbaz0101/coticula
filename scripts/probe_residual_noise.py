"""Strong-form residual of an exact Burgers trajectory, then 1% white noise.

The residual is ``metrics.conserve.burgers_residual_l2``. This script does not
change that definition and does not retune the trust cut. It prints the
measured numbers for one trajectory from the label recipe (nx=64, nt=80,
seed 0, first draw).

1% white noise is i.i.d. Gaussian with standard deviation ``0.01 * std(u)``,
generator seed 0, added to the whole trajectory.

    python -m scripts.probe_residual_noise
"""
from __future__ import annotations

import argparse

import numpy as np

from baselines.classical.burgers1d import generate_dataset
from metrics.conserve import burgers_residual_l2
from metrics.trust import RESIDUAL_UNTRUSTED


def measure(seed: int = 0, noise_frac: float = 0.01) -> dict:
    data = generate_dataset(n_traj=1, nx=64, nt=80, seed=seed)
    u = np.asarray(data["u"][:1], dtype=np.float64)
    nu = np.asarray(data["nu"][:1], dtype=np.float64)
    x = np.asarray(data["x"])
    t = np.asarray(data["t"])
    dx = float(x[1] - x[0])
    dt = float(t[1] - t[0])
    clean = burgers_residual_l2(u, dx, dt, nu)
    rng = np.random.default_rng(0)
    sigma = float(noise_frac) * float(np.std(u))
    noisy = u + rng.normal(0.0, sigma, size=u.shape)
    noisy_res = burgers_residual_l2(noisy, dx, dt, nu)
    return {
        "clean": clean,
        "noisy": noisy_res,
        "sigma": sigma,
        "field_std": float(np.std(u)),
        "nu": float(nu[0]),
        "dx": dx,
        "dt": dt,
        "cut": float(RESIDUAL_UNTRUSTED),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-frac", type=float, default=0.01)
    args = parser.parse_args(argv)
    row = measure(seed=args.seed, noise_frac=args.noise_frac)
    crosses = row["noisy"] > row["cut"]
    print(f"grid nx=64 nt=80 label_seed={args.seed} noise_seed=0")
    print(f"nu={row['nu']:.6e} dx={row['dx']:.6e} dt={row['dt']:.6e}")
    print(f"field_std={row['field_std']:.6e}")
    print(f"noise_frac={args.noise_frac:g} sigma={row['sigma']:.6e}")
    print(f"clean_residual_rel_l2={row['clean']:.6e}")
    print(f"noisy_residual_rel_l2={row['noisy']:.6e}")
    print(f"residual_cut={row['cut']:.6e}")
    print(f"noisy_crosses_cut={'yes' if crosses else 'no'}")


if __name__ == "__main__":
    main()
