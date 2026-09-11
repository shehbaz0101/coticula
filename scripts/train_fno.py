"""Train a tiny FNO (optional --pino physics residual) on Burgers and/or heat2d.

Examples
--------
    python -m scripts.train_fno --pde burgers
    python -m scripts.train_fno --pde heat2d --epochs 20
    python -m scripts.train_fno --pde burgers --pino --lambda-pde 0.1
    python -m scripts.train_fno --smoke   # CPU CI: 2 epochs on synthetic data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a laptop-scale FNO / PINO baseline")
    p.add_argument(
        "--pde",
        choices=("burgers", "heat2d", "both"),
        default="burgers",
        help="Which label set to fit (default: burgers)",
    )
    p.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--modes", type=int, default=None)
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--pino", action="store_true", help="Add PDE residual loss (PINO)")
    p.add_argument(
        "--lambda-pde",
        type=float,
        default=1e-3,
        help="PINO residual weight (light default: 1e-3 so L_data still leads)",
    )
    p.add_argument("--lambda-ic", type=float, default=0.0)
    p.add_argument("--eval-n", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    p.add_argument("--out", type=Path, default=None, help="Checkpoint path (one PDE)")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Ignore labels; train 2 epochs on a tiny synthetic set (CI)",
    )
    return p


def _smoke_burgers() -> dict:
    from baselines.classical.burgers1d import generate_dataset

    return generate_dataset(n_traj=6, nx=16, nt=8, seed=0)


def _smoke_heat() -> dict:
    from baselines.classical.heat2d import generate_dataset

    return generate_dataset(n_traj=4, n=12, nt=6, seed=1)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from baselines.fno.io import default_ckpt
    from baselines.fno.train import train_burgers, train_heat

    pdes = ["burgers", "heat2d"] if args.pde == "both" else [args.pde]
    objective = "pino" if args.pino else "fno"

    for pde in pdes:
        out = args.out
        if out is None or (args.pde == "both"):
            out = default_ckpt(pde, objective)
        if pde == "burgers":
            data = _smoke_burgers() if args.smoke else None
            train_burgers(
                data=data,
                epochs=2 if args.smoke else (args.epochs or 40),
                width=args.width or (4 if args.smoke else 16),
                modes=args.modes or (2 if args.smoke else 8),
                n_layers=args.n_layers or (2 if args.smoke else 3),
                batch_size=args.batch_size or (4 if args.smoke else 8),
                lr=args.lr,
                pino=args.pino,
                lambda_pde=args.lambda_pde,
                lambda_ic=args.lambda_ic,
                eval_n=2 if args.smoke else (args.eval_n or 16),
                seed=args.seed,
                device=args.device or "cpu" if args.smoke else args.device,
                out_path=out,
                log_every=1 if args.smoke else 5,
            )
        else:
            data = _smoke_heat() if args.smoke else None
            train_heat(
                data=data,
                epochs=2 if args.smoke else (args.epochs or 25),
                width=args.width or (4 if args.smoke else 12),
                modes=args.modes or (2 if args.smoke else 6),
                n_layers=args.n_layers or (2 if args.smoke else 2),
                batch_size=args.batch_size or (2 if args.smoke else 4),
                lr=args.lr,
                pino=args.pino,
                lambda_pde=args.lambda_pde,
                lambda_ic=args.lambda_ic,
                eval_n=2 if args.smoke else (args.eval_n or 12),
                seed=args.seed,
                device=args.device or "cpu" if args.smoke else args.device,
                out_path=out,
                log_every=1 if args.smoke else 5,
            )


if __name__ == "__main__":
    main()
