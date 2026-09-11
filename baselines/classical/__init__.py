"""Classical NumPy finite-difference solvers (Week-1 label generators)."""
from .burgers1d import generate_dataset as generate_burgers
from .burgers1d import solve_burgers
from .heat2d import generate_dataset as generate_heat2d
from .heat2d import solve_heat2d

__all__ = [
    "solve_burgers",
    "solve_heat2d",
    "generate_burgers",
    "generate_heat2d",
]
