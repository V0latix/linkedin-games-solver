"""Central solver registry for the Queens puzzle."""

from __future__ import annotations

from collections.abc import Callable

from ...core.types import SolveResult
from .parser import QueensPuzzle
from .solver_baseline import solve_baseline
from .solver_heuristic import solve_heuristic_lcv, solve_heuristic_simple

QueensSolver = Callable[[QueensPuzzle, float | None], SolveResult]

_SOLVERS: dict[str, QueensSolver] = {
    "baseline": solve_baseline,
    "heuristic_simple": solve_heuristic_simple,
    "heuristic_lcv": solve_heuristic_lcv,
}


def list_solvers() -> list[str]:
    return sorted(_SOLVERS)


def get_solver(name: str) -> QueensSolver:
    solver = _SOLVERS.get(name)
    if solver is None:
        known = ", ".join(list_solvers())
        msg = f"Unknown queens solver {name!r}. Known solvers: {known}."
        raise ValueError(msg)
    return solver
