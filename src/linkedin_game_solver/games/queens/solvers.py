"""Central solver registry for the Queens puzzle."""

from __future__ import annotations

from collections.abc import Callable

from ...core.types import SolveResult
from .parser import QueensPuzzle
from .solver_backtracking_bb import solve_backtracking_bb, solve_backtracking_bb_nolcv
from .solver_baseline import solve_baseline
from .solver_csp import solve_csp_ac3
from .solver_dlx import solve_dlx
from .solver_heuristic import solve_heuristic_lcv, solve_heuristic_simple

QueensSolver = Callable[[QueensPuzzle, float | None], SolveResult]

_SOLVERS: dict[str, QueensSolver] = {
    "backtracking_bb": solve_backtracking_bb,
    "backtracking_bb_nolcv": solve_backtracking_bb_nolcv,
    "baseline": solve_baseline,
    "heuristic_simple": solve_heuristic_simple,
    "heuristic_lcv": solve_heuristic_lcv,
    "dlx": solve_dlx,
    "csp_ac3": solve_csp_ac3,
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
