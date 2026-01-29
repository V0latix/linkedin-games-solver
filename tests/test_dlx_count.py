from __future__ import annotations

from linkedin_game_solver.games.queens.parser import parse_puzzle_dict
from linkedin_game_solver.games.queens.solver_dlx import count_solutions_dlx, find_two_solutions_dlx


def _regions_by_row(n: int) -> list[list[int]]:
    return [[r for _ in range(n)] for r in range(n)]


def test_count_solutions_dlx_limit2_stops_at_two() -> None:
    payload = {
        "game": "queens",
        "n": 4,
        "regions": _regions_by_row(4),
        "givens": {"queens": [], "blocked": []},
    }
    puzzle = parse_puzzle_dict(payload)
    count = count_solutions_dlx(puzzle, limit=2)
    assert count == 2


def test_find_two_solutions_returns_two() -> None:
    payload = {
        "game": "queens",
        "n": 4,
        "regions": _regions_by_row(4),
        "givens": {"queens": [], "blocked": []},
    }
    puzzle = parse_puzzle_dict(payload)
    solutions = find_two_solutions_dlx(puzzle)
    assert len(solutions) == 2
    assert solutions[0] != solutions[1]
