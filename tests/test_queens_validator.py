from __future__ import annotations

from pathlib import Path

from linkedin_game_solver.games.queens.parser import QueensSolution, parse_puzzle_file
from linkedin_game_solver.games.queens.validator import validate_solution

SOLUTION_POSITIONS = [(0, 0), (1, 2), (2, 4), (3, 1), (4, 3), (5, 5)]


def _matrix_from_positions(n: int, positions: list[tuple[int, int]]) -> list[list[int]]:
    grid = [[0 for _ in range(n)] for _ in range(n)]
    for r, c in positions:
        grid[r][c] = 1
    return grid


def test_validate_solution_success() -> None:
    puzzle = parse_puzzle_file(Path("data/curated/queens/example_6x6.json"))
    solution = QueensSolution(queens=_matrix_from_positions(puzzle.n, SOLUTION_POSITIONS))
    result = validate_solution(puzzle, solution)
    assert result.ok, result.reason


def test_validate_solution_rejects_adjacency() -> None:
    puzzle = parse_puzzle_file(Path("data/curated/queens/example_6x6.json"))
    bad_positions = [(0, 0), (1, 1), (2, 4), (3, 2), (4, 5), (5, 3)]
    solution = QueensSolution(queens=_matrix_from_positions(puzzle.n, bad_positions))
    result = validate_solution(puzzle, solution)
    assert not result.ok
    assert result.reason is not None
