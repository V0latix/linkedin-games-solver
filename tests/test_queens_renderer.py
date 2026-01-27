from __future__ import annotations

from pathlib import Path

from linkedin_game_solver.games.queens.parser import QueensSolution, parse_puzzle_file
from linkedin_game_solver.games.queens.renderer import render_regions, render_solution

SOLUTION_POSITIONS = [(0, 0), (1, 2), (2, 4), (3, 1), (4, 3), (5, 5)]


def _matrix_from_positions(n: int, positions: list[tuple[int, int]]) -> list[list[int]]:
    grid = [[0 for _ in range(n)] for _ in range(n)]
    for r, c in positions:
        grid[r][c] = 1
    return grid


def test_render_regions_contains_header() -> None:
    puzzle = parse_puzzle_file(Path("data/curated/queens/example_6x6.json"))
    rendered = render_regions(puzzle).text
    assert "game=queens" in rendered
    assert "n=6" in rendered


def test_render_solution_contains_queens_and_blocked() -> None:
    puzzle = parse_puzzle_file(Path("data/curated/queens/example_6x6.json"))
    solution = QueensSolution(queens=_matrix_from_positions(puzzle.n, SOLUTION_POSITIONS))
    rendered = render_solution(puzzle, solution).text
    assert "solution" in rendered
    assert "Q" in rendered
    assert "x" in rendered
    assert "(0,0)" in rendered
