"""Console renderers for Queens puzzles and solutions."""

from __future__ import annotations

from dataclasses import dataclass

from .parser import QueensPuzzle, QueensSolution


@dataclass
class Rendered:
    text: str


def _grid_to_text(grid: list[list[str]]) -> str:
    return "\n".join(" ".join(row) for row in grid)


def render_regions(puzzle: QueensPuzzle) -> Rendered:
    region_grid = [[str(cell) for cell in row] for row in puzzle.regions]
    header = f"game={puzzle.game} n={puzzle.n} regions={len(puzzle.region_ids)}"
    body = _grid_to_text(region_grid)
    return Rendered(text=f"{header}\n{body}")


def render_solution(puzzle: QueensPuzzle, solution: QueensSolution) -> Rendered:
    n = puzzle.n
    grid = [["." for _ in range(n)] for _ in range(n)]

    for r, c in puzzle.blocked:
        grid[r][c] = "x"

    for r, row in enumerate(solution.queens):
        for c, value in enumerate(row):
            if value == 1:
                grid[r][c] = "Q"

    header = render_regions(puzzle).text
    body = _grid_to_text(grid)
    positions = solution.positions()
    pos_text = " ".join(f"({r},{c})" for r, c in positions)
    return Rendered(text=f"{header}\n\nsolution\n{body}\n\nqueens: {pos_text}")
