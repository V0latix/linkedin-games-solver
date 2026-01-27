"""Parser for the LinkedIn Queens puzzle JSON format."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

Cell = tuple[int, int]
Grid = list[list[int]]
QueenMatrix = list[list[int]]


@dataclass
class QueensPuzzle:
    """In-memory representation of a Queens puzzle instance."""

    game: str
    n: int
    regions: Grid
    givens_queens: set[Cell]
    blocked: set[Cell]
    region_ids: set[int]


@dataclass
class QueensSolution:
    """Solution container using a binary queen matrix."""

    queens: QueenMatrix

    def positions(self) -> list[Cell]:
        pos: list[Cell] = []
        for r, row in enumerate(self.queens):
            for c, value in enumerate(row):
                if value == 1:
                    pos.append((r, c))
        return pos


def _as_cell_pairs(values: Iterable[Iterable[int]] | None) -> set[Cell]:
    if not values:
        return set()
    cells: set[Cell] = set()
    for pair in values:
        r, c = pair
        cells.add((int(r), int(c)))
    return cells


def _validate_square(regions: Grid, n: int) -> None:
    if len(regions) != n:
        msg = f"regions must have {n} rows, got {len(regions)}"
        raise ValueError(msg)
    for idx, row in enumerate(regions):
        if len(row) != n:
            msg = f"regions row {idx} must have {n} columns, got {len(row)}"
            raise ValueError(msg)


def _validate_cells_in_bounds(cells: set[Cell], n: int, label: str) -> None:
    for r, c in cells:
        if not (0 <= r < n and 0 <= c < n):
            msg = f"{label} cell {(r, c)} is out of bounds for n={n}"
            raise ValueError(msg)


def parse_puzzle_dict(payload: dict) -> QueensPuzzle:
    game = str(payload.get("game", ""))
    if game != "queens":
        msg = f"expected game='queens', got {game!r}"
        raise ValueError(msg)

    n = int(payload["n"])
    regions_raw = payload["regions"]
    regions: Grid = [[int(x) for x in row] for row in regions_raw]
    _validate_square(regions, n)

    givens = payload.get("givens", {})
    givens_queens = _as_cell_pairs(givens.get("queens"))
    blocked = _as_cell_pairs(givens.get("blocked"))

    _validate_cells_in_bounds(givens_queens, n, "given queen")
    _validate_cells_in_bounds(blocked, n, "blocked")

    if givens_queens & blocked:
        msg = "a given queen cannot also be blocked"
        raise ValueError(msg)

    region_ids = {cell for row in regions for cell in row}
    if len(region_ids) != n:
        msg = (
            "the number of distinct region ids must equal n "
            f"(one queen per region). Got {len(region_ids)} for n={n}."
        )
        raise ValueError(msg)

    return QueensPuzzle(
        game=game,
        n=n,
        regions=regions,
        givens_queens=givens_queens,
        blocked=blocked,
        region_ids=region_ids,
    )


def parse_puzzle_file(path: str | Path) -> QueensPuzzle:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_puzzle_dict(data)


def empty_solution(n: int) -> QueensSolution:
    return QueensSolution(queens=[[0 for _ in range(n)] for _ in range(n)])
