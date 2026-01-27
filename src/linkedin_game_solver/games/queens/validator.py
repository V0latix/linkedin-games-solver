"""Validation utilities for Queens puzzles and solutions."""

from __future__ import annotations

from dataclasses import dataclass

from .parser import Cell, QueensPuzzle, QueensSolution


@dataclass
class ValidationResult:
    ok: bool
    reason: str | None = None


def _queen_positions(solution: QueensSolution) -> list[Cell]:
    return solution.positions()


def _in_bounds(puzzle: QueensPuzzle, r: int, c: int) -> bool:
    return 0 <= r < puzzle.n and 0 <= c < puzzle.n


def _adjacent(a: Cell, b: Cell) -> bool:
    ar, ac = a
    br, bc = b
    return max(abs(ar - br), abs(ac - bc)) <= 1


def validate_solution(puzzle: QueensPuzzle, solution: QueensSolution) -> ValidationResult:
    n = puzzle.n
    queens = solution.queens
    if len(queens) != n or any(len(row) != n for row in queens):
        return ValidationResult(False, "solution matrix must be n x n")

    positions = _queen_positions(solution)
    if any(not _in_bounds(puzzle, r, c) for r, c in positions):
        return ValidationResult(False, "solution contains out-of-bounds queens")

    if any((r, c) in puzzle.blocked for r, c in positions):
        return ValidationResult(False, "solution places a queen on a blocked cell")

    for r, c in puzzle.givens_queens:
        if queens[r][c] != 1:
            return ValidationResult(False, "solution does not satisfy given queens")

    total_queens = len(positions)
    if total_queens != n:
        return ValidationResult(False, f"expected exactly {n} queens, got {total_queens}")

    row_counts = [sum(row) for row in queens]
    if any(count != 1 for count in row_counts):
        return ValidationResult(False, "each row must contain exactly one queen")

    col_counts = [sum(queens[r][c] for r in range(n)) for c in range(n)]
    if any(count != 1 for count in col_counts):
        return ValidationResult(False, "each column must contain exactly one queen")

    region_counts: dict[int, int] = {region_id: 0 for region_id in puzzle.region_ids}
    for r, c in positions:
        region_id = puzzle.regions[r][c]
        region_counts[region_id] = region_counts.get(region_id, 0) + 1
    if any(count != 1 for count in region_counts.values()):
        return ValidationResult(False, "each region must contain exactly one queen")

    for i, a in enumerate(positions):
        for b in positions[i + 1 :]:
            if _adjacent(a, b):
                return ValidationResult(False, "queens cannot be adjacent (8-neighborhood)")

    return ValidationResult(True, None)
