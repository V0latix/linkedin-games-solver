"""Puzzle generator for the LinkedIn Queens puzzle (educational, v1)."""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

from .parser import Cell, Grid, QueensPuzzle, QueensSolution, parse_puzzle_dict
from .validator import validate_solution


@dataclass
class GeneratedPuzzle:
    """Container for a generated puzzle and its known-valid solution."""

    puzzle: QueensPuzzle
    solution: QueensSolution
    payload: dict


def _adjacent(a: Cell, b: Cell) -> bool:
    ar, ac = a
    br, bc = b
    return max(abs(ar - br), abs(ac - bc)) <= 1


def _matrix_from_positions(n: int, positions: Iterable[Cell]) -> list[list[int]]:
    grid = [[0 for _ in range(n)] for _ in range(n)]
    for r, c in positions:
        grid[r][c] = 1
    return grid


def generate_solution(n: int, seed: int | None = None) -> QueensSolution:
    """Generate a valid queens placement (row/col + non-adjacency).

    Intuition: backtrack row by row, enforcing column uniqueness and the
    official non-adjacency constraint. Worst-case complexity is exponential,
    but it works well for small/medium sizes and is good enough for v1.
    """

    rng = random.Random(seed)
    cols_used: set[int] = set()
    positions: list[Cell] = []

    def can_place(r: int, c: int) -> bool:
        if c in cols_used:
            return False
        return all(not _adjacent((r, c), pos) for pos in positions)

    def dfs(row: int) -> bool:
        if row == n:
            return True

        candidates = list(range(n))
        rng.shuffle(candidates)
        for col in candidates:
            if not can_place(row, col):
                continue
            cols_used.add(col)
            positions.append((row, col))
            if dfs(row + 1):
                return True
            positions.pop()
            cols_used.remove(col)
        return False

    if not dfs(0):
        msg = f"Unable to generate a valid solution for n={n}"
        raise ValueError(msg)

    return QueensSolution(queens=_matrix_from_positions(n, positions))


def _neighbors4(n: int, r: int, c: int) -> list[Cell]:
    candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [(rr, cc) for rr, cc in candidates if 0 <= rr < n and 0 <= cc < n]


def generate_regions_from_solution(
    solution: QueensSolution,
    seed: int | None = None,
) -> Grid:
    """Build a region partition compatible with the given solution.

    Strategy: use each queen cell as a seed and grow regions via random
    multi-source expansion until the full grid is covered. This guarantees
    exactly one queen per region because each region has a single seed.
    """

    rng = random.Random(seed)
    n = len(solution.queens)
    positions = solution.positions()
    if len(positions) != n:
        msg = "solution must contain exactly n queens to seed n regions"
        raise ValueError(msg)

    region_by_cell: list[list[int]] = [[-1 for _ in range(n)] for _ in range(n)]
    frontiers: dict[int, set[Cell]] = {}
    region_sizes: dict[int, int] = {}

    for region_id, (r, c) in enumerate(sorted(positions)):
        region_by_cell[r][c] = region_id
        frontiers[region_id] = set(_neighbors4(n, r, c))
        region_sizes[region_id] = 1

    unassigned = {(r, c) for r in range(n) for c in range(n) if region_by_cell[r][c] < 0}

    while unassigned:
        expandable = [rid for rid, frontier in frontiers.items() if frontier & unassigned]
        if not expandable:
            # Fallback: attach a remaining cell to the nearest region by Manhattan distance.
            r, c = next(iter(unassigned))
            best_region = min(
                region_sizes,
                key=lambda rid: min(abs(r - rr) + abs(c - cc) for rr, cc in positions if region_by_cell[rr][cc] == rid),
            )
            region_by_cell[r][c] = best_region
            region_sizes[best_region] += 1
            unassigned.remove((r, c))
            frontiers[best_region].update(_neighbors4(n, r, c))
            continue

        # Prefer smaller regions to keep the partition reasonably balanced.
        min_size = min(region_sizes[rid] for rid in expandable)
        smallest = [rid for rid in expandable if region_sizes[rid] == min_size]
        region_id = rng.choice(smallest)

        candidates = list(frontiers[region_id] & unassigned)
        rng.shuffle(candidates)
        r, c = candidates[0]

        region_by_cell[r][c] = region_id
        region_sizes[region_id] += 1
        unassigned.remove((r, c))
        frontiers[region_id].update(_neighbors4(n, r, c))

    return region_by_cell


def generate_puzzle_payload(n: int, seed: int | None = None) -> tuple[dict, QueensSolution]:
    """Generate a puzzle JSON payload plus its known-valid solution."""

    solution = generate_solution(n, seed=seed)
    regions = generate_regions_from_solution(solution, seed=seed)
    payload = {
        "game": "queens",
        "n": n,
        "regions": regions,
        "givens": {"queens": [], "blocked": []},
    }
    return payload, solution


def generate_puzzle(n: int, seed: int | None = None) -> GeneratedPuzzle:
    """Generate a parsed puzzle along with its known-valid solution."""

    payload, solution = generate_puzzle_payload(n, seed=seed)
    puzzle = parse_puzzle_dict(payload)
    validation = validate_solution(puzzle, solution)
    if not validation.ok:
        msg = f"generator produced an invalid (puzzle, solution) pair: {validation.reason}"
        raise ValueError(msg)
    return GeneratedPuzzle(puzzle=puzzle, solution=solution, payload=payload)
