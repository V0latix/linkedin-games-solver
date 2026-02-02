"""Puzzle generator for Zip."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from .model import Cell, ZipPuzzle, ZipSolution
from .parser import parse_puzzle_dict
from .validator import validate_solution


@dataclass
class GenerationResult:
    payload: dict
    solution: ZipSolution


def _neighbors(n: int, cell: Cell) -> list[Cell]:
    r, c = cell
    out: list[Cell] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < n:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < n:
        out.append((r, c + 1))
    return out


def _serpentine_path(n: int) -> list[Cell]:
    path: list[Cell] = []
    for r in range(n):
        cols = range(n) if r % 2 == 0 else range(n - 1, -1, -1)
        for c in cols:
            path.append((r, c))
    return path


def _random_hamiltonian_path(n: int, rng: random.Random, time_limit_s: float | None) -> list[Cell] | None:
    start_time = time.perf_counter()
    target = n * n

    start = (rng.randrange(n), rng.randrange(n))
    path: list[Cell] = [start]
    visited = {start}

    def timed_out() -> bool:
        if time_limit_s is None:
            return False
        return (time.perf_counter() - start_time) >= time_limit_s

    def dfs(current: Cell) -> bool:
        if timed_out():
            return False
        if len(path) == target:
            return True

        neighbors = [nbr for nbr in _neighbors(n, current) if nbr not in visited]
        candidates: list[tuple[int, float, Cell]] = []
        for cell in neighbors:
            degree = len([nxt for nxt in _neighbors(n, cell) if nxt not in visited])
            candidates.append((degree, rng.random(), cell))
        candidates.sort(key=lambda item: (item[0], item[1]))
        neighbors = [cell for _deg, _rand, cell in candidates]

        for nxt in neighbors:
            if timed_out():
                return False
            visited.add(nxt)
            path.append(nxt)
            if dfs(nxt):
                return True
            path.pop()
            visited.remove(nxt)
        return False

    if dfs(start):
        return path
    return None


def _choose_checkpoints(path: list[Cell], rng: random.Random, checkpoints: int | float) -> dict[int, Cell]:
    total = len(path)
    if isinstance(checkpoints, float):
        if checkpoints <= 0 or checkpoints >= 1:
            msg = "Checkpoint ratio must be between 0 and 1."
            raise ValueError(msg)
        count = max(1, int(round(total * checkpoints)))
    else:
        if checkpoints <= 0:
            msg = "Checkpoint count must be positive."
            raise ValueError(msg)
        count = min(checkpoints, total)

    if count == 1:
        return {1: path[0]}

    indices = [0] + rng.sample(range(1, total), count - 1)
    indices.sort()
    return {k + 1: path[idx] for k, idx in enumerate(indices)}


def _free_neighbors(puzzle: ZipPuzzle, visited: set[Cell], cell: Cell) -> list[Cell]:
    return [nbr for nbr in puzzle.neighbors[cell] if nbr not in visited]


def _isolated_cells(puzzle: ZipPuzzle, visited: set[Cell], current: Cell) -> bool:
    unvisited = []
    for r in range(puzzle.n):
        for c in range(puzzle.n):
            cell = (r, c)
            if cell not in visited:
                unvisited.append(cell)
    if not unvisited:
        return False

    remaining = len(unvisited)
    for cell in unvisited:
        if _free_neighbors(puzzle, visited, cell):
            continue
        if remaining == 1 and current in puzzle.neighbors[cell]:
            continue
        return True
    return False


def _components_ok(puzzle: ZipPuzzle, visited: set[Cell]) -> bool:
    unvisited = {(r, c) for r in range(puzzle.n) for c in range(puzzle.n) if (r, c) not in visited}
    if not unvisited:
        return True
    stack = [next(iter(unvisited))]
    seen = {stack[0]}
    while stack:
        cell = stack.pop()
        for nxt in puzzle.neighbors[cell]:
            if nxt in unvisited and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen == unvisited


def _lcv_score(puzzle: ZipPuzzle, visited: set[Cell], candidate: Cell) -> int:
    score = 0
    for neighbor in _free_neighbors(puzzle, visited, candidate):
        score += len(_free_neighbors(puzzle, visited, neighbor))
    return score


def _count_solutions(puzzle: ZipPuzzle, time_limit_s: float | None) -> int | None:
    start_time = time.perf_counter()
    limit_ms = None if time_limit_s is None else max(0.0, time_limit_s * 1000.0)

    def timed_out() -> bool:
        if limit_ms is None:
            return False
        return (time.perf_counter() - start_time) * 1000.0 >= limit_ms

    if 1 not in puzzle.numbers:
        return 0

    start = puzzle.numbers[1]
    max_k = max(puzzle.numbers.keys())
    next_k = 2 if max_k >= 2 else max_k + 1

    path: list[Cell] = [start]
    visited = {start}
    solutions = 0

    def is_checkpoint(cell: Cell) -> int | None:
        for k, pos in puzzle.numbers.items():
            if pos == cell:
                return k
        return None

    def dfs(current: Cell, next_target: int) -> bool:
        nonlocal solutions
        if timed_out():
            return False
        if len(path) == puzzle.n * puzzle.n:
            solutions += 1
            return solutions < 2
        if _isolated_cells(puzzle, visited, current):
            return True
        if not _components_ok(puzzle, visited):
            return True

        neighbors = _free_neighbors(puzzle, visited, current)
        if not neighbors:
            return True

        neighbors.sort(
            key=lambda cell: (
                -_lcv_score(puzzle, visited, cell),
                len(_free_neighbors(puzzle, visited, cell)),
            )
        )

        for nxt in neighbors:
            if timed_out():
                return False
            k = is_checkpoint(nxt)
            if k is not None and k != next_target:
                continue
            next_for_child = next_target
            if k is not None:
                next_for_child += 1

            visited.add(nxt)
            path.append(nxt)
            cont = dfs(nxt, next_for_child)
            path.pop()
            visited.remove(nxt)
            if not cont:
                return False
            if solutions >= 2:
                return False
        return True

    dfs(start, next_k)
    if timed_out():
        return None
    return solutions


def generate_zip_puzzle_payload(
    n: int,
    seed: int | None = None,
    checkpoints: int | float = 0.2,
    ensure_unique: bool = True,
    unique_timelimit_s: float | None = 1.0,
    max_attempts: int = 200,
    path_timelimit_s: float | None = 0.5,
) -> GenerationResult:
    if n <= 1:
        msg = "n must be at least 2."
        raise ValueError(msg)

    rng = random.Random(seed)
    attempts = 0

    while attempts < max_attempts:
        attempts += 1
        path = _random_hamiltonian_path(n, rng, time_limit_s=path_timelimit_s)
        if path is None:
            path = _serpentine_path(n)

        numbers = _choose_checkpoints(path, rng, checkpoints)
        payload = {
            "game": "zip",
            "n": n,
            "numbers": [{"k": k, "r": r, "c": c} for k, (r, c) in sorted(numbers.items())],
            "walls": [],
        }

        puzzle = parse_puzzle_dict(payload)
        solution = ZipSolution(path=path)
        validation = validate_solution(puzzle, solution)
        if not validation.ok:
            continue

        if ensure_unique:
            count = _count_solutions(puzzle, time_limit_s=unique_timelimit_s)
            if count is None or count != 1:
                continue

        return GenerationResult(payload=payload, solution=solution)

    msg = f"Failed to generate a {'unique ' if ensure_unique else ''}puzzle after {max_attempts} attempts."
    raise ValueError(msg)
