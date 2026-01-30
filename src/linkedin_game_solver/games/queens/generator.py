"""Puzzle generator for the LinkedIn Queens puzzle (educational, v1)."""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

from .parser import Cell, Grid, QueensPuzzle, QueensSolution, parse_puzzle_dict
from .solver_dlx import count_solutions_dlx, find_two_solutions_dlx
from .solvers import get_solver
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
    mode: str = "balanced",
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
    region_rows: dict[int, set[int]] = {}
    region_cols: dict[int, set[int]] = {}
    region_seed: dict[int, Cell] = {}
    last_step: dict[int, tuple[int, int] | None] = {}
    last_cell: dict[int, Cell] = {}

    for region_id, (r, c) in enumerate(sorted(positions)):
        region_by_cell[r][c] = region_id
        frontiers[region_id] = set(_neighbors4(n, r, c))
        region_sizes[region_id] = 1
        region_rows[region_id] = {r}
        region_cols[region_id] = {c}
        region_seed[region_id] = (r, c)
        last_step[region_id] = None
        last_cell[region_id] = (r, c)

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
            region_rows[best_region].add(r)
            region_cols[best_region].add(c)
            prev_r, prev_c = last_cell[best_region]
            last_step[best_region] = (r - prev_r, c - prev_c)
            last_cell[best_region] = (r, c)
            continue

        # Prefer smaller regions to keep the partition reasonably balanced.
        min_size = min(region_sizes[rid] for rid in expandable)
        smallest = [rid for rid in expandable if region_sizes[rid] == min_size]
        region_id = rng.choice(smallest)

        candidates = list(frontiers[region_id] & unassigned)
        rng.shuffle(candidates)

        if mode == "serpentine":
            candidates.sort(key=lambda cell: (cell[0], cell[1] if cell[0] % 2 == 0 else -cell[1]))
        elif mode == "biased":
            preferred = last_step.get(region_id)
            prev = last_cell.get(region_id)
            if preferred is not None and prev is not None:
                pr, pc = preferred
                prev_r, prev_c = prev
                candidates.sort(
                    key=lambda cell: 0 if (cell[0] - prev_r, cell[1] - prev_c) == (pr, pc) else 1
                )
        elif mode == "constrained":
            seed_r, seed_c = region_seed[region_id]
            rows = region_rows[region_id]
            cols = region_cols[region_id]
            candidates.sort(
                key=lambda cell: (
                    abs(cell[0] - seed_r) + abs(cell[1] - seed_c),
                    len(rows | {cell[0]}) + len(cols | {cell[1]}),
                    cell[0],
                    cell[1],
                )
            )

        r, c = candidates[0]

        region_by_cell[r][c] = region_id
        region_sizes[region_id] += 1
        unassigned.remove((r, c))
        frontiers[region_id].update(_neighbors4(n, r, c))
        region_rows[region_id].add(r)
        region_cols[region_id].add(c)
        prev_r, prev_c = last_cell[region_id]
        last_step[region_id] = (r - prev_r, c - prev_c)
        last_cell[region_id] = (r, c)

    return region_by_cell


def _generate_payload_once(
    n: int,
    seed: int | None,
    region_mode: str,
) -> tuple[dict, QueensSolution]:
    solution = generate_solution(n, seed=seed)
    regions = generate_regions_from_solution(solution, seed=seed, mode=region_mode)
    payload = {
        "game": "queens",
        "n": n,
        "regions": regions,
        "givens": {"queens": [], "blocked": []},
    }
    return payload, solution


def _resolve_region_mode(region_mode: str, attempt: int) -> str:
    if region_mode == "mixed":
        modes = ("balanced", "biased", "serpentine", "constrained")
        return modes[attempt % len(modes)]
    return region_mode


def _is_unique_payload(payload: dict, time_limit_s: float | None) -> bool:
    puzzle = parse_puzzle_dict(payload)
    count = count_solutions_dlx(puzzle, limit=2, time_limit_s=time_limit_s)
    return count == 1


def _region_cells(regions: Grid, region_id: int) -> list[Cell]:
    cells: list[Cell] = []
    for r, row in enumerate(regions):
        for c, value in enumerate(row):
            if value == region_id:
                cells.append((r, c))
    return cells


def _is_region_connected(regions: Grid, region_id: int, seed: Cell) -> bool:
    cells = _region_cells(regions, region_id)
    if not cells:
        return False
    if seed not in cells:
        return False
    target = set(cells)
    stack = [seed]
    visited = {seed}
    n = len(regions)
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(n, r, c):
            if (nr, nc) in target and (nr, nc) not in visited:
                visited.add((nr, nc))
                stack.append((nr, nc))
    return visited == target


def _build_seed_map(regions: Grid, solution: QueensSolution) -> dict[int, Cell]:
    seeds: dict[int, Cell] = {}
    for r, c in solution.positions():
        seeds[regions[r][c]] = (r, c)
    return seeds


def _try_repair_regions(
    regions: Grid,
    solution_a: list[Cell],
    solution_b: list[Cell],
    seeds: dict[int, Cell],
) -> Grid | None:
    n = len(regions)
    pos_a = {regions[r][c]: (r, c) for r, c in solution_a}
    pos_b = {regions[r][c]: (r, c) for r, c in solution_b}

    differing_regions = [rid for rid in pos_a if pos_a.get(rid) != pos_b.get(rid)]
    random.shuffle(differing_regions)

    for region_id in differing_regions:
        b_cell = pos_b.get(region_id)
        if b_cell is None:
            continue
        if seeds.get(region_id) == b_cell:
            continue

        br, bc = b_cell
        neighbor_regions = {regions[nr][nc] for nr, nc in _neighbors4(n, br, bc) if regions[nr][nc] != region_id}
        if not neighbor_regions:
            continue

        # Prefer moving into a region that already has a queen in solution_b.
        neighbor_regions = sorted(
            neighbor_regions,
            key=lambda rid: 0 if rid in pos_b and pos_b[rid] != b_cell else 1,
        )

        for target_region in neighbor_regions:
            if seeds.get(target_region) == b_cell:
                continue

            new_regions = [row[:] for row in regions]
            new_regions[br][bc] = target_region

            seed_a = seeds.get(region_id)
            seed_b = seeds.get(target_region)
            if seed_a is None or seed_b is None:
                continue

            if not _is_region_connected(new_regions, region_id, seed_a):
                continue
            if not _is_region_connected(new_regions, target_region, seed_b):
                continue

            return new_regions

    return None


def generate_puzzle_payload(
    n: int,
    seed: int | None = None,
    ensure_unique: bool = True,
    max_attempts: int | None = 50,
    time_limit_s: float | None = None,
    region_mode: str = "balanced",
    selection_mode: str = "first",
    candidates: int = 20,
    score_algo: str = "heuristic_lcv",
    search_until_unique: bool = False,
    progress_every: int | None = None,
    fast_unique: bool = False,
    fast_unique_timelimit_s: float = 0.5,
    repair_steps: int = 0,
) -> tuple[dict, QueensSolution]:
    """Generate a puzzle JSON payload plus its known-valid solution.

    Strategy: solution-first (place all queens), then build connected regions
    by multi-source flood fill seeded at the queens. When ensure_unique=True,
    the generator retries until the puzzle has exactly one solution (checked
    by DLX up to 2 solutions).
    """

    if max_attempts is not None and max_attempts <= 0:
        msg = "max_attempts must be positive"
        raise ValueError(msg)

    rng = random.Random(seed)

    if selection_mode not in {"first", "best"}:
        msg = f"unknown selection_mode={selection_mode!r}"
        raise ValueError(msg)
    if region_mode not in {"balanced", "biased", "serpentine", "constrained", "mixed"}:
        msg = f"unknown region_mode={region_mode!r}"
        raise ValueError(msg)

    repair_attempts = 0
    repair_successes = 0

    if selection_mode == "best":
        solver = get_solver(score_algo)
        best_score = -1.0
        best_payload: dict | None = None
        best_solution: QueensSolution | None = None

        attempts = 0
        while max_attempts is None or attempts < max_attempts:
            for _ in range(candidates):
                attempt_seed = rng.randint(0, 10_000_000) if seed is None else seed + attempts
                mode = _resolve_region_mode(region_mode, attempts)
                payload, solution = _generate_payload_once(n, seed=attempt_seed, region_mode=mode)
                attempts += 1
                if progress_every and attempts % progress_every == 0:
                    print(
                        f"[generator] tried {attempts} candidates "
                        f"(mode=best, region={region_mode}, unique={ensure_unique}) "
                        f"repairs={repair_attempts} repaired={repair_successes}"
                    )

                if ensure_unique:
                    if fast_unique and not _is_unique_payload(payload, fast_unique_timelimit_s):
                        continue
                    if not _is_unique_payload(payload, time_limit_s):
                        if repair_steps > 0:
                            puzzle = parse_puzzle_dict(payload)
                            solutions = find_two_solutions_dlx(puzzle, time_limit_s=time_limit_s)
                            if len(solutions) >= 2:
                                seeds = _build_seed_map(payload["regions"], solution)
                                for _ in range(repair_steps):
                                    repair_attempts += 1
                                    repaired = _try_repair_regions(
                                        payload["regions"], solutions[0], solutions[1], seeds
                                    )
                                    if repaired is None:
                                        break
                                    payload["regions"] = repaired
                                    if _is_unique_payload(payload, time_limit_s):
                                        repair_successes += 1
                                        return payload, solution
                            continue
                        continue

                puzzle = parse_puzzle_dict(payload)
                result = solver(puzzle, time_limit_s=time_limit_s)
                score = float(result.metrics.nodes) + float(result.metrics.backtracks)
                if score > best_score:
                    best_score = score
                    best_payload = payload
                    best_solution = solution

            if best_payload is not None and best_solution is not None:
                return best_payload, best_solution

            if not search_until_unique:
                msg = f"Unable to generate a unique puzzle for n={n} after {attempts} candidates"
                raise ValueError(msg)

        msg = f"Unable to generate a unique puzzle for n={n} after {attempts} candidates"
        raise ValueError(msg)

    attempt = 0
    while max_attempts is None or attempt < max_attempts:
        attempt_seed = rng.randint(0, 10_000_000) if seed is None else seed + attempt
        mode = _resolve_region_mode(region_mode, attempt)
        payload, solution = _generate_payload_once(n, seed=attempt_seed, region_mode=mode)
        attempt += 1
        if progress_every and attempt % progress_every == 0:
            print(
                f"[generator] tried {attempt} candidates "
                f"(mode=first, region={region_mode}, unique={ensure_unique}) "
                f"repairs={repair_attempts} repaired={repair_successes}"
            )
        if ensure_unique:
            if fast_unique and not _is_unique_payload(payload, fast_unique_timelimit_s):
                continue
            if not _is_unique_payload(payload, time_limit_s):
                if repair_steps > 0:
                    puzzle = parse_puzzle_dict(payload)
                    solutions = find_two_solutions_dlx(puzzle, time_limit_s=time_limit_s)
                    if len(solutions) >= 2:
                        seeds = _build_seed_map(payload["regions"], solution)
                        for _ in range(repair_steps):
                            repair_attempts += 1
                            repaired = _try_repair_regions(payload["regions"], solutions[0], solutions[1], seeds)
                            if repaired is None:
                                break
                            payload["regions"] = repaired
                            if _is_unique_payload(payload, time_limit_s):
                                repair_successes += 1
                                return payload, solution
                    continue
                continue
            return payload, solution
        return payload, solution

    msg = f"Unable to generate a unique puzzle for n={n} after {attempt} attempts"
    raise ValueError(msg)


def generate_puzzle(
    n: int,
    seed: int | None = None,
    ensure_unique: bool = True,
    max_attempts: int | None = 50,
    time_limit_s: float | None = None,
    fast_unique: bool = False,
    fast_unique_timelimit_s: float = 0.5,
) -> GeneratedPuzzle:
    """Generate a parsed puzzle along with its known-valid solution."""

    payload, solution = generate_puzzle_payload(
        n,
        seed=seed,
        ensure_unique=ensure_unique,
        max_attempts=max_attempts,
        time_limit_s=time_limit_s,
        fast_unique=fast_unique,
        fast_unique_timelimit_s=fast_unique_timelimit_s,
    )
    puzzle = parse_puzzle_dict(payload)
    validation = validate_solution(puzzle, solution)
    if not validation.ok:
        msg = f"generator produced an invalid (puzzle, solution) pair: {validation.reason}"
        raise ValueError(msg)
    return GeneratedPuzzle(puzzle=puzzle, solution=solution, payload=payload)
