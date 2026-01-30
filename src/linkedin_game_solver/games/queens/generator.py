"""Puzzle generator for the LinkedIn Queens puzzle (educational, v1)."""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from time import perf_counter

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
    region_min_row: dict[int, int] = {}
    region_max_row: dict[int, int] = {}
    region_min_col: dict[int, int] = {}
    region_max_col: dict[int, int] = {}
    region_seed: dict[int, Cell] = {}
    last_step: dict[int, tuple[int, int] | None] = {}
    last_cell: dict[int, Cell] = {}

    for region_id, (r, c) in enumerate(sorted(positions)):
        region_by_cell[r][c] = region_id
        frontiers[region_id] = set(_neighbors4(n, r, c))
        region_sizes[region_id] = 1
        region_rows[region_id] = {r}
        region_cols[region_id] = {c}
        region_min_row[region_id] = r
        region_max_row[region_id] = r
        region_min_col[region_id] = c
        region_max_col[region_id] = c
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
            region_min_row[best_region] = min(region_min_row[best_region], r)
            region_max_row[best_region] = max(region_max_row[best_region], r)
            region_min_col[best_region] = min(region_min_col[best_region], c)
            region_max_col[best_region] = max(region_max_col[best_region], c)
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
            min_r = region_min_row[region_id]
            max_r = region_max_row[region_id]
            min_c = region_min_col[region_id]
            max_c = region_max_col[region_id]

            def constrained_key(cell: Cell) -> tuple[int, int, int, int, int, int]:
                r, c = cell
                row_count = len(rows) + (0 if r in rows else 1)
                col_count = len(cols) + (0 if c in cols else 1)
                row_span = max(max_r, r) - min(min_r, r) + 1
                col_span = max(max_c, c) - min(min_c, c) + 1
                distance = abs(r - seed_r) + abs(c - seed_c)
                return (
                    row_count * col_count,
                    row_span + col_span,
                    distance,
                    row_count + col_count,
                    r,
                    c,
                )

            candidates.sort(key=constrained_key)

        r, c = candidates[0]

        region_by_cell[r][c] = region_id
        region_sizes[region_id] += 1
        unassigned.remove((r, c))
        frontiers[region_id].update(_neighbors4(n, r, c))
        region_rows[region_id].add(r)
        region_cols[region_id].add(c)
        region_min_row[region_id] = min(region_min_row[region_id], r)
        region_max_row[region_id] = max(region_max_row[region_id], r)
        region_min_col[region_id] = min(region_min_col[region_id], c)
        region_max_col[region_id] = max(region_max_col[region_id], c)
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


def _copy_payload(payload: dict) -> dict:
    copy_payload = dict(payload)
    copy_payload["regions"] = [row[:] for row in payload["regions"]]
    givens = payload.get("givens", {})
    copy_payload["givens"] = {
        "queens": [cell[:] for cell in givens.get("queens", [])],
        "blocked": [cell[:] for cell in givens.get("blocked", [])],
    }
    return copy_payload


def _region_ambiguity_score(regions: Grid) -> int:
    n = len(regions)
    rows_by_region = [set() for _ in range(n)]
    cols_by_region = [set() for _ in range(n)]
    for r, row in enumerate(regions):
        for c, region_id in enumerate(row):
            rows_by_region[region_id].add(r)
            cols_by_region[region_id].add(c)
    score = 0
    for region_id in range(n):
        score += len(rows_by_region[region_id]) * len(cols_by_region[region_id])
    return score


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


def _blocked_set(payload: dict) -> set[Cell]:
    givens = payload.get("givens", {})
    blocked = givens.get("blocked", [])
    return {tuple(cell) for cell in blocked}


def _append_block(payload: dict, cell: Cell) -> None:
    givens = payload.setdefault("givens", {})
    blocked = givens.setdefault("blocked", [])
    blocked.append([cell[0], cell[1]])


def _try_block_solution_difference(
    payload: dict,
    solution_a: list[Cell],
    solution_b: list[Cell],
    seeds: dict[int, Cell],
) -> Cell | None:
    """Block a queen cell present in solution_b but not in solution_a."""

    blocked = _blocked_set(payload)
    set_a = set(solution_a)
    candidates = [cell for cell in solution_b if cell not in set_a and cell not in blocked]
    random.shuffle(candidates)
    for cell in candidates:
        region_id = payload["regions"][cell[0]][cell[1]]
        if seeds.get(region_id) == cell:
            continue
        if cell in solution_a:
            continue
        return cell
    return None


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


def _boundary_cells(regions: Grid, region_id: int) -> list[Cell]:
    n = len(regions)
    cells: list[Cell] = []
    for r in range(n):
        for c in range(n):
            if regions[r][c] != region_id:
                continue
            for nr, nc in _neighbors4(n, r, c):
                if regions[nr][nc] != region_id:
                    cells.append((r, c))
                    break
    return cells


def _try_repair_regions_multi(
    regions: Grid,
    solution_a: list[Cell],
    solution_b: list[Cell],
    seeds: dict[int, Cell],
) -> Grid | None:
    """Attempt a two-cell swap across two regions to reduce alternative solutions."""

    pos_a = {regions[r][c]: (r, c) for r, c in solution_a}
    pos_b = {regions[r][c]: (r, c) for r, c in solution_b}
    differing_regions = [rid for rid in pos_a if pos_a.get(rid) != pos_b.get(rid)]
    random.shuffle(differing_regions)

    for i, r1 in enumerate(differing_regions):
        for r2 in differing_regions[i + 1 :]:
            if seeds.get(r1) is None or seeds.get(r2) is None:
                continue
            if seeds[r1] == seeds[r2]:
                continue
            # Pick boundary cells to swap.
            b1 = _boundary_cells(regions, r1)
            b2 = _boundary_cells(regions, r2)
            random.shuffle(b1)
            random.shuffle(b2)
            for c1 in b1[:20]:
                if seeds[r1] == c1:
                    continue
                for c2 in b2[:20]:
                    if seeds[r2] == c2:
                        continue
                    new_regions = [row[:] for row in regions]
                    new_regions[c1[0]][c1[1]] = r2
                    new_regions[c2[0]][c2[1]] = r1
                    if not _is_region_connected(new_regions, r1, seeds[r1]):
                        continue
                    if not _is_region_connected(new_regions, r2, seeds[r2]):
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
    block_steps: int = 0,
    global_time_limit_s: float | None = None,
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
    if global_time_limit_s is not None and global_time_limit_s <= 0:
        msg = "global_time_limit_s must be positive"
        raise ValueError(msg)

    rng = random.Random(seed)

    if selection_mode not in {"first", "best"}:
        msg = f"unknown selection_mode={selection_mode!r}"
        raise ValueError(msg)
    if region_mode not in {"balanced", "biased", "serpentine", "constrained", "mixed"}:
        msg = f"unknown region_mode={region_mode!r}"
        raise ValueError(msg)

    start_time = None if global_time_limit_s is None else perf_counter()

    def _timed_out() -> bool:
        if start_time is None:
            return False
        return (perf_counter() - start_time) >= global_time_limit_s

    fallback_payload: dict | None = None
    fallback_solution: QueensSolution | None = None
    fallback_score: int | None = None

    def _record_best_candidate(payload: dict, solution: QueensSolution) -> None:
        nonlocal fallback_payload, fallback_solution, fallback_score
        if global_time_limit_s is None:
            return
        score = _region_ambiguity_score(payload["regions"])
        if fallback_score is None or score < fallback_score:
            fallback_score = score
            fallback_payload = _copy_payload(payload)
            fallback_solution = solution

    best_score = -1.0
    best_payload: dict | None = None
    best_solution: QueensSolution | None = None

    repair_attempts = 0
    repair_successes = 0
    block_attempts = 0
    block_successes = 0
    fast_unique_pass = 0
    full_unique_pass = 0

    def _return_best_on_timeout() -> tuple[dict, QueensSolution] | None:
        if not _timed_out():
            return None
        if best_payload is not None and best_solution is not None:
            return best_payload, best_solution
        if fallback_payload is not None and fallback_solution is not None:
            return fallback_payload, fallback_solution
        msg = f"Unable to generate a unique puzzle for n={n} before timeout ({global_time_limit_s:.2f}s)"
        raise ValueError(msg)

    def _attempt_repair_and_block(payload: dict, solution: QueensSolution, attempts_label: int) -> bool:
        nonlocal repair_attempts, repair_successes, block_attempts, block_successes, full_unique_pass
        if repair_steps <= 0 and block_steps <= 0:
            return False

        def _log_unique() -> None:
            nonlocal full_unique_pass
            full_unique_pass += 1
            if progress_every:
                print(
                    f"[generator] unique found after {attempts_label} candidates "
                    f"(repairs={repair_attempts}, blocks={block_attempts})"
                )

        def _is_unique_now() -> bool:
            return _is_unique_payload(payload, time_limit_s)

        def _fetch_solutions() -> list[list[Cell]]:
            puzzle = parse_puzzle_dict(payload)
            return find_two_solutions_dlx(puzzle, time_limit_s=time_limit_s)

        if repair_steps > 0:
            for _ in range(repair_steps):
                if _timed_out():
                    return False
                repair_attempts += 1
                solutions = _fetch_solutions()
                if len(solutions) < 2:
                    return _is_unique_now()
                seeds = _build_seed_map(payload["regions"], solution)
                repaired = _try_repair_regions_multi(payload["regions"], solutions[0], solutions[1], seeds)
                if repaired is None:
                    repaired = _try_repair_regions(payload["regions"], solutions[0], solutions[1], seeds)
                if repaired is None:
                    break
                payload["regions"] = repaired
                _record_best_candidate(payload, solution)
                if _is_unique_now():
                    repair_successes += 1
                    _log_unique()
                    return True

        if block_steps > 0:
            for _ in range(block_steps):
                if _timed_out():
                    return False
                block_attempts += 1
                solutions = _fetch_solutions()
                if len(solutions) < 2:
                    return _is_unique_now()
                seeds = _build_seed_map(payload["regions"], solution)
                cell = _try_block_solution_difference(payload, solutions[0], solutions[1], seeds)
                if cell is None:
                    break
                _append_block(payload, cell)
                _record_best_candidate(payload, solution)
                if _is_unique_now():
                    block_successes += 1
                    _log_unique()
                    return True

        return False

    if selection_mode == "best":
        solver = get_solver(score_algo)

        attempts = 0
        while max_attempts is None or attempts < max_attempts:
            timeout_result = _return_best_on_timeout()
            if timeout_result is not None:
                return timeout_result
            for _ in range(candidates):
                timeout_result = _return_best_on_timeout()
                if timeout_result is not None:
                    return timeout_result
                attempt_seed = rng.randint(0, 10_000_000) if seed is None else seed + attempts
                mode = _resolve_region_mode(region_mode, attempts)
                payload, solution = _generate_payload_once(n, seed=attempt_seed, region_mode=mode)
                _record_best_candidate(payload, solution)
                attempts += 1
                if progress_every and attempts % progress_every == 0:
                    print(
                        f"[generator] tried {attempts} candidates "
                        f"(mode=best, region={region_mode}, unique={ensure_unique}) "
                        f"repairs={repair_attempts} repaired={repair_successes} "
                        f"blocks={block_attempts} blocked={block_successes} "
                        f"fast_unique_ok={fast_unique_pass} unique_found={full_unique_pass}"
                    )

                if ensure_unique:
                    if fast_unique and not _is_unique_payload(payload, fast_unique_timelimit_s):
                        if _attempt_repair_and_block(payload, solution, attempts):
                            return payload, solution
                        continue
                    if fast_unique:
                        fast_unique_pass += 1
                    if not _is_unique_payload(payload, time_limit_s):
                        if _attempt_repair_and_block(payload, solution, attempts):
                            return payload, solution
                        continue

                if ensure_unique:
                    full_unique_pass += 1
                    if progress_every:
                        print(
                            f"[generator] unique found after {attempts} candidates "
                            f"(repairs={repair_attempts}, blocks={block_attempts})"
                        )
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
        timeout_result = _return_best_on_timeout()
        if timeout_result is not None:
            return timeout_result
        attempt_seed = rng.randint(0, 10_000_000) if seed is None else seed + attempt
        mode = _resolve_region_mode(region_mode, attempt)
        payload, solution = _generate_payload_once(n, seed=attempt_seed, region_mode=mode)
        _record_best_candidate(payload, solution)
        attempt += 1
        if progress_every and attempt % progress_every == 0:
            print(
                f"[generator] tried {attempt} candidates "
                f"(mode=first, region={region_mode}, unique={ensure_unique}) "
                f"repairs={repair_attempts} repaired={repair_successes} "
                f"blocks={block_attempts} blocked={block_successes} "
                f"fast_unique_ok={fast_unique_pass} unique_found={full_unique_pass}"
            )
        if ensure_unique:
            if fast_unique and not _is_unique_payload(payload, fast_unique_timelimit_s):
                if _attempt_repair_and_block(payload, solution, attempt):
                    return payload, solution
                continue
            if fast_unique:
                fast_unique_pass += 1
            if not _is_unique_payload(payload, time_limit_s):
                if _attempt_repair_and_block(payload, solution, attempt):
                    return payload, solution
                continue
            full_unique_pass += 1
            if progress_every:
                print(
                    f"[generator] unique found after {attempt} candidates "
                    f"(repairs={repair_attempts}, blocks={block_attempts})"
                )
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
    global_time_limit_s: float | None = None,
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
        global_time_limit_s=global_time_limit_s,
    )
    puzzle = parse_puzzle_dict(payload)
    validation = validate_solution(puzzle, solution)
    if not validation.ok:
        msg = f"generator produced an invalid (puzzle, solution) pair: {validation.reason}"
        raise ValueError(msg)
    return GeneratedPuzzle(puzzle=puzzle, solution=solution, payload=payload)
