"""Backtracking solver with simple branch-and-bound style pruning."""

from __future__ import annotations

from dataclasses import dataclass

from ...core.metrics import Timer
from ...core.types import SolveMetrics, SolveResult
from .parser import Cell, QueensPuzzle, QueensSolution, empty_solution
from .validator import validate_solution


@dataclass
class _State:
    queens: list[list[int]]
    row_has: list[bool]
    col_has: list[bool]
    region_has: dict[int, bool]
    adj_counts: list[list[int]]


def _build_neighbors(n: int) -> list[list[list[Cell]]]:
    neighbors: list[list[list[Cell]]] = [[[] for _ in range(n)] for _ in range(n)]
    for r in range(n):
        for c in range(n):
            cells: list[Cell] = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n:
                        cells.append((nr, nc))
            neighbors[r][c] = cells
    return neighbors


def _build_row_domains(puzzle: QueensPuzzle) -> tuple[list[list[int]], dict[int, int]]:
    n = puzzle.n
    domains: list[list[int]] = []
    givens_by_row: dict[int, int] = {}

    for r, c in puzzle.givens_queens:
        if r in givens_by_row and givens_by_row[r] != c:
            msg = f"invalid givens: multiple queens in row {r}"
            raise ValueError(msg)
        givens_by_row[r] = c

    for r in range(n):
        blocked_cols = {c for (rr, c) in puzzle.blocked if rr == r}
        if r in givens_by_row:
            domains.append([givens_by_row[r]])
        else:
            domains.append([c for c in range(n) if c not in blocked_cols])

    return domains, givens_by_row


def _initialize_state(
    puzzle: QueensPuzzle,
) -> tuple[_State, list[list[int]], dict[int, int], list[list[list[Cell]]]]:
    n = puzzle.n
    solution = empty_solution(n)
    state = _State(
        queens=solution.queens,
        row_has=[False for _ in range(n)],
        col_has=[False for _ in range(n)],
        region_has={region_id: False for region_id in puzzle.region_ids},
        adj_counts=[[0 for _ in range(n)] for _ in range(n)],
    )

    domains, givens_by_row = _build_row_domains(puzzle)
    neighbors = _build_neighbors(n)

    def can_place(r: int, c: int) -> bool:
        if (r, c) in puzzle.blocked:
            return False
        if state.row_has[r] or state.col_has[c]:
            return False
        region_id = puzzle.regions[r][c]
        if state.region_has.get(region_id, False):
            return False
        return state.adj_counts[r][c] == 0

    def place(r: int, c: int) -> None:
        state.queens[r][c] = 1
        state.row_has[r] = True
        state.col_has[c] = True
        region_id = puzzle.regions[r][c]
        state.region_has[region_id] = True
        for nr, nc in neighbors[r][c]:
            state.adj_counts[nr][nc] += 1

    # Apply givens early.
    for r, c in sorted(puzzle.givens_queens):
        if (r, c) in puzzle.blocked:
            msg = f"invalid givens: queen at {(r, c)} is blocked"
            raise ValueError(msg)
        if not can_place(r, c):
            msg = f"invalid givens: queen at {(r, c)} violates constraints"
            raise ValueError(msg)
        place(r, c)

    return state, domains, givens_by_row, neighbors


def _solve_backtracking_bb(
    puzzle: QueensPuzzle,
    time_limit_s: float | None,
    use_lcv: bool,
) -> SolveResult:
    """Backtracking with MRV ordering + forward-checking style pruning."""

    timer = Timer()
    timer.start()
    metrics = SolveMetrics()
    limit_ms = None if time_limit_s is None else max(0.0, time_limit_s * 1000.0)

    try:
        state, domains, givens_by_row, neighbors = _initialize_state(puzzle)
    except ValueError as exc:
        metrics.time_ms = timer.elapsed_ms()
        return SolveResult(
            solved=False,
            solution=None,
            metrics=metrics,
            error=f"Invalid puzzle givens: {exc}",
        )

    def timed_out() -> bool:
        return limit_ms is not None and timer.elapsed_ms() >= limit_ms

    def can_place(r: int, c: int) -> bool:
        if (r, c) in puzzle.blocked:
            return False
        if state.row_has[r] or state.col_has[c]:
            return False
        region_id = puzzle.regions[r][c]
        if state.region_has.get(region_id, False):
            return False
        return state.adj_counts[r][c] == 0

    def place(r: int, c: int) -> None:
        state.queens[r][c] = 1
        state.row_has[r] = True
        state.col_has[c] = True
        region_id = puzzle.regions[r][c]
        state.region_has[region_id] = True
        for nr, nc in neighbors[r][c]:
            state.adj_counts[nr][nc] += 1

    def unplace(r: int, c: int) -> None:
        state.queens[r][c] = 0
        state.row_has[r] = False
        state.col_has[c] = False
        region_id = puzzle.regions[r][c]
        state.region_has[region_id] = False
        for nr, nc in neighbors[r][c]:
            state.adj_counts[nr][nc] -= 1

    def choose_row() -> tuple[int | None, list[int]]:
        best_row: int | None = None
        best_options: list[int] = []
        best_size = 10**9
        for r in range(puzzle.n):
            if state.row_has[r]:
                continue
            if r in givens_by_row:
                if can_place(r, givens_by_row[r]):
                    return r, [givens_by_row[r]]
                return r, []
            options = [c for c in domains[r] if can_place(r, c)]
            if not options:
                return r, []
            if len(options) < best_size:
                best_size = len(options)
                best_row = r
                best_options = options
                if best_size == 1:
                    break
        return best_row, best_options

    def bound_ok() -> bool:
        remaining_rows = [r for r in range(puzzle.n) if not state.row_has[r]]
        remaining_cols = [c for c in range(puzzle.n) if not state.col_has[c]]
        remaining_regions = [rid for rid in puzzle.region_ids if not state.region_has.get(rid, False)]

        if len(remaining_rows) != len(remaining_cols):
            return False
        if len(remaining_rows) != len(remaining_regions):
            return False

        col_has_option = {c: False for c in remaining_cols}
        region_has_option = {rid: False for rid in remaining_regions}

        for r in remaining_rows:
            has_option = False
            for c in domains[r]:
                if not can_place(r, c):
                    continue
                has_option = True
                if c in col_has_option:
                    col_has_option[c] = True
                region_id = puzzle.regions[r][c]
                if region_id in region_has_option:
                    region_has_option[region_id] = True
            if not has_option:
                return False

        if not all(col_has_option.values()):
            return False
        if not all(region_has_option.values()):
            return False
        return True

    def lcv_score(row: int, col: int) -> int:
        score = 0
        region_id = puzzle.regions[row][col]
        for rr in range(puzzle.n):
            if state.row_has[rr] or rr == row:
                continue
            for cc in domains[rr]:
                if not can_place(rr, cc):
                    continue
                if cc == col:
                    score += 1
                    continue
                if puzzle.regions[rr][cc] == region_id:
                    score += 1
                    continue
                if abs(rr - row) <= 1 and abs(cc - col) <= 1:
                    score += 1
        return score

    def dfs() -> bool | None:
        if timed_out():
            return None
        if not bound_ok():
            metrics.backtracks += 1
            return False
        row, options = choose_row()
        if row is None:
            return True
        if not options:
            metrics.backtracks += 1
            return False

        if use_lcv and len(options) > 1 and row not in givens_by_row:
            options = sorted(options, key=lambda c: lcv_score(row, c))

        for col in options:
            if timed_out():
                return None
            metrics.nodes += 1
            if not can_place(row, col):
                continue
            place(row, col)
            result = dfs()
            if result is None:
                return None
            if result:
                return True
            unplace(row, col)
        metrics.backtracks += 1
        return False

    solved = dfs()
    metrics.time_ms = timer.elapsed_ms()

    if solved is None:
        return SolveResult(
            solved=False,
            solution=None,
            metrics=metrics,
            error="Timeout: solver exceeded the time limit.",
        )

    if not solved:
        return SolveResult(
            solved=False,
            solution=None,
            metrics=metrics,
            error="No solution found under the given constraints.",
        )

    solution = QueensSolution(queens=state.queens)
    validation = validate_solution(puzzle, solution)
    if not validation.ok:
        return SolveResult(
            solved=False,
            solution=None,
            metrics=metrics,
            error=f"Solver produced an invalid solution: {validation.reason}",
        )

    return SolveResult(solved=True, solution=solution, metrics=metrics, error=None)


def solve_backtracking_bb(puzzle: QueensPuzzle, time_limit_s: float | None = None) -> SolveResult:
    """Backtracking with MRV + pruning + LCV ordering."""

    return _solve_backtracking_bb(puzzle, time_limit_s=time_limit_s, use_lcv=True)


def solve_backtracking_bb_nolcv(puzzle: QueensPuzzle, time_limit_s: float | None = None) -> SolveResult:
    """Backtracking with MRV + pruning, without LCV ordering."""

    return _solve_backtracking_bb(puzzle, time_limit_s=time_limit_s, use_lcv=False)
