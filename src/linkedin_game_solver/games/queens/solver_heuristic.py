"""Heuristic backtracking solvers for the LinkedIn Queens puzzle."""

from __future__ import annotations

from dataclasses import dataclass

from ...core.metrics import Timer
from ...core.types import SolveMetrics, SolveResult
from .parser import Cell, QueensPuzzle, QueensSolution, empty_solution
from .validator import validate_solution


@dataclass
class _State:
    queens: list[list[int]]
    row_counts: list[int]
    col_counts: list[int]
    region_counts: dict[int, int]
    queen_positions: set[Cell]


def _region_id(puzzle: QueensPuzzle, r: int, c: int) -> int:
    return puzzle.regions[r][c]


def _adjacent(a: Cell, b: Cell) -> bool:
    ar, ac = a
    br, bc = b
    return max(abs(ar - br), abs(ac - bc)) <= 1


def _can_place(puzzle: QueensPuzzle, state: _State, r: int, c: int) -> bool:
    if (r, c) in puzzle.blocked:
        return False
    if state.row_counts[r] >= 1:
        return False
    if state.col_counts[c] >= 1:
        return False

    region_id = _region_id(puzzle, r, c)
    if state.region_counts.get(region_id, 0) >= 1:
        return False

    return all(not _adjacent(pos, (r, c)) for pos in state.queen_positions)


def _place(puzzle: QueensPuzzle, state: _State, r: int, c: int) -> None:
    state.queens[r][c] = 1
    state.row_counts[r] += 1
    state.col_counts[c] += 1
    region_id = _region_id(puzzle, r, c)
    state.region_counts[region_id] = state.region_counts.get(region_id, 0) + 1
    state.queen_positions.add((r, c))


def _unplace(puzzle: QueensPuzzle, state: _State, r: int, c: int) -> None:
    state.queens[r][c] = 0
    state.row_counts[r] -= 1
    state.col_counts[c] -= 1
    region_id = _region_id(puzzle, r, c)
    state.region_counts[region_id] -= 1
    state.queen_positions.remove((r, c))


def _build_row_domains(puzzle: QueensPuzzle) -> tuple[list[set[int]], dict[int, int]]:
    n = puzzle.n
    domains: list[set[int]] = []
    givens_by_row: dict[int, int] = {}

    for r, c in puzzle.givens_queens:
        if r in givens_by_row and givens_by_row[r] != c:
            msg = f"invalid givens: multiple queens in row {r}"
            raise ValueError(msg)
        givens_by_row[r] = c

    for r in range(n):
        blocked_cols = {c for (rr, c) in puzzle.blocked if rr == r}
        if r in givens_by_row:
            domains.append({givens_by_row[r]})
        else:
            domains.append({c for c in range(n) if c not in blocked_cols})

    return domains, givens_by_row


def _initialize_state(puzzle: QueensPuzzle) -> tuple[_State, list[set[int]], dict[int, int]]:
    n = puzzle.n
    solution = empty_solution(n)
    state = _State(
        queens=solution.queens,
        row_counts=[0 for _ in range(n)],
        col_counts=[0 for _ in range(n)],
        region_counts={region_id: 0 for region_id in puzzle.region_ids},
        queen_positions=set(),
    )

    domains, givens_by_row = _build_row_domains(puzzle)

    for r, c in sorted(puzzle.givens_queens):
        if (r, c) in puzzle.blocked:
            msg = f"invalid givens: queen at {(r, c)} is blocked"
            raise ValueError(msg)
        if not _can_place(puzzle, state, r, c):
            msg = f"invalid givens: queen at {(r, c)} violates constraints"
            raise ValueError(msg)
        _place(puzzle, state, r, c)

    return state, domains, givens_by_row


def _unfilled_rows(state: _State) -> list[int]:
    return [r for r, count in enumerate(state.row_counts) if count == 0]


def _allowed_columns(puzzle: QueensPuzzle, state: _State, row: int, domain: set[int]) -> list[int]:
    return [col for col in domain if _can_place(puzzle, state, row, col)]


def _select_row_mrv(
    puzzle: QueensPuzzle,
    state: _State,
    domains: list[set[int]],
) -> tuple[int | None, dict[int, list[int]]]:
    candidates = _unfilled_rows(state)
    if not candidates:
        return None, {}

    allowed_by_row: dict[int, list[int]] = {}
    best_row: int | None = None
    best_size: int | None = None

    for row in candidates:
        allowed = _allowed_columns(puzzle, state, row, domains[row])
        allowed_by_row[row] = allowed
        size = len(allowed)
        if best_size is None or size < best_size or (size == best_size and row < best_row):
            best_row = row
            best_size = size

    return best_row, allowed_by_row


def _forward_check(
    puzzle: QueensPuzzle,
    state: _State,
    domains: list[set[int]],
    placed_row: int,
    placed_col: int,
) -> bool:
    _place(puzzle, state, placed_row, placed_col)
    try:
        for row in _unfilled_rows(state):
            allowed = _allowed_columns(puzzle, state, row, domains[row])
            if not allowed:
                return False
        return True
    finally:
        _unplace(puzzle, state, placed_row, placed_col)


def _order_columns_simple(columns: list[int]) -> list[int]:
    return sorted(columns)


def _order_columns_lcv(
    puzzle: QueensPuzzle,
    state: _State,
    domains: list[set[int]],
    row: int,
    columns: list[int],
) -> list[int]:
    scores: list[tuple[int, int]] = []

    for col in columns:
        _place(puzzle, state, row, col)
        try:
            remaining_options = 0
            dead_end = False
            for other_row in _unfilled_rows(state):
                allowed = _allowed_columns(puzzle, state, other_row, domains[other_row])
                if not allowed:
                    dead_end = True
                    break
                remaining_options += len(allowed)
            score = -1 if dead_end else remaining_options
            scores.append((score, col))
        finally:
            _unplace(puzzle, state, row, col)

    scores.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return [col for _, col in scores]


def _finalize_result(puzzle: QueensPuzzle, state: _State, metrics: SolveMetrics) -> SolveResult:
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


def solve_heuristic_simple(puzzle: QueensPuzzle, time_limit_s: float | None = None) -> SolveResult:
    """MRV row selection with simple column ordering.

    Intuition: always pick the row with the fewest valid moves (MRV) to reduce
    branching early. Complexity remains exponential but often with far fewer
    explored nodes than the baseline.
    """

    timer = Timer()
    timer.start()
    metrics = SolveMetrics()
    limit_ms = None if time_limit_s is None else max(0.0, time_limit_s * 1000.0)

    try:
        state, domains, _givens_by_row = _initialize_state(puzzle)
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

    def dfs() -> bool | None:
        if timed_out():
            return None

        row, allowed_by_row = _select_row_mrv(puzzle, state, domains)
        if row is None:
            return True

        allowed = allowed_by_row[row]
        if not allowed:
            metrics.backtracks += 1
            return False

        for col in _order_columns_simple(allowed):
            if timed_out():
                return None
            metrics.nodes += 1
            if not _can_place(puzzle, state, row, col):
                continue
            if not _forward_check(puzzle, state, domains, row, col):
                continue
            _place(puzzle, state, row, col)
            result = dfs()
            if result is None:
                return None
            if result:
                return True
            _unplace(puzzle, state, row, col)

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

    return _finalize_result(puzzle, state, metrics)


def solve_heuristic_lcv(puzzle: QueensPuzzle, time_limit_s: float | None = None) -> SolveResult:
    """MRV row selection with LCV ordering and forward checking.

    Intuition: combine MRV (hardest row first) with LCV (least constraining
    column first) and forward checking to prune dead ends early.
    """

    timer = Timer()
    timer.start()
    metrics = SolveMetrics()
    limit_ms = None if time_limit_s is None else max(0.0, time_limit_s * 1000.0)

    try:
        state, domains, _givens_by_row = _initialize_state(puzzle)
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

    def dfs() -> bool | None:
        if timed_out():
            return None

        row, allowed_by_row = _select_row_mrv(puzzle, state, domains)
        if row is None:
            return True

        allowed = allowed_by_row[row]
        if not allowed:
            metrics.backtracks += 1
            return False

        for col in _order_columns_lcv(puzzle, state, domains, row, allowed):
            if timed_out():
                return None
            metrics.nodes += 1
            if not _can_place(puzzle, state, row, col):
                continue
            if not _forward_check(puzzle, state, domains, row, col):
                continue
            _place(puzzle, state, row, col)
            result = dfs()
            if result is None:
                return None
            if result:
                return True
            _unplace(puzzle, state, row, col)

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

    return _finalize_result(puzzle, state, metrics)
