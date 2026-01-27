"""Baseline backtracking solver for the LinkedIn Queens puzzle."""

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

    # Apply givens first so conflicts are caught early with an explicit message.
    for r, c in sorted(puzzle.givens_queens):
        if (r, c) in puzzle.blocked:
            msg = f"invalid givens: queen at {(r, c)} is blocked"
            raise ValueError(msg)
        if not _can_place(puzzle, state, r, c):
            msg = f"invalid givens: queen at {(r, c)} violates constraints"
            raise ValueError(msg)
        _place(puzzle, state, r, c)

    return state, domains, givens_by_row


def _next_row(state: _State) -> int | None:
    for r, count in enumerate(state.row_counts):
        if count == 0:
            return r
    return None


def solve_baseline(puzzle: QueensPuzzle, time_limit_s: float | None = None) -> SolveResult:
    """Solve a Queens puzzle with straightforward row-by-row backtracking.

    Intuition: place exactly one queen per row, checking constraints against the
    current partial matrix. This is exponential in the worst case but provides
    a clean baseline for comparison.
    """

    timer = Timer()
    timer.start()
    metrics = SolveMetrics()
    limit_ms = None if time_limit_s is None else max(0.0, time_limit_s * 1000.0)

    try:
        state, domains, givens_by_row = _initialize_state(puzzle)
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

        row = _next_row(state)
        if row is None:
            return True

        if row in givens_by_row:
            # Should already be placed during initialization, but guard anyway.
            col = givens_by_row[row]
            if timed_out():
                return None
            metrics.nodes += 1
            if _can_place(puzzle, state, row, col):
                _place(puzzle, state, row, col)
                result = dfs()
                if result is None:
                    return None
                if result:
                    return True
                _unplace(puzzle, state, row, col)
            metrics.backtracks += 1
            return False

        for col in sorted(domains[row]):
            if timed_out():
                return None
            metrics.nodes += 1
            if not _can_place(puzzle, state, row, col):
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
