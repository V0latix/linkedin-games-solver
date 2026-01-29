"""DLX (Algorithm X) solver for the LinkedIn Queens puzzle."""

from __future__ import annotations

from dataclasses import dataclass

from ...core.metrics import Timer
from ...core.types import SolveMetrics, SolveResult
from .parser import Cell, QueensPuzzle, QueensSolution
from .validator import validate_solution


@dataclass(eq=False)
class _DLXNode:
    left: _DLXNode
    right: _DLXNode
    up: _DLXNode
    down: _DLXNode
    column: _ColumnNode
    row_id: Cell | None = None


@dataclass(eq=False)
class _ColumnNode(_DLXNode):
    name: str = ""
    size: int = 0
    primary: bool = True


class _DLX:
    def __init__(self) -> None:
        self.root = _ColumnNode(
            left=None,  # type: ignore[arg-type]
            right=None,  # type: ignore[arg-type]
            up=None,  # type: ignore[arg-type]
            down=None,  # type: ignore[arg-type]
            column=None,  # type: ignore[arg-type]
            name="root",
            size=0,
            primary=True,
        )
        self.root.left = self.root.right = self.root
        self.columns: dict[str, _ColumnNode] = {}

    def add_column(self, name: str, primary: bool = True) -> _ColumnNode:
        column = _ColumnNode(
            left=None,  # type: ignore[arg-type]
            right=None,  # type: ignore[arg-type]
            up=None,  # type: ignore[arg-type]
            down=None,  # type: ignore[arg-type]
            column=None,  # type: ignore[arg-type]
            name=name,
            size=0,
            primary=primary,
        )
        column.up = column.down = column

        if primary:
            column.right = self.root
            column.left = self.root.left
            self.root.left.right = column
            self.root.left = column
        else:
            column.left = column.right = column

        column.column = column
        self.columns[name] = column
        return column

    def add_row(self, row_id: Cell, column_names: list[str]) -> _DLXNode:
        nodes: list[_DLXNode] = []
        for name in column_names:
            column = self.columns[name]
            node = _DLXNode(
                left=None,  # type: ignore[arg-type]
                right=None,  # type: ignore[arg-type]
                up=None,  # type: ignore[arg-type]
                down=None,  # type: ignore[arg-type]
                column=column,
                row_id=row_id,
            )
            node.down = column
            node.up = column.up
            column.up.down = node
            column.up = node
            column.size += 1
            nodes.append(node)

        for idx, node in enumerate(nodes):
            node.right = nodes[(idx + 1) % len(nodes)]
            node.left = nodes[(idx - 1) % len(nodes)]

        return nodes[0]

    def cover(self, column: _ColumnNode) -> None:
        if column.primary:
            column.right.left = column.left
            column.left.right = column.right
        row = column.down
        while row != column:
            node = row.right
            while node != row:
                node.down.up = node.up
                node.up.down = node.down
                node.column.size -= 1
                node = node.right
            row = row.down

    def uncover(self, column: _ColumnNode) -> None:
        row = column.up
        while row != column:
            node = row.left
            while node != row:
                node.column.size += 1
                node.down.up = node
                node.up.down = node
                node = node.left
            row = row.up
        if column.primary:
            column.right.left = column
            column.left.right = column


def _adjacent(a: Cell, b: Cell) -> bool:
    ar, ac = a
    br, bc = b
    return max(abs(ar - br), abs(ac - bc)) <= 1


def _validate_givens(puzzle: QueensPuzzle) -> str | None:
    rows = set()
    cols = set()
    regions = set()
    for r, c in puzzle.givens_queens:
        if (r, c) in puzzle.blocked:
            return "given queen is on a blocked cell"
        if r in rows:
            return "multiple given queens in the same row"
        if c in cols:
            return "multiple given queens in the same column"
        region_id = puzzle.regions[r][c]
        if region_id in regions:
            return "multiple given queens in the same region"
        rows.add(r)
        cols.add(c)
        regions.add(region_id)

    givens = list(puzzle.givens_queens)
    for i, a in enumerate(givens):
        for b in givens[i + 1 :]:
            if _adjacent(a, b):
                return "given queens are adjacent"

    return None


def _build_adjacency_columns(
    puzzle: QueensPuzzle,
    candidates: list[Cell],
) -> tuple[dict[Cell, list[str]], list[str]]:
    n = puzzle.n
    cell_id = {cell: cell[0] * n + cell[1] for cell in candidates}
    adjacency_by_cell: dict[Cell, list[str]] = {cell: [] for cell in candidates}
    adjacency_columns: list[str] = []

    for r, c in candidates:
        current_id = cell_id[(r, c)]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if not (0 <= nr < n and 0 <= nc < n):
                    continue
                neighbor = (nr, nc)
                if neighbor not in cell_id:
                    continue
                neighbor_id = cell_id[neighbor]
                if current_id >= neighbor_id:
                    continue
                name = f"adj_{current_id}_{neighbor_id}"
                adjacency_columns.append(name)
                adjacency_by_cell[(r, c)].append(name)
                adjacency_by_cell[neighbor].append(name)

    return adjacency_by_cell, adjacency_columns


def count_solutions_dlx(
    puzzle: QueensPuzzle,
    limit: int = 2,
    time_limit_s: float | None = None,
) -> int:
    """Count solutions up to a limit. Returns 0/1/limit (limit == 2 means 2+).

    If a timeout occurs, returns `limit` (treat as non-unique for generators).
    """

    if limit <= 0:
        return 0

    timer = Timer()
    timer.start()
    limit_ms = None if time_limit_s is None else max(0.0, time_limit_s * 1000.0)

    givens_error = _validate_givens(puzzle)
    if givens_error:
        return 0

    candidates = [(r, c) for r in range(puzzle.n) for c in range(puzzle.n) if (r, c) not in puzzle.blocked]
    dlx = _DLX()

    for r in range(puzzle.n):
        dlx.add_column(f"row_{r}", primary=True)
    for c in range(puzzle.n):
        dlx.add_column(f"col_{c}", primary=True)
    for region_id in sorted(puzzle.region_ids):
        dlx.add_column(f"region_{region_id}", primary=True)

    adjacency_by_cell, adjacency_columns = _build_adjacency_columns(puzzle, candidates)
    for name in adjacency_columns:
        dlx.add_column(name, primary=False)

    row_nodes: dict[Cell, _DLXNode] = {}
    for r, c in candidates:
        region_id = puzzle.regions[r][c]
        columns = [f"row_{r}", f"col_{c}", f"region_{region_id}"] + adjacency_by_cell[(r, c)]
        row_nodes[(r, c)] = dlx.add_row((r, c), columns)

    def timed_out() -> bool:
        return limit_ms is not None and timer.elapsed_ms() >= limit_ms

    def cover_row(row: _DLXNode) -> None:
        node = row
        while True:
            dlx.cover(node.column)
            node = node.right
            if node == row:
                break

    def uncover_row(row: _DLXNode) -> None:
        node = row.left
        while True:
            dlx.uncover(node.column)
            if node == row:
                break
            node = node.left

    for given in sorted(puzzle.givens_queens):
        row = row_nodes.get(given)
        if row is None:
            return 0
        cover_row(row)

    def choose_column() -> _ColumnNode | None:
        col = dlx.root.right
        if col == dlx.root:
            return None
        best = col
        col = col.right
        while col != dlx.root:
            if col.size < best.size:
                best = col
            col = col.right
        return best

    count = 0

    def search() -> bool:
        nonlocal count
        if timed_out():
            return False
        column = choose_column()
        if column is None:
            count += 1
            return count < limit
        if column.size == 0:
            return True

        dlx.cover(column)
        row = column.down
        while row != column:
            if timed_out():
                dlx.uncover(column)
                return False
            node = row.right
            while node != row:
                dlx.cover(node.column)
                node = node.right

            should_continue = search()
            if not should_continue:
                dlx.uncover(column)
                return False

            node = row.left
            while node != row:
                dlx.uncover(node.column)
                node = node.left
            row = row.down

        dlx.uncover(column)
        return True

    search()
    if timed_out():
        return limit
    return min(count, limit)


def find_two_solutions_dlx(
    puzzle: QueensPuzzle,
    time_limit_s: float | None = None,
) -> list[list[Cell]]:
    """Return up to two distinct solutions (each as list of cells)."""

    timer = Timer()
    timer.start()
    limit_ms = None if time_limit_s is None else max(0.0, time_limit_s * 1000.0)

    givens_error = _validate_givens(puzzle)
    if givens_error:
        return []

    candidates = [(r, c) for r in range(puzzle.n) for c in range(puzzle.n) if (r, c) not in puzzle.blocked]
    dlx = _DLX()

    for r in range(puzzle.n):
        dlx.add_column(f"row_{r}", primary=True)
    for c in range(puzzle.n):
        dlx.add_column(f"col_{c}", primary=True)
    for region_id in sorted(puzzle.region_ids):
        dlx.add_column(f"region_{region_id}", primary=True)

    adjacency_by_cell, adjacency_columns = _build_adjacency_columns(puzzle, candidates)
    for name in adjacency_columns:
        dlx.add_column(name, primary=False)

    row_nodes: dict[Cell, _DLXNode] = {}
    for r, c in candidates:
        region_id = puzzle.regions[r][c]
        columns = [f"row_{r}", f"col_{c}", f"region_{region_id}"] + adjacency_by_cell[(r, c)]
        row_nodes[(r, c)] = dlx.add_row((r, c), columns)

    def timed_out() -> bool:
        return limit_ms is not None and timer.elapsed_ms() >= limit_ms

    def cover_row(row: _DLXNode) -> None:
        node = row
        while True:
            dlx.cover(node.column)
            node = node.right
            if node == row:
                break

    for given in sorted(puzzle.givens_queens):
        row = row_nodes.get(given)
        if row is None:
            return []
        cover_row(row)

    def choose_column() -> _ColumnNode | None:
        col = dlx.root.right
        if col == dlx.root:
            return None
        best = col
        col = col.right
        while col != dlx.root:
            if col.size < best.size:
                best = col
            col = col.right
        return best

    solutions: list[list[Cell]] = []
    solution_nodes: list[_DLXNode] = []

    def search() -> bool:
        if timed_out():
            return False
        column = choose_column()
        if column is None:
            positions = [node.row_id for node in solution_nodes if node.row_id is not None]
            solutions.append(list(positions))
            return len(solutions) < 2
        if column.size == 0:
            return True

        dlx.cover(column)
        row = column.down
        while row != column:
            if timed_out():
                dlx.uncover(column)
                return False
            solution_nodes.append(row)
            node = row.right
            while node != row:
                dlx.cover(node.column)
                node = node.right

            should_continue = search()
            if not should_continue:
                dlx.uncover(column)
                return False

            node = row.left
            while node != row:
                dlx.uncover(node.column)
                node = node.left
            solution_nodes.pop()
            row = row.down

        dlx.uncover(column)
        return True

    search()
    return solutions


def solve_dlx(puzzle: QueensPuzzle, time_limit_s: float | None = None) -> SolveResult:
    """Solve Queens using Algorithm X with Dancing Links (DLX).

    Rows represent possible queen placements (one per cell). Primary columns
    encode row/col/region constraints (exactly one). Secondary columns encode
    adjacency edges (at most one), preventing adjacent queens.
    """

    metrics = SolveMetrics()
    timer = Timer()
    timer.start()
    limit_ms = None if time_limit_s is None else max(0.0, time_limit_s * 1000.0)

    givens_error = _validate_givens(puzzle)
    if givens_error:
        metrics.time_ms = timer.elapsed_ms()
        return SolveResult(
            solved=False,
            solution=None,
            metrics=metrics,
            error=f"Invalid puzzle givens: {givens_error}",
        )

    candidates = [(r, c) for r in range(puzzle.n) for c in range(puzzle.n) if (r, c) not in puzzle.blocked]
    dlx = _DLX()

    for r in range(puzzle.n):
        dlx.add_column(f"row_{r}", primary=True)
    for c in range(puzzle.n):
        dlx.add_column(f"col_{c}", primary=True)
    for region_id in sorted(puzzle.region_ids):
        dlx.add_column(f"region_{region_id}", primary=True)

    adjacency_by_cell, adjacency_columns = _build_adjacency_columns(puzzle, candidates)
    for name in adjacency_columns:
        dlx.add_column(name, primary=False)

    row_nodes: dict[Cell, _DLXNode] = {}
    for r, c in candidates:
        region_id = puzzle.regions[r][c]
        columns = [f"row_{r}", f"col_{c}", f"region_{region_id}"] + adjacency_by_cell[(r, c)]
        row_nodes[(r, c)] = dlx.add_row((r, c), columns)

    solution_nodes: list[_DLXNode] = []

    def timed_out() -> bool:
        return limit_ms is not None and timer.elapsed_ms() >= limit_ms

    def cover_row(row: _DLXNode) -> None:
        node = row
        while True:
            dlx.cover(node.column)
            node = node.right
            if node == row:
                break

    def uncover_row(row: _DLXNode) -> None:
        node = row.left
        while True:
            dlx.uncover(node.column)
            if node == row:
                break
            node = node.left

    for given in sorted(puzzle.givens_queens):
        row = row_nodes.get(given)
        if row is None:
            metrics.time_ms = timer.elapsed_ms()
            return SolveResult(
                solved=False,
                solution=None,
                metrics=metrics,
                error="Invalid puzzle givens: given queen is on a blocked cell",
            )
        cover_row(row)
        solution_nodes.append(row)

    def choose_column() -> _ColumnNode | None:
        col = dlx.root.right
        if col == dlx.root:
            return None
        best = col
        col = col.right
        while col != dlx.root:
            if col.size < best.size:
                best = col
            col = col.right
        return best

    def search() -> bool | None:
        if timed_out():
            return None
        column = choose_column()
        if column is None:
            return True
        if column.size == 0:
            return False

        dlx.cover(column)
        row = column.down
        while row != column:
            if timed_out():
                dlx.uncover(column)
                return None
            solution_nodes.append(row)
            metrics.nodes += 1
            node = row.right
            while node != row:
                dlx.cover(node.column)
                node = node.right

            result = search()
            if result is None:
                dlx.uncover(column)
                return None
            if result:
                return True

            node = row.left
            while node != row:
                dlx.uncover(node.column)
                node = node.left
            solution_nodes.pop()
            metrics.backtracks += 1
            row = row.down

        dlx.uncover(column)
        return False

    solved = search()
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

    positions = [node.row_id for node in solution_nodes if node.row_id is not None]
    queens = [[0 for _ in range(puzzle.n)] for _ in range(puzzle.n)]
    for r, c in positions:
        queens[r][c] = 1

    solution = QueensSolution(queens=queens)
    validation = validate_solution(puzzle, solution)
    if not validation.ok:
        return SolveResult(
            solved=False,
            solution=None,
            metrics=metrics,
            error=f"Solver produced an invalid solution: {validation.reason}",
        )

    return SolveResult(solved=True, solution=solution, metrics=metrics, error=None)
