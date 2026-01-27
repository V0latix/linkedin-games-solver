from __future__ import annotations

from pathlib import Path

from linkedin_game_solver.games.queens.parser import (
    QueensPuzzle,
    parse_puzzle_dict,
    parse_puzzle_file,
)
from linkedin_game_solver.games.queens.solver_baseline import solve_baseline
from linkedin_game_solver.games.queens.solver_heuristic import (
    solve_heuristic_lcv,
    solve_heuristic_simple,
)
from linkedin_game_solver.games.queens.validator import validate_solution


def _make_invalid_givens_puzzle() -> QueensPuzzle:
    n = 6
    regions = [[c for c in range(n)] for _ in range(n)]
    payload = {
        "game": "queens",
        "n": n,
        "regions": regions,
        "givens": {"queens": [[0, 0], [1, 1]], "blocked": []},
    }
    return parse_puzzle_dict(payload)


def test_solve_baseline_example() -> None:
    puzzle = parse_puzzle_file(Path("data/curated/queens/example_6x6.json"))
    result = solve_baseline(puzzle)
    assert result.solved, result.error
    assert result.solution is not None
    validation = validate_solution(puzzle, result.solution)
    assert validation.ok, validation.reason
    assert result.metrics.nodes > 0
    assert result.metrics.time_ms >= 0.0


def test_solve_heuristic_simple_example() -> None:
    puzzle = parse_puzzle_file(Path("data/curated/queens/example_6x6.json"))
    result = solve_heuristic_simple(puzzle)
    assert result.solved, result.error
    assert result.solution is not None
    validation = validate_solution(puzzle, result.solution)
    assert validation.ok, validation.reason


def test_solve_heuristic_lcv_example() -> None:
    puzzle = parse_puzzle_file(Path("data/curated/queens/example_6x6.json"))
    result = solve_heuristic_lcv(puzzle)
    assert result.solved, result.error
    assert result.solution is not None
    validation = validate_solution(puzzle, result.solution)
    assert validation.ok, validation.reason


def test_invalid_givens_return_explicit_error() -> None:
    puzzle = _make_invalid_givens_puzzle()
    result = solve_baseline(puzzle)
    assert not result.solved
    assert result.error is not None
    assert "Invalid puzzle givens" in result.error


def test_time_limit_triggers_timeout() -> None:
    puzzle = parse_puzzle_file(Path("data/curated/queens/example_6x6.json"))
    result = solve_baseline(puzzle, time_limit_s=0.0)
    assert not result.solved
    assert result.error is not None
    assert "timeout" in result.error.lower()
