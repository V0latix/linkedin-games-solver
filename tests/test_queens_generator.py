from __future__ import annotations

from linkedin_game_solver.games.queens.generator import generate_puzzle, generate_puzzle_payload
from linkedin_game_solver.games.queens.parser import parse_puzzle_dict
from linkedin_game_solver.games.queens.solver_baseline import solve_baseline
from linkedin_game_solver.games.queens.solver_dlx import count_solutions_dlx
from linkedin_game_solver.games.queens.validator import validate_solution


def test_generate_puzzle_payload_is_parseable_and_validates_solution() -> None:
    payload, solution = generate_puzzle_payload(n=6, seed=7, ensure_unique=False)
    puzzle = parse_puzzle_dict(payload)
    result = validate_solution(puzzle, solution)
    assert result.ok, result.reason


def test_generated_puzzle_is_solved_by_baseline() -> None:
    generated = generate_puzzle(n=6, seed=11, ensure_unique=False)
    result = solve_baseline(generated.puzzle)
    assert result.solved, result.error
    assert result.solution is not None
    validation = validate_solution(generated.puzzle, result.solution)
    assert validation.ok, validation.reason


def test_generate_puzzle_payload_fast_unique_is_parseable() -> None:
    payload, solution = generate_puzzle_payload(
        n=6,
        seed=9,
        ensure_unique=False,
        fast_unique=True,
        fast_unique_timelimit_s=0.5,
    )
    puzzle = parse_puzzle_dict(payload)
    result = validate_solution(puzzle, solution)
    assert result.ok, result.reason


def test_generate_puzzle_payload_block_steps() -> None:
    payload, solution = generate_puzzle_payload(
        n=6,
        seed=5,
        ensure_unique=False,
        block_steps=2,
    )
    puzzle = parse_puzzle_dict(payload)
    result = validate_solution(puzzle, solution)
    assert result.ok, result.reason


def test_generated_unique_n13() -> None:
    import os

    if os.environ.get("RUN_SLOW_TESTS") != "1":
        return

    payload, _solution = generate_puzzle_payload(
        n=13,
        ensure_unique=True,
        max_attempts=5000,
        time_limit_s=2.0,
        region_mode="constrained",
        selection_mode="first",
        search_until_unique=False,
        fast_unique=True,
        fast_unique_timelimit_s=0.5,
        progress_every=None,
    )
    puzzle = parse_puzzle_dict(payload)
    assert count_solutions_dlx(puzzle, limit=2, time_limit_s=2.0) == 1
