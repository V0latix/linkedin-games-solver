from __future__ import annotations

from linkedin_game_solver.games.zip.generator import generate_zip_puzzle_payload
from linkedin_game_solver.games.zip.parser import parse_puzzle_dict
from linkedin_game_solver.games.zip.validator import validate_solution


def test_zip_generator_returns_valid_payload() -> None:
    result = generate_zip_puzzle_payload(
        n=4,
        seed=123,
        checkpoints=3,
        ensure_unique=False,
        max_attempts=10,
        path_timelimit_s=0.2,
    )
    puzzle = parse_puzzle_dict(result.payload)
    validation = validate_solution(puzzle, result.solution)
    assert validation.ok, validation.reason


def test_zip_generator_unique_with_full_checkpoints() -> None:
    result = generate_zip_puzzle_payload(
        n=3,
        seed=7,
        checkpoints=9,
        ensure_unique=True,
        unique_timelimit_s=0.5,
        max_attempts=5,
        path_timelimit_s=0.2,
    )
    puzzle = parse_puzzle_dict(result.payload)
    validation = validate_solution(puzzle, result.solution)
    assert validation.ok, validation.reason
