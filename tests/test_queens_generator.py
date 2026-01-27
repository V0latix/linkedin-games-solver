from __future__ import annotations

from linkedin_game_solver.games.queens.generator import generate_puzzle, generate_puzzle_payload
from linkedin_game_solver.games.queens.parser import parse_puzzle_dict
from linkedin_game_solver.games.queens.solver_baseline import solve_baseline
from linkedin_game_solver.games.queens.validator import validate_solution


def test_generate_puzzle_payload_is_parseable_and_validates_solution() -> None:
    payload, solution = generate_puzzle_payload(n=6, seed=7)
    puzzle = parse_puzzle_dict(payload)
    result = validate_solution(puzzle, solution)
    assert result.ok, result.reason


def test_generated_puzzle_is_solved_by_baseline() -> None:
    generated = generate_puzzle(n=6, seed=11)
    result = solve_baseline(generated.puzzle)
    assert result.solved, result.error
    assert result.solution is not None
    validation = validate_solution(generated.puzzle, result.solution)
    assert validation.ok, validation.reason
