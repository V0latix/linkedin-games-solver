from __future__ import annotations

from pathlib import Path

from linkedin_game_solver.cli import main
from linkedin_game_solver.games.queens.parser import parse_puzzle_file
from linkedin_game_solver.games.queens.solver_baseline import solve_baseline
from linkedin_game_solver.games.queens.validator import validate_solution


def test_cli_generate_solve_saves_a_solvable_puzzle(tmp_path: Path) -> None:
    outdir = tmp_path / "generated"
    exit_code = main(
        [
            "generate-solve",
            "--game",
            "queens",
            "--n",
            "6",
            "--algo",
            "heuristic_lcv",
            "--seed",
            "123",
            "--outdir",
            str(outdir),
        ]
    )
    assert exit_code == 0

    output_path = outdir / "queens_n6_seed123.json"
    assert output_path.exists()

    puzzle = parse_puzzle_file(output_path)
    result = solve_baseline(puzzle)
    assert result.solved, result.error
    assert result.solution is not None
    validation = validate_solution(puzzle, result.solution)
    assert validation.ok, validation.reason
