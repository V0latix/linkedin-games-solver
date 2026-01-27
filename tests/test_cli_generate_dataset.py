from __future__ import annotations

import json
from pathlib import Path

from linkedin_game_solver.cli import main
from linkedin_game_solver.games.queens.parser import parse_puzzle_file
from linkedin_game_solver.games.queens.solver_baseline import solve_baseline
from linkedin_game_solver.games.queens.validator import validate_solution


def test_cli_generate_dataset_creates_multiple_sizes(tmp_path: Path) -> None:
    outdir = tmp_path / "generated"
    exit_code = main(
        [
            "generate-dataset",
            "--game",
            "queens",
            "--sizes",
            "6,7",
            "--count",
            "2",
            "--seed",
            "10",
            "--algo",
            "heuristic_lcv",
            "--outdir",
            str(outdir),
            "--max-attempts",
            "5",
        ]
    )
    assert exit_code == 0

    size6_dir = outdir / "size_6"
    size7_dir = outdir / "size_7"
    assert size6_dir.exists()
    assert size7_dir.exists()

    size6_files = sorted(size6_dir.glob("*.json"))
    size7_files = sorted(size7_dir.glob("*.json"))
    assert len(size6_files) == 2
    assert len(size7_files) == 2

    for path in size6_files + size7_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["game"] == "queens"
        puzzle = parse_puzzle_file(path)
        result = solve_baseline(puzzle)
        assert result.solved, result.error
        assert result.solution is not None
        validation = validate_solution(puzzle, result.solution)
        assert validation.ok, validation.reason
