from __future__ import annotations

import json
from pathlib import Path

from linkedin_game_solver.cli import main
from linkedin_game_solver.games.queens.generator import generate_puzzle_payload


def _write_payload(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_cli_solve_and_render(tmp_path: Path) -> None:
    payload, _solution = generate_puzzle_payload(n=6, seed=123)
    puzzle_path = tmp_path / "puzzle.json"
    _write_payload(puzzle_path, payload)

    exit_code = main(
        [
            "solve",
            "--game",
            "queens",
            "--algo",
            "heuristic_lcv",
            "--input",
            str(puzzle_path),
        ]
    )
    assert exit_code == 0

    exit_code = main(
        [
            "render",
            "--game",
            "queens",
            "--input",
            str(puzzle_path),
        ]
    )
    assert exit_code == 0
