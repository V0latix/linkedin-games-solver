from __future__ import annotations

from pathlib import Path

from linkedin_game_solver.games.queens.parser import parse_puzzle_file


def test_parse_example_file() -> None:
    puzzle = parse_puzzle_file(Path("data/curated/queens/example_6x6.json"))
    assert puzzle.game == "queens"
    assert puzzle.n == 6
    assert len(puzzle.region_ids) == puzzle.n
    assert (0, 1) in puzzle.blocked
    assert (2, 2) in puzzle.blocked
