from __future__ import annotations

import json
from pathlib import Path

from linkedin_game_solver.cli import main
from linkedin_game_solver.games.queens.generator import generate_puzzle_payload


def _write_generated_puzzle(path: Path, n: int, seed: int) -> None:
    payload, _solution = generate_puzzle_payload(n=n, seed=seed)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_cli_bench_writes_markdown_report(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    _write_generated_puzzle(dataset_dir / "puzzle_a.json", n=6, seed=1)
    _write_generated_puzzle(dataset_dir / "puzzle_b.json", n=6, seed=2)

    report_path = tmp_path / "queens_bench.md"
    exit_code = main(
        [
            "bench",
            "--game",
            "queens",
            "--dataset",
            str(dataset_dir),
            "--algo",
            "baseline,heuristic_simple,heuristic_lcv",
            "--report",
            str(report_path),
        ]
    )
    assert exit_code == 0
    assert report_path.exists()

    report = report_path.read_text(encoding="utf-8")
    assert "# Queens Benchmark Report" in report
    assert "baseline" in report
    assert "heuristic_lcv" in report


def test_cli_bench_recursive_report(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    size6_dir = dataset_dir / "size_6"
    size7_dir = dataset_dir / "size_7"
    size6_dir.mkdir(parents=True)
    size7_dir.mkdir(parents=True)

    _write_generated_puzzle(size6_dir / "puzzle_a.json", n=6, seed=1)
    _write_generated_puzzle(size7_dir / "puzzle_b.json", n=7, seed=2)

    report_path = tmp_path / "queens_bench_recursive.md"
    exit_code = main(
        [
            "bench",
            "--game",
            "queens",
            "--dataset",
            str(dataset_dir),
            "--algo",
            "baseline,heuristic_simple,heuristic_lcv",
            "--report",
            str(report_path),
            "--recursive",
            "--top-k",
            "1",
        ]
    )
    assert exit_code == 0
    report = report_path.read_text(encoding="utf-8")
    assert "# Queens Benchmark Report (Recursive)" in report
    assert "Size 6" in report
    assert "Size 7" in report
    assert "Timeouts by Algorithm" in report
