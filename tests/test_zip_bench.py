from __future__ import annotations

from pathlib import Path

from linkedin_game_solver.benchmarks.zip import run_and_report


def test_zip_bench_generates_report_and_runs(tmp_path: Path) -> None:
    report_path = tmp_path / "zip_bench.md"
    runs_out = tmp_path / "zip_runs.jsonl"

    rows, report = run_and_report(
        dataset=Path("data/curated/zip"),
        algo_csv="baseline",
        report_path=report_path,
        limit=1,
        time_limit_s=1.0,
        runs_out=runs_out,
    )

    assert rows
    assert report
    assert report_path.exists()
    assert runs_out.exists()
