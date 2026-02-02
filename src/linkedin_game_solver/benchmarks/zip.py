"""Benchmark runner and Markdown reporting for the Zip puzzle."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from linkedin_game_solver.core.types import SolveMetrics
from linkedin_game_solver.games.zip.model import ZipPuzzle
from linkedin_game_solver.games.zip.parser import parse_puzzle_file
from linkedin_game_solver.games.zip.solvers import get_solver, list_solvers


@dataclass
class BenchRow:
    puzzle_id: str
    algo: str
    solved: bool
    metrics: SolveMetrics
    error: str | None
    n: int
    source: str


@dataclass
class AlgoSummary:
    algo: str
    puzzles: int
    solved: int
    solve_rate: float
    avg_time_ms: float
    median_time_ms: float
    avg_nodes: float
    avg_backtracks: float


def _parse_algo_list(algo_csv: str) -> list[str]:
    names = [part.strip() for part in algo_csv.split(",") if part.strip()]
    if not names:
        msg = "No algorithms provided."
        raise ValueError(msg)
    for name in names:
        get_solver(name)
    return names


def _puzzle_id_from_path(path: Path) -> str:
    return path.stem


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _load_puzzles(dataset_dir: Path, limit: int | None) -> list[tuple[str, ZipPuzzle, str]]:
    if dataset_dir.is_file():
        puzzle = parse_puzzle_file(dataset_dir)
        return [(_puzzle_id_from_path(dataset_dir), puzzle, "unknown")]

    if not dataset_dir.exists():
        msg = f"Dataset directory does not exist: {dataset_dir}"
        raise ValueError(msg)

    paths = sorted(p for p in dataset_dir.glob("*.json") if p.is_file())
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        msg = f"No puzzle JSON files found in {dataset_dir}"
        raise ValueError(msg)

    puzzles: list[tuple[str, ZipPuzzle, str]] = []
    for path in paths:
        puzzle = parse_puzzle_file(path)
        puzzles.append((_puzzle_id_from_path(path), puzzle, "unknown"))
    return puzzles


def _load_puzzles_recursive(
    dataset_dir: Path,
    limit: int | None,
) -> dict[str, list[tuple[str, ZipPuzzle, str]]]:
    if dataset_dir.is_file():
        puzzle = parse_puzzle_file(dataset_dir)
        size_key = f"size_{puzzle.n}"
        return {size_key: [(_puzzle_id_from_path(dataset_dir), puzzle, "unknown")]}

    if not dataset_dir.exists():
        msg = f"Dataset directory does not exist: {dataset_dir}"
        raise ValueError(msg)

    size_dirs = sorted(p for p in dataset_dir.glob("size_*") if p.is_dir())
    if not size_dirs:
        msg = f"No size_* directories found under {dataset_dir}"
        raise ValueError(msg)

    grouped: dict[str, list[tuple[str, ZipPuzzle, str]]] = {}
    for size_dir in size_dirs:
        size_key = size_dir.name
        grouped[size_key] = _load_puzzles(size_dir, limit=limit)
    return grouped


def _parse_dataset_paths(raw: str | Path) -> list[Path]:
    raw_str = str(raw)
    parts = [part.strip() for part in raw_str.split(",") if part.strip()]
    return [Path(part) for part in parts]


def run_benchmark(
    dataset_path: Path,
    algo_csv: str,
    limit: int | None = None,
    time_limit_s: float | None = None,
) -> list[BenchRow]:
    algo_names = _parse_algo_list(algo_csv)
    puzzles = _load_puzzles(dataset_path, limit=limit)

    rows: list[BenchRow] = []
    for puzzle_id, puzzle, source in puzzles:
        for algo in algo_names:
            solver = get_solver(algo)
            result = solver(puzzle, time_limit_s=time_limit_s)
            rows.append(
                BenchRow(
                    puzzle_id=puzzle_id,
                    algo=algo,
                    solved=result.solved,
                    metrics=result.metrics,
                    error=result.error,
                    n=puzzle.n,
                    source=source,
                )
            )
    return rows


def run_benchmark_multi(
    dataset_paths: Iterable[Path],
    algo_csv: str,
    limit: int | None = None,
    time_limit_s: float | None = None,
) -> list[BenchRow]:
    rows: list[BenchRow] = []
    for path in dataset_paths:
        rows.extend(run_benchmark(path, algo_csv, limit=limit, time_limit_s=time_limit_s))
    return rows


def _summarize(rows: list[BenchRow]) -> list[AlgoSummary]:
    grouped: dict[str, list[BenchRow]] = defaultdict(list)
    for row in rows:
        grouped[row.algo].append(row)

    summaries: list[AlgoSummary] = []
    for algo, algo_rows in grouped.items():
        puzzles = len(algo_rows)
        solved_rows = [row for row in algo_rows if row.solved]
        solved = len(solved_rows)
        solve_rate = 0.0 if puzzles == 0 else solved / puzzles

        time_values = [row.metrics.time_ms for row in solved_rows]
        node_values = [float(row.metrics.nodes) for row in solved_rows]
        backtrack_values = [float(row.metrics.backtracks) for row in solved_rows]

        summaries.append(
            AlgoSummary(
                algo=algo,
                puzzles=puzzles,
                solved=solved,
                solve_rate=solve_rate,
                avg_time_ms=sum(time_values) / len(time_values) if time_values else 0.0,
                median_time_ms=_median(time_values),
                avg_nodes=sum(node_values) / len(node_values) if node_values else 0.0,
                avg_backtracks=sum(backtrack_values) / len(backtrack_values) if backtrack_values else 0.0,
            )
        )
    return sorted(summaries, key=lambda item: item.algo)


def _build_report(
    rows: list[BenchRow],
    dataset_label: str | Path,
    top_k: int,
    time_limit_s: float | None,
) -> str:
    summaries = _summarize(rows)
    lines = [
        "# Zip Benchmark Report",
        "",
        f"Dataset: `{dataset_label}`",
        f"Runs: {len(rows)}",
    ]
    if time_limit_s is not None:
        lines.append(f"Time limit per run: {time_limit_s:.2f}s")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Algo | Puzzles | Solved | Solve rate | Avg time (ms) | Median time (ms) | Avg nodes | Avg backtracks |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for summary in summaries:
        lines.append(
            f"| {summary.algo} | {summary.puzzles} | {summary.solved} | {summary.solve_rate:.1%} | "
            f"{summary.avg_time_ms:.2f} | {summary.median_time_ms:.2f} | {summary.avg_nodes:.1f} | "
            f"{summary.avg_backtracks:.1f} |"
        )

    lines.append("")
    lines.append("## Top slowest solved puzzles")
    lines.append("")
    lines.append("| Algo | Puzzle | N | Time (ms) | Nodes | Backtracks | Source |")
    lines.append("|---|---|---:|---:|---:|---:|---|")

    by_algo: dict[str, list[BenchRow]] = defaultdict(list)
    for row in rows:
        if row.solved:
            by_algo[row.algo].append(row)

    for algo, algo_rows in sorted(by_algo.items()):
        ordered = sorted(algo_rows, key=lambda row: (-row.metrics.time_ms, row.puzzle_id))
        for row in ordered[:top_k]:
            lines.append(
                f"| {algo} | {row.puzzle_id} | {row.n} | {row.metrics.time_ms:.2f} | "
                f"{row.metrics.nodes} | {row.metrics.backtracks} | {row.source} |"
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Times are measured inside each solver using `perf_counter()`.",
            "- Averages and medians are computed over solved puzzles only.",
        ]
    )
    return "\n".join(lines)


def _write_runs_jsonl(rows: list[BenchRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(rows, start=1):
            record = {
                "id": idx,
                "puzzle_id": row.puzzle_id,
                "n": row.n,
                "algo": row.algo,
                "solved": row.solved,
                "time_ms": row.metrics.time_ms,
                "nodes": row.metrics.nodes,
                "backtracks": row.metrics.backtracks,
                "timeout": bool(row.error and "timeout" in row.error.lower()),
                "source": row.source,
            }
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def run_and_report(
    dataset: str | Path,
    algo_csv: str,
    report_path: Path,
    limit: int | None = None,
    recursive: bool = False,
    top_k: int = 3,
    time_limit_s: float | None = None,
    runs_out: Path | None = None,
) -> tuple[list[BenchRow], str]:
    dataset_paths = _parse_dataset_paths(dataset)

    if recursive:
        grouped_rows: dict[str, list[BenchRow]] = defaultdict(list)
        for path in dataset_paths:
            grouped_puzzles = _load_puzzles_recursive(path, limit=limit)
            for size_key, puzzles in grouped_puzzles.items():
                rows: list[BenchRow] = []
                algo_names = _parse_algo_list(algo_csv)
                for puzzle_id, puzzle, source in puzzles:
                    for algo in algo_names:
                        solver = get_solver(algo)
                        result = solver(puzzle, time_limit_s=time_limit_s)
                        rows.append(
                            BenchRow(
                                puzzle_id=f"{size_key}/{puzzle_id}",
                                algo=algo,
                                solved=result.solved,
                                metrics=result.metrics,
                                error=result.error,
                                n=puzzle.n,
                                source=source,
                            )
                        )
                grouped_rows[size_key].extend(rows)
        all_rows = [row for rows in grouped_rows.values() for row in rows]
    else:
        all_rows = run_benchmark_multi(
            dataset_paths,
            algo_csv,
            limit=limit,
            time_limit_s=time_limit_s,
        )

    report = _build_report(all_rows, dataset_label=dataset, top_k=top_k, time_limit_s=time_limit_s)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    if runs_out is not None:
        _write_runs_jsonl(all_rows, runs_out)

    return all_rows, report


def available_algorithms() -> list[str]:
    return list_solvers()
