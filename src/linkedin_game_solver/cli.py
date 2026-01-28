"""Command-line interface for the LinkedIn game solver."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from linkedin_game_solver.benchmarks.queens import available_algorithms, run_and_report
from linkedin_game_solver.datasets.exporter import export_dataset
from linkedin_game_solver.datasets.normalize import normalize_dataset
from linkedin_game_solver.datasets.organize import organize_by_size
from linkedin_game_solver.datasets.unique import annotate_manifest_in_place
from linkedin_game_solver.games.queens.generator import generate_puzzle_payload
from linkedin_game_solver.games.queens.importers.samimsu import import_samimsu_dataset
from linkedin_game_solver.games.queens.parser import parse_puzzle_dict, parse_puzzle_file
from linkedin_game_solver.games.queens.renderer import render_puzzle, render_solution
from linkedin_game_solver.games.queens.solvers import get_solver
from linkedin_game_solver.games.queens.validator import validate_solution


def _resolve_queens_solver(algo: str):
    return get_solver(algo)


def _handle_solve(args: argparse.Namespace) -> int:
    if args.game != "queens":
        msg = "solve currently supports only --game queens."
        raise ValueError(msg)

    puzzle = parse_puzzle_file(args.input)
    solver = _resolve_queens_solver(args.algo)
    result = solver(puzzle, time_limit_s=args.timelimit)

    if not result.solved or result.solution is None:
        reason = result.error or "Solver failed without an explicit error."
        print(f"Solver failed: {reason}")
        return 2

    validation = validate_solution(puzzle, result.solution)
    if not validation.ok:
        print(f"Solver produced an invalid solution: {validation.reason}")
        return 3

    print(
        "metrics:",
        f"time_ms={result.metrics.time_ms:.3f}",
        f"nodes={result.metrics.nodes}",
        f"backtracks={result.metrics.backtracks}",
    )

    if args.render:
        print()
        print(render_solution(puzzle, result.solution).text)

    return 0


def _handle_render(args: argparse.Namespace) -> int:
    if args.game != "queens":
        msg = "render currently supports only --game queens."
        raise ValueError(msg)

    puzzle = parse_puzzle_file(args.input)
    print(render_puzzle(puzzle).text)
    return 0


def _handle_generate_solve(args: argparse.Namespace) -> int:
    if args.game != "queens":
        msg = "generate-solve currently supports only --game queens."
        raise ValueError(msg)

    solver = _resolve_queens_solver(args.algo)
    payload, known_solution = generate_puzzle_payload(n=args.n, seed=args.seed)
    puzzle = parse_puzzle_dict(payload)
    result = solver(puzzle)

    if not result.solved or result.solution is None:
        reason = result.error or "Solver failed without an explicit error."
        print(f"Solver failed: {reason}")
        return 2

    validation = validate_solution(puzzle, result.solution)
    if not validation.ok:
        print(f"Solver produced an invalid solution: {validation.reason}")
        return 3

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    seed_part = "noseed" if args.seed is None else f"seed{args.seed}"
    filename = f"queens_n{args.n}_{seed_part}.json"
    output_path = outdir / filename
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved solvable puzzle to: {output_path}")
    print(
        "metrics:",
        f"time_ms={result.metrics.time_ms:.3f}",
        f"nodes={result.metrics.nodes}",
        f"backtracks={result.metrics.backtracks}",
    )
    if args.render:
        print()
        print(render_solution(puzzle, result.solution).text)

    # Sanity check: the generator's known solution should also validate.
    known_validation = validate_solution(puzzle, known_solution)
    if not known_validation.ok:
        print(f"Warning: known generated solution failed validation: {known_validation.reason}")
        return 4

    return 0


def _handle_bench(args: argparse.Namespace) -> int:
    if args.game != "queens":
        msg = "bench currently supports only --game queens."
        raise ValueError(msg)

    if args.timelimit is not None and args.timelimit <= 0:
        msg = "Time limit must be a positive number of seconds."
        raise ValueError(msg)

    rows, _report = run_and_report(
        dataset=args.dataset,
        algo_csv=args.algo,
        report_path=args.report,
        limit=args.limit,
        recursive=args.recursive,
        top_k=args.top_k,
        time_limit_s=args.timelimit,
        runs_out=args.runs_out,
    )
    print(f"Benchmarked {len(rows)} runs across dataset: {args.dataset}")
    print(f"Report written to: {args.report}")
    return 0


def _parse_sizes(raw: str) -> list[int]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        msg = "No sizes provided."
        raise ValueError(msg)
    sizes: list[int] = []
    for part in parts:
        value = int(part)
        if value <= 0:
            msg = f"Invalid size {value}. Sizes must be positive."
            raise ValueError(msg)
        sizes.append(value)
    return sizes


def _next_seed(rng: random.Random | None, seed_cursor: int | None) -> tuple[int | None, int | None]:
    if rng is None:
        if seed_cursor is None:
            return None, None
        return seed_cursor, seed_cursor + 1

    seed_value = rng.randint(0, 10_000_000)
    return seed_value, seed_cursor


def _handle_generate_dataset(args: argparse.Namespace) -> int:
    if args.game != "queens":
        msg = "generate-dataset currently supports only --game queens."
        raise ValueError(msg)

    solver = _resolve_queens_solver(args.algo)
    sizes = _parse_sizes(args.sizes)
    max_attempts = args.max_attempts or (args.count * 10)

    rng = None if args.seed is not None else random.Random()
    seed_cursor = args.seed

    total_written = 0
    for n in sizes:
        outdir = args.outdir / f"size_{n}"
        outdir.mkdir(parents=True, exist_ok=True)

        for index in range(args.count):
            attempts = 0
            while attempts < max_attempts:
                seed_value, seed_cursor = _next_seed(rng, seed_cursor)
                payload, _known_solution = generate_puzzle_payload(n=n, seed=seed_value)
                puzzle = parse_puzzle_dict(payload)
                result = solver(puzzle)

                if result.solved and result.solution is not None:
                    seed_part = "noseed" if seed_value is None else f"seed{seed_value}"
                    filename = f"queens_n{n}_{index:03d}_{seed_part}.json"
                    output_path = outdir / filename
                    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    total_written += 1
                    break

                attempts += 1

            if attempts >= max_attempts:
                msg = (
                    f"Failed to generate a solvable puzzle for n={n} "
                    f"after {max_attempts} attempts."
                )
                raise ValueError(msg)

    print(f"Generated {total_written} solvable puzzles into {args.outdir}")
    return 0


def _handle_import_samimsu(args: argparse.Namespace) -> int:
    stats = import_samimsu_dataset(
        source_root=args.source,
        outdir=args.outdir,
        on_invalid=args.on_invalid,
    )
    print(
        "Imported samimsu dataset:",
        f"files={stats.total_files}",
        f"imported={stats.imported}",
        f"skipped={stats.skipped}",
    )
    return 0


def _handle_organize_dataset(args: argparse.Namespace) -> int:
    if args.game != "queens":
        msg = "organize-dataset currently supports only --game queens."
        raise ValueError(msg)

    stats = organize_by_size(
        input_dir=args.input,
        output_dir=args.outdir,
        mode=args.mode,
    )
    print(
        "Organized dataset:",
        f"files={stats.total_files}",
        f"moved={stats.moved}",
        f"skipped={stats.skipped}",
    )
    return 0


def _handle_export_dataset(args: argparse.Namespace) -> int:
    stats = export_dataset(
        input_dir=args.input,
        output_path=args.out,
        source=args.source,
        allow_duplicates=args.allow_duplicates,
    )
    print(
        "Exported dataset:",
        f"files={stats.total_files}",
        f"exported={stats.exported}",
        f"skipped={stats.skipped}",
        f"output={args.out}",
    )
    return 0


def _handle_normalize_dataset(args: argparse.Namespace) -> int:
    stats = normalize_dataset(
        input_path=args.input,
        output_path=args.out,
    )
    print(
        "Normalized dataset:",
        f"total={stats.total}",
        f"converted={stats.converted}",
        f"unchanged={stats.unchanged}",
        f"skipped={stats.skipped}",
    )
    return 0


def _handle_mark_unique(args: argparse.Namespace) -> int:
    stats = annotate_manifest_in_place(
        input_path=args.input,
        max_check=args.limit,
        time_limit_s=args.timelimit,
    )
    print(
        "Annotated puzzle uniqueness:",
        f"total={stats.total}",
        f"unique={stats.unique}",
        f"multiple={stats.non_unique}",
        f"unknown={stats.unknown}",
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lgs", description="LinkedIn puzzle solver (educational).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    solve = subparsers.add_parser("solve", help="Solve a puzzle from a JSON file.")
    solve.add_argument("--game", required=True, help="Game name (e.g., queens).")
    solve.add_argument("--algo", required=True, help="Algorithm name.")
    solve.add_argument("--input", required=True, type=Path, help="Path to puzzle JSON.")
    solve.add_argument(
        "--timelimit",
        type=float,
        default=None,
        help="Optional time limit in seconds.",
    )
    solve.add_argument(
        "--render",
        action="store_true",
        help="Render the solved grid to the console.",
    )

    render = subparsers.add_parser("render", help="Render a puzzle JSON.")
    render.add_argument("--game", required=True, help="Game name (e.g., queens).")
    render.add_argument("--input", required=True, type=Path, help="Path to puzzle JSON.")

    bench = subparsers.add_parser("bench", help="Benchmark algorithms on a dataset.")
    bench.add_argument("--game", required=True, help="Game name (e.g., queens).")
    bench.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Directory or manifest JSON. For multiple, pass comma-separated paths.",
    )
    bench.add_argument(
        "--algo",
        required=True,
        help="Comma-separated list of algorithms to benchmark.",
    )
    bench.add_argument(
        "--report",
        type=Path,
        default=Path("reports/queens_bench.md"),
        help="Path to the Markdown report to write.",
    )
    bench.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of puzzles to benchmark.",
    )
    bench.add_argument(
        "--recursive",
        action="store_true",
        help="Scan size_* subdirectories and build a sectioned report.",
    )
    bench.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of slowest puzzles to show per section.",
    )
    bench.add_argument(
        "--timelimit",
        type=float,
        default=1.0,
        help="Time limit per puzzle in seconds (default: 1.0).",
    )
    bench.add_argument(
        "--runs-out",
        type=Path,
        default=None,
        help="Write per-run results to a JSONL file.",
    )

    generate_solve = subparsers.add_parser(
        "generate-solve",
        help="Generate a puzzle, solve it, and save it if solvable.",
    )
    generate_solve.add_argument("--game", default="queens", help="Game name (only queens supported).")
    generate_solve.add_argument("--n", required=True, type=int, help="Grid size (n x n).")
    generate_solve.add_argument(
        "--algo",
        default="heuristic_lcv",
        help="Solver to use (queens): " + ", ".join(available_algorithms()) + ".",
    )
    generate_solve.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    generate_solve.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/generated/queens"),
        help="Directory to write generated puzzles.",
    )
    generate_solve.add_argument(
        "--render",
        action="store_true",
        help="Render the solved grid to the console.",
    )

    generate_dataset = subparsers.add_parser(
        "generate-dataset",
        help="Generate many solvable puzzles across multiple sizes.",
    )
    generate_dataset.add_argument("--game", default="queens", help="Game name (only queens supported).")
    generate_dataset.add_argument(
        "--sizes",
        required=True,
        help="Comma-separated list of sizes, e.g., 6,7,8.",
    )
    generate_dataset.add_argument(
        "--count",
        required=True,
        type=int,
        help="Number of puzzles to generate per size.",
    )
    generate_dataset.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional base seed for reproducible generation.",
    )
    generate_dataset.add_argument(
        "--algo",
        default="heuristic_lcv",
        help="Solver to validate puzzles: " + ", ".join(available_algorithms()) + ".",
    )
    generate_dataset.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/generated/queens"),
        help="Base directory to write generated datasets.",
    )
    generate_dataset.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Maximum generation attempts per puzzle (default: count * 10).",
    )

    import_samimsu = subparsers.add_parser(
        "import-samimsu",
        help="Import puzzles from the MIT-licensed samimsu/queens-game-linkedin repo.",
    )
    import_samimsu.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to the cloned queens-game-linkedin repo.",
    )
    import_samimsu.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/imported/queens/samimsu"),
        help="Directory to write imported puzzles.",
    )
    import_samimsu.add_argument(
        "--on-invalid",
        choices=["skip", "fail"],
        default="skip",
        help="Skip invalid puzzles or fail fast.",
    )

    organize_dataset = subparsers.add_parser(
        "organize-dataset",
        help="Organize a dataset into size_N folders.",
    )
    organize_dataset.add_argument("--game", default="queens", help="Game name (only queens supported).")
    organize_dataset.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing JSON puzzles.",
    )
    organize_dataset.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/imported/queens/samimsu_by_size"),
        help="Output directory with size_N folders.",
    )
    organize_dataset.add_argument(
        "--mode",
        choices=["move", "copy"],
        default="move",
        help="Move or copy files into size folders.",
    )

    export_dataset_cmd = subparsers.add_parser(
        "export-dataset",
        help="Export many JSON puzzles into a single manifest.",
    )
    export_dataset_cmd.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing JSON puzzles (recursive).",
    )
    export_dataset_cmd.add_argument(
        "--out",
        type=Path,
        default=Path("data/puzzles.json"),
        help="Path to the manifest JSON output.",
    )
    export_dataset_cmd.add_argument(
        "--source",
        default="imported",
        help="Source label stored in the manifest (e.g., imported, generated).",
    )
    export_dataset_cmd.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Allow duplicate puzzles (otherwise duplicates are skipped).",
    )

    normalize_dataset_cmd = subparsers.add_parser(
        "normalize-dataset",
        help="Normalize regions to matrix format (in-place by default).",
    )
    normalize_dataset_cmd.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSON file or directory.",
    )
    normalize_dataset_cmd.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON file (default: in-place).",
    )

    mark_unique_cmd = subparsers.add_parser(
        "mark-unique",
        help="Annotate puzzles with solution uniqueness (unique/multiple/unknown).",
    )
    mark_unique_cmd.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to puzzles manifest JSON (in-place update).",
    )
    mark_unique_cmd.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on puzzles to check.",
    )
    mark_unique_cmd.add_argument(
        "--timelimit",
        type=float,
        default=None,
        help="Optional time limit per puzzle in seconds.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "generate-solve":
            return _handle_generate_solve(args)
        if args.command == "solve":
            return _handle_solve(args)
        if args.command == "render":
            return _handle_render(args)
        if args.command == "bench":
            return _handle_bench(args)
        if args.command == "generate-dataset":
            return _handle_generate_dataset(args)
        if args.command == "import-samimsu":
            return _handle_import_samimsu(args)
        if args.command == "organize-dataset":
            return _handle_organize_dataset(args)
        if args.command == "export-dataset":
            return _handle_export_dataset(args)
        if args.command == "normalize-dataset":
            return _handle_normalize_dataset(args)
        if args.command == "mark-unique":
            return _handle_mark_unique(args)
    except ValueError as exc:
        parser.error(str(exc))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
