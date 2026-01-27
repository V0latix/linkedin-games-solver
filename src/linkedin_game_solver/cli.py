"""Command-line interface for the LinkedIn game solver."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from linkedin_game_solver.benchmarks.queens import available_algorithms, run_and_report
from linkedin_game_solver.games.queens.generator import generate_puzzle_payload
from linkedin_game_solver.games.queens.parser import parse_puzzle_dict
from linkedin_game_solver.games.queens.renderer import render_solution
from linkedin_game_solver.games.queens.solvers import get_solver
from linkedin_game_solver.games.queens.validator import validate_solution


def _resolve_queens_solver(algo: str):
    return get_solver(algo)


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
        dataset_dir=args.dataset,
        algo_csv=args.algo,
        report_path=args.report,
        limit=args.limit,
        recursive=args.recursive,
        top_k=args.top_k,
        time_limit_s=args.timelimit,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lgs", description="LinkedIn puzzle solver (educational).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    solve = subparsers.add_parser("solve", help="Solve a puzzle from a JSON file.")
    solve.add_argument("--game", required=True, help="Game name (e.g., queens).")
    solve.add_argument("--algo", required=True, help="Algorithm name.")
    solve.add_argument("--input", required=True, type=Path, help="Path to puzzle JSON.")

    render = subparsers.add_parser("render", help="Render a puzzle JSON.")
    render.add_argument("--input", required=True, type=Path, help="Path to puzzle JSON.")

    bench = subparsers.add_parser("bench", help="Benchmark algorithms on a dataset.")
    bench.add_argument("--game", required=True, help="Game name (e.g., queens).")
    bench.add_argument("--dataset", required=True, type=Path, help="Directory of puzzles.")
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
        default=None,
        help="Time limit per puzzle in seconds (e.g., 0.5).",
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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "generate-solve":
            return _handle_generate_solve(args)
        if args.command == "solve":
            parser.error("'solve' is not wired yet. Next step will implement it.")
        if args.command == "render":
            parser.error("'render' is not wired yet. Next step will implement it.")
        if args.command == "bench":
            return _handle_bench(args)
        if args.command == "generate-dataset":
            return _handle_generate_dataset(args)
    except ValueError as exc:
        parser.error(str(exc))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
