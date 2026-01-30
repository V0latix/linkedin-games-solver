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


def _profile_defaults(mode: str) -> dict[str, object]:
    if mode == "solve":
        return {
            "region_mode": "mixed",
            "selection": "best",
            "candidates": 30,
            "fast_unique": False,
            "fast_unique_timelimit": 0.5,
            "repair_steps": 0,
            "block_steps": 0,
            "search_until_unique": False,
            "progress_every": 500,
            "allow_multiple": False,
            "global_timeout": None,
        }
    return {
        "region_mode": "mixed",
        "selection": "first",
        "candidates": 20,
        "fast_unique": False,
        "fast_unique_timelimit": 0.5,
        "repair_steps": 0,
        "block_steps": 0,
        "search_until_unique": False,
        "progress_every": 500,
        "allow_multiple": False,
        "global_timeout": None,
    }


def _print_profiles() -> None:
    print("Profiles:")
    print("  fast:")
    print("    - region_mode: mixed")
    print("    - selection: first")
    print("    - candidates: 100")
    print("    - fast_unique: true (0.2s)")
    print("    - search_until_unique: false")
    print("  unique:")
    print("    - region_mode: constrained")
    print("    - selection: best")
    print("    - candidates: 300")
    print("    - fast_unique: false")
    print("    - repair_steps: 30")
    print("    - block_steps: 5")
    print("    - search_until_unique: true")
    print("  strict:")
    print("    - region_mode: constrained")
    print("    - selection: best")
    print("    - candidates: 500")
    print("    - fast_unique: false")
    print("    - repair_steps: 50")
    print("    - block_steps: 10")
    print("    - search_until_unique: true")


def _apply_profile(args: argparse.Namespace, mode: str) -> None:
    base = _profile_defaults(mode)
    profile = args.profile
    if profile == "doc":
        _print_profiles()
        raise SystemExit(0)
    if profile == "fast":
        base.update(
            {
                "selection": "first",
                "candidates": 100,
                "fast_unique": True,
                "fast_unique_timelimit": 0.2,
                "repair_steps": 0,
                "block_steps": 0,
                "search_until_unique": False,
                "progress_every": 200,
            }
        )
    elif profile == "unique":
        base.update(
            {
                "region_mode": "constrained",
                "selection": "best",
                "candidates": 300,
                "fast_unique": False,
                "fast_unique_timelimit": 0.5,
                "repair_steps": 30,
                "block_steps": 5,
                "search_until_unique": True,
                "progress_every": 200,
            }
        )
    elif profile == "strict":
        base.update(
            {
                "region_mode": "constrained",
                "selection": "best",
                "candidates": 500,
                "fast_unique": False,
                "repair_steps": 50,
                "block_steps": 10,
                "search_until_unique": True,
                "progress_every": 200,
            }
        )

    for key, value in base.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)


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
        print(render_puzzle(puzzle).text)
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

    _apply_profile(args, "solve")
    solver = _resolve_queens_solver(args.algo)
    max_attempts = args.max_attempts
    if not args.search_until_unique:
        max_attempts = max_attempts or 50

    region_mode = "constrained" if args.region_mode == "constrainde" else args.region_mode

    payload, known_solution = generate_puzzle_payload(
        n=args.n,
        seed=args.seed,
        ensure_unique=not args.allow_multiple,
        max_attempts=max_attempts,
        region_mode=region_mode,
        selection_mode=args.selection,
        candidates=args.candidates,
        score_algo=args.score_algo,
        search_until_unique=args.search_until_unique,
        progress_every=args.progress_every,
        fast_unique=args.fast_unique,
        fast_unique_timelimit_s=args.fast_unique_timelimit,
        repair_steps=args.repair_steps,
        block_steps=args.block_steps,
        global_time_limit_s=args.global_timeout,
    )
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

    _apply_profile(args, "dataset")
    solver = _resolve_queens_solver(args.algo)
    sizes = _parse_sizes(args.sizes)
    max_attempts = args.max_attempts
    region_mode = "constrained" if args.region_mode == "constrainde" else args.region_mode
    if not args.search_until_unique:
        max_attempts = max_attempts or (args.count * 10)

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
                payload, _known_solution = generate_puzzle_payload(
                    n=n,
                    seed=seed_value,
                    ensure_unique=not args.allow_multiple,
                    max_attempts=1 if not args.search_until_unique else max_attempts,
                    region_mode=region_mode,
                    selection_mode=args.selection,
                    candidates=args.candidates,
                    score_algo=args.score_algo,
                    search_until_unique=args.search_until_unique,
                    progress_every=args.progress_every,
                    fast_unique=args.fast_unique,
                    fast_unique_timelimit_s=args.fast_unique_timelimit,
                    repair_steps=args.repair_steps,
                    block_steps=args.block_steps,
                    global_time_limit_s=args.global_timeout,
                )
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


def _handle_fill_runs(args: argparse.Namespace) -> int:
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "scripts/fill_runs.py",
        "--manifest",
        args.manifest,
        "--runs",
        str(args.runs),
        "--timelimit",
        str(args.timelimit),
    ]
    if args.algos:
        cmd.extend(["--algos", args.algos])
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])

    completed = subprocess.run(cmd, check=False)
    return completed.returncode


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
    solve_core = generate_solve.add_argument_group("Core")
    solve_core.add_argument("--game", default="queens", help="Game name (only queens supported).")
    solve_core.add_argument("--n", required=True, type=int, help="Grid size (n x n).")
    solve_core.add_argument(
        "--algo",
        default="dlx",
        help="Solver to verify: " + ", ".join(available_algorithms()) + ".",
    )
    solve_core.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")

    solve_profile = generate_solve.add_argument_group("Profile")
    solve_profile.add_argument(
        "--profile",
        choices=["fast", "unique", "strict", "doc"],
        default=None,
        help="Preset configuration (overrides defaults unless you set a flag explicitly).",
    )

    solve_gen = generate_solve.add_argument_group("Generation")
    solve_gen.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Max attempts to generate a unique puzzle (ignored with --search-until-unique).",
    )
    solve_gen.add_argument(
        "--global-timeout",
        type=float,
        default=None,
        help="Stop generation after this many seconds and return the best candidate seen.",
    )
    solve_gen.add_argument(
        "--region-mode",
        choices=["balanced", "biased", "serpentine", "constrained", "constrainde", "mixed"],
        default=None,
        help="Region generation style.",
    )
    solve_gen.add_argument(
        "--selection",
        choices=["first", "best"],
        default=None,
        help="Select first valid puzzle or keep best-scoring.",
    )
    solve_gen.add_argument(
        "--candidates",
        type=int,
        default=None,
        help="Number of candidates to score with --selection best.",
    )
    solve_gen.add_argument(
        "--score-algo",
        default="heuristic_lcv",
        help="Algorithm used to score difficulty.",
    )

    solve_unique = generate_solve.add_argument_group("Uniqueness")
    solve_unique.add_argument(
        "--allow-multiple",
        action="store_true",
        default=None,
        help="Allow puzzles with multiple solutions (disables uniqueness check).",
    )
    solve_unique.add_argument(
        "--search-until-unique",
        action="store_true",
        default=None,
        help="Keep searching until a unique puzzle is found.",
    )
    solve_unique.add_argument(
        "--fast-unique",
        action="store_true",
        default=None,
        help="Use a fast DLX pre-check to reject obvious non-unique puzzles.",
    )
    solve_unique.add_argument(
        "--fast-unique-timelimit",
        type=float,
        default=None,
        help="Time limit in seconds for fast uniqueness pre-check.",
    )
    solve_unique.add_argument(
        "--repair-steps",
        type=int,
        default=None,
        help="Attempts to locally repair regions when non-unique.",
    )
    solve_unique.add_argument(
        "--block-steps",
        type=int,
        default=None,
        help="Attempts to add blocked cells to eliminate alternative solutions.",
    )

    solve_debug = generate_solve.add_argument_group("Output")
    solve_debug.add_argument(
        "--progress-every",
        type=int,
        default=None,
        help="Print a progress line every N candidates.",
    )
    solve_debug.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/generated/queens"),
        help="Directory to write generated puzzles.",
    )
    solve_debug.add_argument(
        "--render",
        action="store_true",
        help="Render the solved grid to the console.",
    )

    generate_dataset = subparsers.add_parser(
        "generate-dataset",
        help="Generate many solvable puzzles across multiple sizes.",
    )
    dataset_core = generate_dataset.add_argument_group("Core")
    dataset_core.add_argument("--game", default="queens", help="Game name (only queens supported).")
    dataset_core.add_argument(
        "--sizes",
        required=True,
        help="Comma-separated list of sizes, e.g., 6,7,8.",
    )
    dataset_core.add_argument(
        "--count",
        required=True,
        type=int,
        help="Number of puzzles to generate per size.",
    )
    dataset_core.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional base seed for reproducible generation.",
    )
    dataset_core.add_argument(
        "--algo",
        default="heuristic_lcv",
        help="Solver to validate puzzles: " + ", ".join(available_algorithms()) + ".",
    )
    dataset_core.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/generated/queens"),
        help="Base directory to write generated datasets.",
    )

    dataset_profile = generate_dataset.add_argument_group("Profile")
    dataset_profile.add_argument(
        "--profile",
        choices=["fast", "unique", "strict", "doc"],
        default=None,
        help="Preset configuration (overrides defaults unless you set a flag explicitly).",
    )

    dataset_gen = generate_dataset.add_argument_group("Generation")
    dataset_gen.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Maximum generation attempts per puzzle (ignored with --search-until-unique).",
    )
    dataset_gen.add_argument(
        "--global-timeout",
        type=float,
        default=None,
        help="Stop generation after this many seconds and return the best candidate seen.",
    )
    dataset_gen.add_argument(
        "--region-mode",
        choices=["balanced", "biased", "serpentine", "constrained", "constrainde", "mixed"],
        default=None,
        help="Region generation style.",
    )
    dataset_gen.add_argument(
        "--selection",
        choices=["first", "best"],
        default=None,
        help="Select first valid puzzle or keep best-scoring.",
    )
    dataset_gen.add_argument(
        "--candidates",
        type=int,
        default=None,
        help="Number of candidates to score with --selection best.",
    )
    dataset_gen.add_argument(
        "--score-algo",
        default="heuristic_lcv",
        help="Algorithm used to score difficulty.",
    )

    dataset_unique = generate_dataset.add_argument_group("Uniqueness")
    dataset_unique.add_argument(
        "--allow-multiple",
        action="store_true",
        default=None,
        help="Allow puzzles with multiple solutions (disables uniqueness check).",
    )
    dataset_unique.add_argument(
        "--search-until-unique",
        action="store_true",
        default=None,
        help="Keep searching until a unique puzzle is found.",
    )
    dataset_unique.add_argument(
        "--fast-unique",
        action="store_true",
        default=None,
        help="Use a fast DLX pre-check to reject obvious non-unique puzzles.",
    )
    dataset_unique.add_argument(
        "--fast-unique-timelimit",
        type=float,
        default=None,
        help="Time limit in seconds for fast uniqueness pre-check.",
    )
    dataset_unique.add_argument(
        "--repair-steps",
        type=int,
        default=None,
        help="Attempts to locally repair regions when non-unique.",
    )
    dataset_unique.add_argument(
        "--block-steps",
        type=int,
        default=None,
        help="Attempts to add blocked cells to eliminate alternative solutions.",
    )

    dataset_debug = generate_dataset.add_argument_group("Output")
    dataset_debug.add_argument(
        "--progress-every",
        type=int,
        default=None,
        help="Print a progress line every N candidates.",
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

    fill_runs_cmd = subparsers.add_parser(
        "fill-runs",
        help="Append missing benchmark runs to a JSONL file.",
    )
    fill_runs_cmd.add_argument(
        "--manifest",
        required=True,
        help="Comma-separated manifest paths (e.g., data/puzzles.json,data/puzzles_generated.json).",
    )
    fill_runs_cmd.add_argument(
        "--runs",
        type=Path,
        default=Path("data/benchmarks/queens_runs.jsonl"),
        help="Runs JSONL file to update.",
    )
    fill_runs_cmd.add_argument(
        "--algos",
        default=None,
        help="Comma-separated list of algorithms (default: all).",
    )
    fill_runs_cmd.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on puzzles to consider (first N per manifest).",
    )
    fill_runs_cmd.add_argument(
        "--timelimit",
        type=float,
        default=1.0,
        help="Time limit per puzzle in seconds (default: 1.0).",
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
        if args.command == "fill-runs":
            return _handle_fill_runs(args)
    except ValueError as exc:
        parser.error(str(exc))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
