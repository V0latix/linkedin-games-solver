"""Command-line interface for the LinkedIn game solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from linkedin_game_solver.games.queens.generator import generate_puzzle_payload
from linkedin_game_solver.games.queens.parser import parse_puzzle_dict
from linkedin_game_solver.games.queens.renderer import render_solution
from linkedin_game_solver.games.queens.solver_baseline import solve_baseline
from linkedin_game_solver.games.queens.solver_heuristic import (
    solve_heuristic_lcv,
    solve_heuristic_simple,
)
from linkedin_game_solver.games.queens.validator import validate_solution


def _resolve_queens_solver(algo: str):
    solvers = {
        "baseline": solve_baseline,
        "heuristic_simple": solve_heuristic_simple,
        "heuristic_lcv": solve_heuristic_lcv,
    }
    solver = solvers.get(algo)
    if solver is None:
        known = ", ".join(sorted(solvers))
        msg = f"Unknown queens solver {algo!r}. Known solvers: {known}."
        raise ValueError(msg)
    return solver


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

    generate_solve = subparsers.add_parser(
        "generate-solve",
        help="Generate a puzzle, solve it, and save it if solvable.",
    )
    generate_solve.add_argument("--game", default="queens", help="Game name (only queens supported).")
    generate_solve.add_argument("--n", required=True, type=int, help="Grid size (n x n).")
    generate_solve.add_argument(
        "--algo",
        default="heuristic_lcv",
        help="Solver to use: baseline | heuristic_simple | heuristic_lcv.",
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
            parser.error("'bench' is not wired yet. Next step will implement it.")
    except ValueError as exc:
        parser.error(str(exc))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
