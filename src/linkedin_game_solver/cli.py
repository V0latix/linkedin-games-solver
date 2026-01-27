"""Command-line interface for the LinkedIn game solver."""

from __future__ import annotations

import argparse
from pathlib import Path


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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # The full wiring will be added in later steps.
    if args.command == "solve":
        parser.error("'solve' is not wired yet. Next step will implement it.")
    if args.command == "render":
        parser.error("'render' is not wired yet. Next step will implement it.")
    if args.command == "bench":
        parser.error("'bench' is not wired yet. Next step will implement it.")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
