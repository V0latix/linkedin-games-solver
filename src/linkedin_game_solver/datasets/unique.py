"""Check uniqueness of Queens puzzles using DLX."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from linkedin_game_solver.games.queens.parser import parse_puzzle_dict
from linkedin_game_solver.games.queens.solver_dlx import count_solutions_dlx


@dataclass
class UniqueStats:
    total: int = 0
    unique: int = 0
    non_unique: int = 0
    skipped: int = 0
    unknown: int = 0


def _load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("game") != "queens" or "puzzles" not in payload:
        msg = f"Invalid manifest: {path}"
        raise ValueError(msg)
    return payload


def filter_unique_manifest(
    input_path: Path,
    output_path: Path,
    max_check: int | None = None,
) -> UniqueStats:
    payload = _load_manifest(input_path)
    puzzles = payload["puzzles"]

    filtered: list[dict] = []
    stats = UniqueStats()

    for puzzle in puzzles:
        if max_check is not None and stats.total >= max_check:
            break
        stats.total += 1
        try:
            p = parse_puzzle_dict(
                {
                    "game": "queens",
                    "n": puzzle["n"],
                    "regions": puzzle["regions"],
                    "givens": puzzle.get("givens", {"queens": [], "blocked": []}),
                }
            )
            count = count_solutions_dlx(p, limit=2)
            if count == 1:
                filtered.append(puzzle)
                stats.unique += 1
            else:
                stats.non_unique += 1
        except Exception:
            stats.skipped += 1

    payload["puzzles"] = filtered
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return stats


def filter_unique_manifests(
    inputs: Iterable[Path],
    output_dir: Path,
    max_check: int | None = None,
) -> dict[str, UniqueStats]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, UniqueStats] = {}
    for path in inputs:
        out = output_dir / path.name
        results[path.name] = filter_unique_manifest(path, out, max_check=max_check)
    return results


def annotate_manifest_in_place(
    input_path: Path,
    max_check: int | None = None,
    time_limit_s: float | None = None,
) -> UniqueStats:
    payload = _load_manifest(input_path)
    puzzles = payload["puzzles"]

    stats = UniqueStats()

    for puzzle in puzzles:
        if max_check is not None and stats.total >= max_check:
            break
        stats.total += 1
        try:
            parsed = parse_puzzle_dict(
                {
                    "game": "queens",
                    "n": puzzle["n"],
                    "regions": puzzle["regions"],
                    "givens": puzzle.get("givens", {"queens": [], "blocked": []}),
                }
            )
            count = count_solutions_dlx(parsed, limit=2, time_limit_s=time_limit_s)
            if count == 1:
                puzzle["solution"] = "unique"
                stats.unique += 1
            elif count >= 2:
                puzzle["solution"] = "multiple"
                stats.non_unique += 1
            else:
                puzzle["solution"] = "unknown"
                stats.unknown += 1
        except Exception:
            puzzle["solution"] = "unknown"
            stats.unknown += 1
            stats.skipped += 1

    input_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return stats
