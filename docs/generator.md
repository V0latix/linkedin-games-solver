# Queens Generator Notes

This document explains how the Queens puzzle generator evolved to search for **unique** solutions and what strategies were tried. It is a practical, engineering-focused record of the approach, constraints, and trade-offs.

## Goal

Generate **Queens** puzzles (region partitions) with **exactly one solution** for sizes `n >= 8` (target `n >= 13`) **without givens** (no pre-placed queens). We allow **blocked cells** as a later constraint if needed.

Queens rules recap:
- Grid `n x n`
- Exactly **1 queen per row, column, and region**
- No adjacent queens (including diagonals)
- Regions must be **connected**

## High-Level Pipeline

1. **Solution-first**: generate a valid full queen placement.
2. **Region partitioning**: grow regions from queen seeds (multi-source flood-fill).
3. **Uniqueness check**: use DLX (Algorithm X) to count solutions up to 2.
4. **Accept / reject**:
   - If count == 1: accept.
   - Else: try **repairs** or **block steps** (optional).
   - Otherwise reject and retry.

## 1) Solution-First Placement

We generate a full solution by backtracking:
- Place one queen per row.
- Ensure column uniqueness.
- Enforce **non-adjacency** (Chebyshev distance > 1).

This gives a complete assignment with `n` queens, one per row and column.

## 2) Region Partitioning (Flood-Fill)

We build regions by **growing from the queen positions**. Each queen cell is a **region seed**, guaranteeing one queen per region.

Growth is done by multi-source expansion:
- Each region has a frontier (4-neighbors).
- We expand one cell at a time until all cells are assigned.
- We keep regions **connected** by construction.

### Region Modes

The generator supports multiple modes (argument `--region-mode`):

- `balanced`: favor smaller regions to keep sizes even.
- `biased`: prefer continuing in the same direction (elongated shapes).
- `serpentine`: row-wise snake growth (more entangled shapes).
- `constrained`: **compact around the seed** to limit ambiguity (prioritizes candidates that keep row/col span small and minimize row/col combinations).
- `mixed`: cycles all modes (includes `constrained`).

`constrained` is the most important for uniqueness attempts because compact regions reduce alternative placements.

## 3) Uniqueness with DLX

We use **DLX (Algorithm X)** as the uniqueness oracle.

### `count_solutions_dlx(puzzle, limit=2)`
- Returns `0`, `1`, or `2` (meaning `>=2`).
- Stops as soon as 2 solutions are found.
- On timeout, it returns `2` (treated as non-unique).

This is the **final check**. Every accepted puzzle must pass this count.

## 4) Generation Strategy & Options

### `selection_mode`
Controls how candidates are chosen:
- `first`: accept the first candidate that passes uniqueness.
- `best`: evaluate multiple candidates and keep the “hardest”.

### `fast_unique`
Optional pre-check:
- Runs a **short DLX count** with a small time limit (default 0.5s).
- If it already finds 2 solutions quickly, we skip the candidate.
- Final decision still uses the full count.

### `search_until_unique`
If enabled, the generator keeps searching without a hard cap (unless `--max-attempts` is set).

### Global timeouts + best-candidate fallback
`global_time_limit_s` (CLI: `--global-timeout`) sets a wall-clock limit for generation.
If the limit is reached before a unique puzzle is found, the generator returns the
best candidate seen so far. The fallback prefers low-ambiguity region layouts
based on row/column spread (and in `selection=best`, it prefers the best-scoring
candidate when available).

### Progress logs
Printed every `--progress-every N` candidates, including:
- total candidates
- repair attempts
- block attempts

## 5) Repair Attempts (Local Region Swaps)

When multiple solutions are found, we try to **adjust regions locally** to eliminate one solution:

### Simple repair (1-cell move)
Move a conflicting cell from one region to a neighbor region **if both regions stay connected**.

### Multi-swap repair (2-cell swap)
Swap two boundary cells between two regions (again, connectivity preserved).

Repairs are iterative: after each successful move/swap, DLX is rerun to fetch the
current pair of solutions, so each step targets the actual difference in the
updated puzzle.

Repairs are bounded by `--repair-steps`.

**Observation:** In practice, repairs alone rarely yield uniqueness for n≥8.

## 6) Blocked-Cell Strategy (Accepted)

To improve uniqueness without givens, we allow **blocked cells**:

Procedure:
1. DLX returns two distinct solutions `A` and `B`.
2. Choose a queen cell that is in `B` but not in `A`.
3. Add this cell to `givens.blocked`.
4. Re-test uniqueness.

This preserves the “main” solution while removing an alternative.

Controlled by:
- `--block-steps`

## Practical Notes

- **Uniqueness without givens is extremely rare** for n≥8.
- `constrained` regions help but are not sufficient alone.
- `block_steps` is currently the most effective way to get uniqueness at larger sizes.

## Example CLI

```bash
.venv/bin/python -m linkedin_game_solver.cli generate-solve \
  --game queens --n 10 \
  --region-mode constrained \
  --selection best \
  --candidates 300 \
  --search-until-unique \
  --fast-unique --fast-unique-timelimit 0.5 \
  --repair-steps 30 \
  --block-steps 5 \
  --progress-every 200
```
