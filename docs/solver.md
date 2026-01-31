# Queens Solvers — Detailed Guide

This document explains **how each solver works**, step by step, and what it depends
on. The goal is to make the implementation easy to reason about and compare.

## Shared Model (Applies to All Solvers)

All solvers enforce the same constraints:
1. **One queen per row**
2. **One queen per column**
3. **One queen per region**
4. **No adjacency** (no two queens at Chebyshev distance ≤ 1)
5. **Regions are connected** (guaranteed by the generator, not checked at solve time)

Common data structures:
- `QueensPuzzle`: grid size, region map, blocked cells, given queens.
- `QueensSolution`: binary matrix of queens.
- `SolveMetrics`: nodes, backtracks, time.

## Solver Summaries

### dlx (Algorithm X + Dancing Links)
**What it is**
- A classic **exact cover** solver using **Algorithm X** with **Dancing Links**.

**Steps**
1. Build an exact‑cover matrix:
   - **Primary constraints**: one queen per row, column, region.
   - **Secondary constraints**: adjacency pairs (no two adjacent queens).
2. Use DLX to cover/uncover columns with minimal overhead.
3. Always choose the **smallest column** (most constrained) first.

**Why it’s fast**
- O(1) cover/uncover with linked lists.
- Very strong branching heuristic (smallest column).
- Works directly on constraints rather than on the grid.

**Dependencies**
- `solver_dlx.py`
- Uses no external libs.

---

### baseline (Plain Backtracking)
**What it is**
- Simple **row‑by‑row** backtracking.

**Steps**
1. Pick the next empty row.
2. Try each column in that row.
3. Check row/column/region/adjacency validity.
4. Recurse; backtrack on failure.

**Strengths**
- Simple, reliable, easy to debug.

**Weaknesses**
- Huge search space → many nodes/backtracks.

**Dependencies**
- `solver_baseline.py`

---

### backtracking_bb (MRV + Branch‑and‑Bound + LCV)
**What it is**
- Backtracking with **better ordering** and **strong pruning**.

**Key ideas**
- **MRV** (Minimum Remaining Values): choose the row with the fewest legal moves.
- **Branch‑and‑bound pruning**: cut branches that cannot possibly finish.
- **LCV** (Least Constraining Value): try columns that block fewer future moves.

**Steps**
1. Compute legal columns per row.
2. Choose the row with **fewest options** (MRV).
3. Apply **bounds**:
   - If any remaining row has 0 options → fail.
   - If any remaining column or region has 0 possible placement → fail.
   - If counts of remaining rows/cols/regions don’t match → fail.
4. Order candidate columns by **LCV**.
5. Recurse and backtrack.

**Strengths**
- Much faster than baseline while still easy to understand.

**Weaknesses**
- Still exponential; DLX usually faster on large puzzles.

**Dependencies**
- `solver_backtracking_bb.py`

---

### backtracking_bb_nolcv
**What it is**
- Same as `backtracking_bb` **without LCV ordering**.

**Purpose**
- Ablation baseline: measure how much LCV helps.

**Dependencies**
- `solver_backtracking_bb.py`

---

### heuristic_simple
**What it is**
- Heuristic backtracking with lightweight variable ordering.

**Steps**
1. Pick the next variable using a simple heuristic.
2. Try values in a fixed order.
3. Backtrack on conflicts.

**Strengths**
- Faster than baseline on many puzzles.

**Weaknesses**
- Weaker pruning than `heuristic_lcv`.

**Dependencies**
- `solver_heuristic.py`

---

### heuristic_lcv
**What it is**
- Heuristic backtracking with **MRV + LCV** globally.

**Steps**
1. Select the most constrained variable (MRV).
2. Order values with LCV.
3. Backtrack on conflicts.

**Strengths**
- Better pruning than `heuristic_simple`.

**Weaknesses**
- Still slower than DLX on large n.

**Dependencies**
- `solver_heuristic.py`

---

### csp_ac3
**What it is**
- **Constraint Programming** baseline using AC‑3 + backtracking.

**Steps**
1. Initialize domains per variable.
2. Apply **AC‑3** arc consistency propagation.
3. Backtracking search on the reduced domains.

**Strengths**
- Good pedagogical CSP reference.
- Cuts domains early.

**Weaknesses**
- Propagation overhead can outweigh gains on easy puzzles.

**Dependencies**
- `solver_csp.py`

## Practical Guidance
- **Fastest overall**: `dlx`
- **Best pedagogical backtracking**: `backtracking_bb`
- **Compare heuristics**: `heuristic_simple` vs `heuristic_lcv`
- **CSP flavor**: `csp_ac3`
