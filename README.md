# linkedin-game-solver

Framework educatif de resolution de puzzles en Python. Le premier jeu implemente
est **Queens** (LinkedIn).

## Demarrage rapide

```bash
python -m pytest
ruff check .
```

## CLI

L'entree CLI est `lgs`.

Generer et resoudre un puzzle :

```bash
lgs generate-solve --n 6 --seed 123 --render
```

Generer un dataset (plusieurs tailles) :

```bash
lgs generate-dataset --sizes 6,7,8 --count 20 --seed 100 --algo heuristic_lcv
```

Benchmarker un dataset :

```bash
lgs bench \
  --game queens \
  --dataset data/generated/queens \
  --algo baseline,heuristic_simple,heuristic_lcv,dlx,csp_ac3,backtracking_bb \
  --report reports/queens_bench_all.md \
  --recursive \
  --top-k 3
```

Generer le rapport data science enrichi :

```bash
.venv/bin/python scripts/make_queens_report.py \
  --runs data/benchmarks/queens_runs.jsonl \
  --manifest data/puzzles_unique.json \
  --out reports/queens_report.md \
  --figdir reports/figures
```

## Algorithms (resume pedagogique)

Documentation complete : `docs/solver.md`

- **dlx** : exact cover + Algorithm X + Dancing Links (reference perf).
- **baseline** : backtracking simple ligne par ligne.
- **backtracking_bb** : MRV + pruning + LCV local.
- **backtracking_bb_nolcv** : ablation du LCV.
- **heuristic_simple** : backtracking heuristique leger.
- **heuristic_lcv** : MRV + LCV global.
- **csp_ac3** : propagation de contraintes (AC-3) + backtracking.

## Metriques (benchmarks)

- **nodes** : nombre de placements testes
- **backtracks** : nombre de retours en arriere
- **time_ms** : temps ecoule

## Notes

- Projet educatif uniquement. Pas d'automatisation en production sur LinkedIn.
