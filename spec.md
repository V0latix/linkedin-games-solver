# Specification (Queens)

Ce document decrit le format des donnees et le pipeline principal du projet
`linkedin-game-solver` pour le jeu Queens.

## 1. Regles du puzzle
- Grille `n x n`
- **1 reine par ligne**
- **1 reine par colonne**
- **1 reine par region**
- **Pas de reines adjacentes** (y compris diagonales)
- Les regions sont **connectees** (garanti par le generateur)

## 2. Formats de donnees

### 2.1 Puzzle JSON (fichier individuel)
```json
{
  "game": "queens",
  "n": 6,
  "regions": [[0,0,1,1,2,2], ...],
  "givens": {
    "queens": [[r,c]],
    "blocked": [[r,c]]
  },
  "solution": "unique"
}
```

Champs:
- `regions`: matrice d'ids de region
- `givens.queens`: reines imposees (optionnel)
- `givens.blocked`: cases bloquees (optionnel)
- `solution`: `"unique"` si prouve, sinon absent

### 2.2 Manifest (dataset)
```json
{
  "game": "queens",
  "version": 1,
  "puzzles": [
    {
      "id": 1,
      "source": "imported|generated",
      "n": 6,
      "regions": [[...]],
      "givens": {"queens": [], "blocked": []},
      "solution": "unique"
    }
  ]
}
```

### 2.3 Runs JSONL (benchmarks)
Une ligne par run:
```json
{
  "id": 123,
  "puzzle_id": 42,
  "n": 6,
  "algo": "dlx",
  "solved": true,
  "time_ms": 0.34,
  "nodes": 12,
  "backtracks": 4,
  "timeout": false,
  "source": "generated"
}
```

## 3. Pipeline principal

1. **Generation** (optionnel)
   - `generate-solve` / `generate-dataset`
   - regions via flood-fill multi-source
2. **Validation** via solveurs (DLX, etc.)
3. **Export** en manifest (`export-dataset`)
4. **Marquage unicite** (`mark-unique`)
5. **Benchmarks** (`fill-runs`)
6. **Rapport DS** (`scripts/make_queens_report.py`)

## 4. Solveurs disponibles
Voir `docs/solver.md` pour les details:
- `dlx`, `baseline`, `backtracking_bb`, `backtracking_bb_nolcv`,
- `heuristic_simple`, `heuristic_lcv`, `csp_ac3`

## 5. Rapport Data Science
Commande de generation:
```bash
.venv/bin/python scripts/make_queens_report.py \
  --runs data/benchmarks/queens_runs.jsonl \
  --manifest data/puzzles_unique.json \
  --out reports/queens_report.md \
  --figdir reports/figures
```

Le rapport inclut:
- stats globales par algo
- split imported vs generated
- stats par taille `n`
- features de regions et correlations
- diagnostics (puzzles difficiles)

## 6. Reproductibilite
- Donnees: manifests + runs JSONL
- Scripts deterministes (tri stables, top-K par ordre fixe)
- Pas de modification des formats JSONL existants
