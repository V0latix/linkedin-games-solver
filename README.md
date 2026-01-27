# linkedin-game-solver

Educational puzzle solver framework in Python. The first implemented game is LinkedIn **Queens**.

## Quickstart

```bash
python -m pytest
ruff check .
```

## CLI

The CLI entrypoint is `lgs` (wiring in progress):

```bash
lgs solve --game queens --algo baseline --input data/curated/queens/example_6x6.json
```
