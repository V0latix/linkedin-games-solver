import { useEffect, useMemo, useRef, useState } from 'react';

type View = 'queens' | 'zip';

type CellKey = string;

type QueensPuzzle = {
  id: number;
  n: number;
  regions: number[][];
  givens: {
    queens: [number, number][];
    blocked: [number, number][];
  };
};

type QueensPuzzleNormalized = QueensPuzzle & {
  givensQueens: Set<CellKey>;
  blocked: Set<CellKey>;
  regionIds: number[];
  regionColors: Map<number, string>;
};

type ZipPuzzle = {
  id: number;
  n: number;
  numbers: { k: number; r: number; c: number }[];
  walls: { r1: number; c1: number; r2: number; c2: number }[];
};

type ZipPuzzleNormalized = ZipPuzzle & {
  numberByKey: Map<CellKey, number>;
  numberToKey: Map<number, CellKey>;
  wallsSet: Set<string>;
  neighbors: Map<CellKey, CellKey[]>;
};

const QUEENS_PALETTE = [
  '#e7f27c',
  '#ff8a73',
  '#b9e6a5',
  '#e6e3e0',
  '#9fc4ff',
  '#c7b0ea',
  '#ffd2a1',
  '#f3b7d4',
  '#b8f1ed',
  '#d9f7a1',
];

const ZIP_COLOR = 'var(--zip-path)';
const ZIP_STROKE = 'var(--zip-stroke)';
const ZIP_NODE = 'var(--zip-node)';

function cellKey(r: number, c: number): CellKey {
  return `${r},${c}`;
}

function parseCellKey(key: CellKey): [number, number] {
  const [r, c] = key.split(',').map(Number);
  return [r, c];
}

function normalizeQueensPuzzle(puzzle: QueensPuzzle): QueensPuzzleNormalized {
  const regionIds = new Set<number>();
  puzzle.regions.forEach((row) => row.forEach((value) => regionIds.add(value)));

  const givensQueens = new Set<CellKey>(
    puzzle.givens?.queens?.map(([r, c]) => cellKey(r, c)) ?? []
  );
  const blocked = new Set<CellKey>(
    puzzle.givens?.blocked?.map(([r, c]) => cellKey(r, c)) ?? []
  );

  const ids = Array.from(regionIds).sort((a, b) => a - b);
  const regionColors = new Map<number, string>();
  ids.forEach((id, index) => {
    regionColors.set(id, QUEENS_PALETTE[index % QUEENS_PALETTE.length]);
  });

  return {
    ...puzzle,
    givensQueens,
    blocked,
    regionIds: ids,
    regionColors,
  };
}

function normalizeWallKey(a: CellKey, b: CellKey) {
  return a < b ? `${a}|${b}` : `${b}|${a}`;
}

function normalizeZipPuzzle(puzzle: ZipPuzzle): ZipPuzzleNormalized {
  const numberByKey = new Map<CellKey, number>();
  const numberToKey = new Map<number, CellKey>();
  puzzle.numbers.forEach(({ k, r, c }) => {
    const key = cellKey(r, c);
    numberByKey.set(key, k);
    numberToKey.set(k, key);
  });

  const wallsSet = new Set<string>();
  puzzle.walls.forEach(({ r1, c1, r2, c2 }) => {
    wallsSet.add(normalizeWallKey(cellKey(r1, c1), cellKey(r2, c2)));
  });

  const neighbors = new Map<CellKey, CellKey[]>();
  for (let r = 0; r < puzzle.n; r += 1) {
    for (let c = 0; c < puzzle.n; c += 1) {
      const key = cellKey(r, c);
      const list: CellKey[] = [];
      const candidates: [number, number][] = [
        [r - 1, c],
        [r + 1, c],
        [r, c - 1],
        [r, c + 1],
      ];
      candidates.forEach(([nr, nc]) => {
        if (nr < 0 || nc < 0 || nr >= puzzle.n || nc >= puzzle.n) {
          return;
        }
        const neighborKey = cellKey(nr, nc);
        if (!wallsSet.has(normalizeWallKey(key, neighborKey))) {
          list.push(neighborKey);
        }
      });
      neighbors.set(key, list);
    }
  }

  return {
    ...puzzle,
    numberByKey,
    numberToKey,
    wallsSet,
    neighbors,
  };
}

function computeQueensConflicts(puzzle: QueensPuzzleNormalized, queens: Set<CellKey>) {
  const n = puzzle.n;
  const positions: { r: number; c: number; key: CellKey }[] = [];
  queens.forEach((key) => {
    const [r, c] = parseCellKey(key);
    positions.push({ r, c, key });
  });

  const rowCounts = Array.from({ length: n }, () => 0);
  const colCounts = Array.from({ length: n }, () => 0);
  const regionCounts = new Map<number, number>();

  positions.forEach(({ r, c }) => {
    rowCounts[r] += 1;
    colCounts[c] += 1;
    const regionId = puzzle.regions[r][c];
    regionCounts.set(regionId, (regionCounts.get(regionId) ?? 0) + 1);
  });

  const conflicts = new Set<CellKey>();

  positions.forEach(({ r, c, key }) => {
    const regionId = puzzle.regions[r][c];
    if (rowCounts[r] > 1 || colCounts[c] > 1 || (regionCounts.get(regionId) ?? 0) > 1) {
      conflicts.add(key);
    }
  });

  const queenKeys = new Set(positions.map((pos) => pos.key));
  positions.forEach(({ r, c, key }) => {
    for (let dr = -1; dr <= 1; dr += 1) {
      for (let dc = -1; dc <= 1; dc += 1) {
        if (dr === 0 && dc === 0) {
          continue;
        }
        const nr = r + dr;
        const nc = c + dc;
        if (nr < 0 || nc < 0 || nr >= n || nc >= n) {
          continue;
        }
        if (queenKeys.has(cellKey(nr, nc))) {
          conflicts.add(key);
          conflicts.add(cellKey(nr, nc));
        }
      }
    }
  });

  positions.forEach(({ key }) => {
    if (puzzle.blocked.has(key)) {
      conflicts.add(key);
    }
  });

  return conflicts;
}

function validateQueensSolution(puzzle: QueensPuzzleNormalized, queens: Set<CellKey>) {
  const n = puzzle.n;
  const positions: { r: number; c: number; key: CellKey }[] = [];
  queens.forEach((key) => {
    const [r, c] = parseCellKey(key);
    positions.push({ r, c, key });
  });

  for (const key of puzzle.givensQueens) {
    if (!queens.has(key)) {
      return { ok: false, reason: 'Les reines données doivent rester en place.' };
    }
  }

  if (positions.length !== n) {
    return { ok: false, reason: `Il faut exactement ${n} reines.` };
  }

  const rowCounts = Array.from({ length: n }, () => 0);
  const colCounts = Array.from({ length: n }, () => 0);
  const regionCounts = new Map<number, number>();

  positions.forEach(({ r, c }) => {
    rowCounts[r] += 1;
    colCounts[c] += 1;
    const regionId = puzzle.regions[r][c];
    regionCounts.set(regionId, (regionCounts.get(regionId) ?? 0) + 1);
  });

  if (rowCounts.some((count) => count !== 1)) {
    return { ok: false, reason: 'Chaque ligne doit contenir exactement une reine.' };
  }

  if (colCounts.some((count) => count !== 1)) {
    return { ok: false, reason: 'Chaque colonne doit contenir exactement une reine.' };
  }

  for (const id of puzzle.regionIds) {
    if ((regionCounts.get(id) ?? 0) !== 1) {
      return { ok: false, reason: 'Chaque région doit contenir exactement une reine.' };
    }
  }

  const queenKeys = new Set(positions.map((pos) => pos.key));
  for (const { r, c } of positions) {
    for (let dr = -1; dr <= 1; dr += 1) {
      for (let dc = -1; dc <= 1; dc += 1) {
        if (dr === 0 && dc === 0) {
          continue;
        }
        const nr = r + dr;
        const nc = c + dc;
        if (nr < 0 || nc < 0 || nr >= n || nc >= n) {
          continue;
        }
        if (queenKeys.has(cellKey(nr, nc))) {
          return { ok: false, reason: 'Les reines ne peuvent pas être adjacentes.' };
        }
      }
    }
  }

  return { ok: true, reason: null };
}

function solveQueens(puzzle: QueensPuzzleNormalized) {
  const n = puzzle.n;
  const fixedCols = Array(n).fill(-1);

  puzzle.givensQueens.forEach((key) => {
    const [r, c] = parseCellKey(key);
    fixedCols[r] = c;
  });

  const usedCols = new Set<number>();
  const usedRegions = new Set<number>();
  const cols = Array(n).fill(-1);

  function backtrack(row: number, prevCol: number | null): boolean {
    if (row === n) {
      return true;
    }
    const fixed = fixedCols[row];
    const candidates = fixed >= 0 ? [fixed] : Array.from({ length: n }, (_, i) => i);

    for (const col of candidates) {
      const key = cellKey(row, col);
      if (puzzle.blocked.has(key)) {
        continue;
      }
      if (usedCols.has(col)) {
        continue;
      }
      const regionId = puzzle.regions[row][col];
      if (usedRegions.has(regionId)) {
        continue;
      }
      if (prevCol !== null && Math.abs(col - prevCol) <= 1) {
        continue;
      }

      usedCols.add(col);
      usedRegions.add(regionId);
      cols[row] = col;

      if (backtrack(row + 1, col)) {
        return true;
      }

      usedCols.delete(col);
      usedRegions.delete(regionId);
      cols[row] = -1;
    }

    return false;
  }

  if (!backtrack(0, null)) {
    return null;
  }
  return cols;
}

function validateZipPrefix(puzzle: ZipPuzzleNormalized, path: CellKey[]) {
  const seen = new Set<CellKey>();
  let lastKey: CellKey | null = null;
  let maxNumberSeen = 0;

  for (const key of path) {
    if (seen.has(key)) {
      return { ok: false, reason: 'Une case est visitée plusieurs fois.', maxNumberSeen };
    }
    seen.add(key);

    if (lastKey) {
      const neighbors = puzzle.neighbors.get(lastKey) ?? [];
      if (!neighbors.includes(key)) {
        return {
          ok: false,
          reason: 'Le chemin doit rester adjacent et respecter les murs.',
          maxNumberSeen,
        };
      }
    }

    const number = puzzle.numberByKey.get(key);
    if (number !== undefined) {
      if (number !== maxNumberSeen + 1) {
        return { ok: false, reason: "Les nombres doivent être visités dans l'ordre.", maxNumberSeen };
      }
      maxNumberSeen = number;
    }

    lastKey = key;
  }

  return { ok: true, reason: null, maxNumberSeen };
}

function orderZipNeighbors(
  neighbors: CellKey[],
  puzzle: ZipPuzzleNormalized,
  visited: Set<CellKey>,
  maxNumberSeen: number
) {
  const nextNumber = maxNumberSeen + 1;
  return neighbors
    .filter((key) => !visited.has(key))
    .map((key) => {
      const number = puzzle.numberByKey.get(key);
      const priority = number === nextNumber ? -100 : 0;
      const degree = (puzzle.neighbors.get(key) ?? []).filter((n) => !visited.has(n)).length;
      return { key, score: priority + degree };
    })
    .sort((a, b) => a.score - b.score)
    .map((entry) => entry.key);
}

function solveZip(puzzle: ZipPuzzleNormalized, prefixPath: CellKey[]) {
  const total = puzzle.n * puzzle.n;
  let path = [...prefixPath];
  let visited = new Set(path);

  if (path.length === 0) {
    const startKey = puzzle.numberToKey.get(1);
    if (!startKey) {
      return null;
    }
    path = [startKey];
    visited = new Set(path);
  }

  const validation = validateZipPrefix(puzzle, path);
  if (!validation.ok) {
    return null;
  }

  function dfs(currentPath: CellKey[], currentVisited: Set<CellKey>, maxNumberSeen: number): CellKey[] | null {
    if (currentPath.length === total) {
      return [...currentPath];
    }

    const currentKey = currentPath[currentPath.length - 1];
    const neighbors = puzzle.neighbors.get(currentKey) ?? [];
    const ordered = orderZipNeighbors(neighbors, puzzle, currentVisited, maxNumberSeen);

    for (const nextKey of ordered) {
      const number = puzzle.numberByKey.get(nextKey);
      if (number !== undefined && number !== maxNumberSeen + 1) {
        continue;
      }
      const nextMax = number ?? maxNumberSeen;
      currentVisited.add(nextKey);
      currentPath.push(nextKey);

      const result = dfs(currentPath, currentVisited, nextMax);
      if (result) {
        return result;
      }

      currentPath.pop();
      currentVisited.delete(nextKey);
    }

    return null;
  }

  return dfs(path, visited, validation.maxNumberSeen);
}

function validateZipSolution(puzzle: ZipPuzzleNormalized, path: CellKey[]) {
  const n = puzzle.n;
  if (path.length !== n * n) {
    return { ok: false, reason: 'Le chemin doit couvrir toute la grille.' };
  }

  const seen = new Set<CellKey>();
  for (let i = 0; i < path.length; i += 1) {
    const key = path[i];
    if (seen.has(key)) {
      return { ok: false, reason: 'Une case est visitée plusieurs fois.' };
    }
    seen.add(key);

    if (i > 0) {
      const prev = path[i - 1];
      const neighbors = puzzle.neighbors.get(prev) ?? [];
      if (!neighbors.includes(key)) {
        return { ok: false, reason: 'Le chemin doit rester adjacent et respecter les murs.' };
      }
    }
  }

  const positions = new Map<CellKey, number>();
  path.forEach((key, index) => {
    positions.set(key, index);
  });

  const maxNumber = Math.max(...Array.from(puzzle.numberToKey.keys()));
  let lastIndex = -1;
  for (let k = 1; k <= maxNumber; k += 1) {
    const key = puzzle.numberToKey.get(k);
    if (!key) {
      return { ok: false, reason: `Nombre ${k} manquant.` };
    }
    const idx = positions.get(key);
    if (idx === undefined || idx <= lastIndex) {
      return { ok: false, reason: "Les nombres doivent être visités dans l'ordre." };
    }
    lastIndex = idx;
  }

  return { ok: true, reason: null };
}

function directionFromTo(fromKey: CellKey, toKey: CellKey) {
  const [fr, fc] = parseCellKey(fromKey);
  const [tr, tc] = parseCellKey(toKey);
  if (tr === fr - 1 && tc === fc) return 'n';
  if (tr === fr + 1 && tc === fc) return 's';
  if (tr === fr && tc === fc - 1) return 'w';
  if (tr === fr && tc === fc + 1) return 'e';
  return null;
}

function zipPathStyle(segments: Set<string>) {
  if (segments.size === 0) {
    return {};
  }
  const gradients: string[] = [];
  const sizes: string[] = [];
  const positions: string[] = [];

  if (segments.has('n')) {
    gradients.push(`linear-gradient(${ZIP_COLOR}, ${ZIP_COLOR})`);
    sizes.push(`${ZIP_STROKE} 50%`);
    positions.push('center top');
  }
  if (segments.has('s')) {
    gradients.push(`linear-gradient(${ZIP_COLOR}, ${ZIP_COLOR})`);
    sizes.push(`${ZIP_STROKE} 50%`);
    positions.push('center bottom');
  }
  if (segments.has('w')) {
    gradients.push(`linear-gradient(${ZIP_COLOR}, ${ZIP_COLOR})`);
    sizes.push(`50% ${ZIP_STROKE}`);
    positions.push('left center');
  }
  if (segments.has('e')) {
    gradients.push(`linear-gradient(${ZIP_COLOR}, ${ZIP_COLOR})`);
    sizes.push(`50% ${ZIP_STROKE}`);
    positions.push('right center');
  }

  if (segments.has('n') && !segments.has('s')) {
    gradients.push(`radial-gradient(circle, ${ZIP_COLOR} 0 60%, transparent 61%)`);
    sizes.push(`${ZIP_STROKE} ${ZIP_STROKE}`);
    positions.push('center top');
  }
  if (segments.has('s') && !segments.has('n')) {
    gradients.push(`radial-gradient(circle, ${ZIP_COLOR} 0 60%, transparent 61%)`);
    sizes.push(`${ZIP_STROKE} ${ZIP_STROKE}`);
    positions.push('center bottom');
  }
  if (segments.has('w') && !segments.has('e')) {
    gradients.push(`radial-gradient(circle, ${ZIP_COLOR} 0 60%, transparent 61%)`);
    sizes.push(`${ZIP_STROKE} ${ZIP_STROKE}`);
    positions.push('left center');
  }
  if (segments.has('e') && !segments.has('w')) {
    gradients.push(`radial-gradient(circle, ${ZIP_COLOR} 0 60%, transparent 61%)`);
    sizes.push(`${ZIP_STROKE} ${ZIP_STROKE}`);
    positions.push('right center');
  }

  gradients.push(`radial-gradient(circle, ${ZIP_COLOR} 0 60%, transparent 61%)`);
  sizes.push(`${ZIP_NODE} ${ZIP_NODE}`);
  positions.push('center center');

  return {
    backgroundImage: gradients.join(', '),
    backgroundSize: sizes.join(', '),
    backgroundPosition: positions.join(', '),
    backgroundRepeat: 'no-repeat',
  };
}

export default function App() {
  const [view, setView] = useState<View>('queens');
  const [queensPuzzles, setQueensPuzzles] = useState<QueensPuzzleNormalized[]>([]);
  const [queensPuzzle, setQueensPuzzle] = useState<QueensPuzzleNormalized | null>(null);
  const [queensIndex, setQueensIndex] = useState<number | null>(null);
  const [queensMarks, setQueensMarks] = useState<Set<CellKey>>(new Set());
  const [queensPlaced, setQueensPlaced] = useState<Set<CellKey>>(new Set());
  const [queensStatus, setQueensStatus] = useState('Chargement…');
  const [queensStatusType, setQueensStatusType] = useState<'ok' | 'warn' | 'error' | null>(null);

  const [zipPuzzles, setZipPuzzles] = useState<ZipPuzzleNormalized[]>([]);
  const [zipPuzzle, setZipPuzzle] = useState<ZipPuzzleNormalized | null>(null);
  const [zipIndex, setZipIndex] = useState<number | null>(null);
  const [zipPath, setZipPath] = useState<CellKey[]>([]);
  const [zipStatus, setZipStatus] = useState('Chargement…');
  const [zipStatusType, setZipStatusType] = useState<'ok' | 'warn' | 'error' | null>(null);

  const queensDrag = useRef({
    active: false,
    moved: false,
    startKey: null as CellKey | null,
    lastKey: null as CellKey | null,
    markingStarted: false,
  });

  const zipDrag = useRef({
    active: false,
    lastKey: null as CellKey | null,
  });

  useEffect(() => {
    let cancelled = false;
    fetch('./data/queens_unique.json')
      .then((res) => res.json())
      .then((data) => {
        if (cancelled) return;
        const puzzles = (data.puzzles as QueensPuzzle[]).map(normalizeQueensPuzzle);
        setQueensPuzzles(puzzles);
        const index = puzzles.length ? Math.floor(Math.random() * puzzles.length) : null;
        if (index !== null) {
          const puzzle = puzzles[index];
          setQueensPuzzle(puzzle);
          setQueensIndex(index);
          setQueensPlaced(new Set(puzzle.givensQueens));
        }
      })
      .catch(() => {
        if (!cancelled) {
          setQueensStatus('Impossible de charger les puzzles Queens.');
          setQueensStatusType('error');
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    fetch('./data/zip_unique.json')
      .then((res) => res.json())
      .then((data) => {
        if (cancelled) return;
        const puzzles = (data.puzzles as ZipPuzzle[]).map(normalizeZipPuzzle);
        setZipPuzzles(puzzles);
        const index = puzzles.length ? Math.floor(Math.random() * puzzles.length) : null;
        if (index !== null) {
          setZipPuzzle(puzzles[index]);
          setZipIndex(index);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setZipStatus('Impossible de charger les puzzles Zip.');
          setZipStatusType('error');
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const queensConflicts = useMemo(() => {
    if (!queensPuzzle) return new Set<CellKey>();
    return computeQueensConflicts(queensPuzzle, queensPlaced);
  }, [queensPuzzle, queensPlaced]);

  useEffect(() => {
    if (!queensPuzzle) return;
    const total = queensPuzzle.n;
    const placed = queensPlaced.size;
    const solved =
      placed === total &&
      queensConflicts.size === 0 &&
      validateQueensSolution(queensPuzzle, queensPlaced).ok;
    if (solved) {
      setQueensStatus("C'est bien, t'as réussi. Clique sur \"Puzzle suivant\".");
      setQueensStatusType('ok');
      return;
    }
    let message = `Puzzle ${queensPuzzle.id} · ${queensPuzzle.n}x${queensPuzzle.n}. Reines: ${placed}/${total}.`;
    if (queensConflicts.size > 0) {
      message += ` Conflits: ${queensConflicts.size}.`;
      setQueensStatusType('warn');
    } else {
      setQueensStatusType(null);
    }
    setQueensStatus(message);
  }, [queensPuzzle, queensPlaced, queensConflicts]);

  useEffect(() => {
    if (!zipPuzzle) return;
    const total = zipPuzzle.n * zipPuzzle.n;
    const placed = zipPath.length;
    const solved = placed === total && validateZipSolution(zipPuzzle, zipPath).ok;
    if (solved) {
      setZipStatus("C'est bien, t'as réussi. Clique sur \"Puzzle suivant\".");
      setZipStatusType('ok');
      return;
    }
    setZipStatus(`Puzzle ${zipPuzzle.id} · ${zipPuzzle.n}x${zipPuzzle.n}. Parcours: ${placed}/${total}.`);
    setZipStatusType(null);
  }, [zipPuzzle, zipPath]);

  function pickNextIndex(current: number | null, list: unknown[]) {
    if (!list.length) return null;
    if (list.length === 1) return 0;
    let index = Math.floor(Math.random() * list.length);
    while (index === current) {
      index = Math.floor(Math.random() * list.length);
    }
    return index;
  }

  function resetQueens(puzzle: QueensPuzzleNormalized) {
    setQueensPlaced(new Set(puzzle.givensQueens));
    setQueensMarks(new Set());
  }

  function cycleQueensCell(key: CellKey, puzzle: QueensPuzzleNormalized) {
    if (puzzle.blocked.has(key) || puzzle.givensQueens.has(key)) return;
    const nextQueens = new Set(queensPlaced);
    const nextMarks = new Set(queensMarks);
    if (nextQueens.has(key)) {
      nextQueens.delete(key);
    } else if (nextMarks.has(key)) {
      nextMarks.delete(key);
      nextQueens.add(key);
    } else {
      nextMarks.add(key);
    }
    setQueensPlaced(nextQueens);
    setQueensMarks(nextMarks);
  }

  function markQueensCell(key: CellKey, puzzle: QueensPuzzleNormalized) {
    if (puzzle.blocked.has(key) || puzzle.givensQueens.has(key)) return false;
    if (queensPlaced.has(key)) return false;
    const nextMarks = new Set(queensMarks);
    nextMarks.add(key);
    setQueensMarks(nextMarks);
    return true;
  }

  function handleQueensPointerDown(event: React.PointerEvent) {
    if (!queensPuzzle) return;
    const target = (event.target as HTMLElement).closest<HTMLButtonElement>('.cell');
    if (!target) return;
    const r = Number(target.dataset.r);
    const c = Number(target.dataset.c);
    const key = cellKey(r, c);
    if (queensPuzzle.blocked.has(key) || queensPuzzle.givensQueens.has(key)) return;
    event.preventDefault();
    markQueensCell(key, queensPuzzle);
    queensDrag.current = {
      active: true,
      moved: false,
      startKey: key,
      lastKey: key,
      markingStarted: false,
    };
    target.setPointerCapture(event.pointerId);
  }

  function handleQueensPointerMove(event: React.PointerEvent) {
    if (!queensPuzzle || !queensDrag.current.active) return;
    const element = document.elementFromPoint(event.clientX, event.clientY) as HTMLElement | null;
    const cell = element?.closest<HTMLButtonElement>('.cell');
    if (!cell) return;
    const r = Number(cell.dataset.r);
    const c = Number(cell.dataset.c);
    const key = cellKey(r, c);
    if (key === queensDrag.current.lastKey) return;
    queensDrag.current.moved = true;
    queensDrag.current.lastKey = key;

    if (!queensDrag.current.markingStarted && queensDrag.current.startKey) {
      markQueensCell(queensDrag.current.startKey, queensPuzzle);
      queensDrag.current.markingStarted = true;
    }
    markQueensCell(key, queensPuzzle);
  }

  function handleQueensPointerUp(event: React.PointerEvent) {
    if (!queensPuzzle || !queensDrag.current.active) return;
    if (!queensDrag.current.moved && queensDrag.current.startKey) {
      cycleQueensCell(queensDrag.current.startKey, queensPuzzle);
    }
    (event.target as HTMLElement).releasePointerCapture(event.pointerId);
    queensDrag.current = { active: false, moved: false, startKey: null, lastKey: null, markingStarted: false };
  }

  function handleQueensHint() {
    if (!queensPuzzle) return;
    const solution = solveQueens(queensPuzzle);
    if (!solution) {
      setQueensStatus('Indice indisponible pour ce puzzle.');
      setQueensStatusType('error');
      return;
    }
    const solutionKeys = solution.map((col, row) => cellKey(row, col));
    for (const key of solutionKeys) {
      if (!queensPlaced.has(key)) {
        const nextQueens = new Set(queensPlaced);
        nextQueens.add(key);
        const nextMarks = new Set(queensMarks);
        nextMarks.delete(key);
        setQueensPlaced(nextQueens);
        setQueensMarks(nextMarks);
        setQueensStatus('Indice appliqué : une reine a été ajoutée.');
        setQueensStatusType('ok');
        return;
      }
    }
  }

  function handleQueensVerify() {
    if (!queensPuzzle) return;
    const result = validateQueensSolution(queensPuzzle, queensPlaced);
    if (result.ok) {
      setQueensStatus('Bravo, solution correcte !');
      setQueensStatusType('ok');
    } else {
      setQueensStatus(result.reason ?? 'Solution incorrecte.');
      setQueensStatusType('error');
    }
  }

  function handleQueensNext() {
    const index = pickNextIndex(queensIndex, queensPuzzles);
    if (index === null) return;
    const puzzle = queensPuzzles[index];
    setQueensPuzzle(puzzle);
    setQueensIndex(index);
    resetQueens(puzzle);
  }

  function handleZipPointerDown(event: React.PointerEvent) {
    if (!zipPuzzle) return;
    const target = (event.target as HTMLElement).closest<HTMLDivElement>('.zip-cell');
    if (!target) return;
    event.preventDefault();
    zipDrag.current.active = true;
    zipDrag.current.lastKey = null;
    target.setPointerCapture(event.pointerId);
    handleZipInteraction(target);
  }

  function handleZipPointerMove(event: React.PointerEvent) {
    if (!zipPuzzle || !zipDrag.current.active) return;
    const element = document.elementFromPoint(event.clientX, event.clientY) as HTMLElement | null;
    const cell = element?.closest<HTMLDivElement>('.zip-cell');
    if (!cell) return;
    handleZipInteraction(cell);
  }

  function handleZipPointerUp(event: React.PointerEvent) {
    if (!zipPuzzle || !zipDrag.current.active) return;
    (event.target as HTMLElement).releasePointerCapture(event.pointerId);
    zipDrag.current.active = false;
    zipDrag.current.lastKey = null;
  }

  function handleZipInteraction(cell: HTMLDivElement) {
    if (!zipPuzzle) return;
    const r = Number(cell.dataset.r);
    const c = Number(cell.dataset.c);
    const key = cellKey(r, c);
    if (key === zipDrag.current.lastKey) return;
    zipDrag.current.lastKey = key;

    if (zipPath.length === 0) {
      setZipPath([key]);
      return;
    }

    const lastKey = zipPath[zipPath.length - 1];
    if (key === lastKey) return;
    const existingIndex = zipPath.indexOf(key);
    if (existingIndex !== -1) {
      setZipPath(zipPath.slice(0, existingIndex + 1));
      return;
    }
    const neighbors = zipPuzzle.neighbors.get(lastKey) ?? [];
    if (!neighbors.includes(key)) {
      return;
    }
    setZipPath([...zipPath, key]);
  }

  function handleZipHint() {
    if (!zipPuzzle) return;
    const validation = validateZipPrefix(zipPuzzle, zipPath);
    if (!validation.ok) {
      setZipStatus(validation.reason ?? 'Chemin invalide.');
      setZipStatusType('error');
      return;
    }

    if (zipPath.length === 0) {
      const startKey = zipPuzzle.numberToKey.get(1);
      if (startKey) {
        setZipPath([startKey]);
        setZipStatus('Indice appliqué : départ placé.');
        setZipStatusType('ok');
        return;
      }
    }

    const solution = solveZip(zipPuzzle, zipPath);
    if (!solution) {
      setZipStatus('Indice indisponible pour ce chemin.');
      setZipStatusType('error');
      return;
    }
    if (solution.length <= zipPath.length) {
      setZipStatus('Toutes les cases sont déjà remplies.');
      setZipStatusType('ok');
      return;
    }
    setZipPath([...zipPath, solution[zipPath.length]]);
    setZipStatus('Indice appliqué : étape suivante ajoutée.');
    setZipStatusType('ok');
  }

  function handleZipVerify() {
    if (!zipPuzzle) return;
    const result = validateZipSolution(zipPuzzle, zipPath);
    if (result.ok) {
      setZipStatus('Bravo, solution correcte !');
      setZipStatusType('ok');
    } else {
      setZipStatus(result.reason ?? 'Solution incorrecte.');
      setZipStatusType('error');
    }
  }

  function handleZipNext() {
    const index = pickNextIndex(zipIndex, zipPuzzles);
    if (index === null) return;
    setZipPuzzle(zipPuzzles[index]);
    setZipIndex(index);
    setZipPath([]);
  }

  function handleZipReset() {
    setZipPath([]);
  }

  const queensGrid = useMemo(() => {
    if (!queensPuzzle) return null;
    const rows = [];
    for (let r = 0; r < queensPuzzle.n; r += 1) {
      const cells = [];
      for (let c = 0; c < queensPuzzle.n; c += 1) {
        const key = cellKey(r, c);
        const isBlocked = queensPuzzle.blocked.has(key);
        const isGiven = queensPuzzle.givensQueens.has(key);
        const isQueen = queensPlaced.has(key) || isGiven;
        const isMark = queensMarks.has(key);
        const hasConflict = queensConflicts.has(key);
        const style = { background: queensPuzzle.regionColors.get(queensPuzzle.regions[r][c]) } as const;
        cells.push(
          <button
            key={key}
            type="button"
            className={[
              'cell',
              isBlocked ? 'blocked' : '',
              isGiven ? 'given' : '',
              isQueen ? 'queen' : '',
              !isQueen && isMark ? 'mark' : '',
              hasConflict ? 'conflict' : '',
            ]
              .filter(Boolean)
              .join(' ')}
            data-r={r}
            data-c={c}
            disabled={isBlocked || isGiven}
            style={style}
          >
            {isBlocked ? '×' : isQueen ? '♛' : isMark ? '×' : ''}
          </button>
        );
      }
      rows.push(cells);
    }
    return rows;
  }, [queensPuzzle, queensPlaced, queensMarks, queensConflicts]);

  const zipGrid = useMemo(() => {
    if (!zipPuzzle) return null;
    const pathSet = new Set(zipPath);
    const indexByKey = new Map<CellKey, number>();
    zipPath.forEach((key, index) => indexByKey.set(key, index));
    const rows = [];
    for (let r = 0; r < zipPuzzle.n; r += 1) {
      const cells = [];
      for (let c = 0; c < zipPuzzle.n; c += 1) {
        const key = cellKey(r, c);
        const number = zipPuzzle.numberByKey.get(key);
        const isPath = pathSet.has(key);
        const isActive = zipPath[zipPath.length - 1] === key;
        const segments = new Set<string>();
        const index = indexByKey.get(key);
        if (index !== undefined) {
          if (index > 0) {
            const prevKey = zipPath[index - 1];
            const dir = directionFromTo(key, prevKey);
            if (dir) segments.add(dir);
          }
          if (index < zipPath.length - 1) {
            const nextKey = zipPath[index + 1];
            const dir = directionFromTo(key, nextKey);
            if (dir) segments.add(dir);
          }
        }
        const north = r > 0 ? cellKey(r - 1, c) : null;
        const south = r < zipPuzzle.n - 1 ? cellKey(r + 1, c) : null;
        const west = c > 0 ? cellKey(r, c - 1) : null;
        const east = c < zipPuzzle.n - 1 ? cellKey(r, c + 1) : null;
        const wallTop = north && zipPuzzle.wallsSet.has(normalizeWallKey(key, north));
        const wallBottom = south && zipPuzzle.wallsSet.has(normalizeWallKey(key, south));
        const wallLeft = west && zipPuzzle.wallsSet.has(normalizeWallKey(key, west));
        const wallRight = east && zipPuzzle.wallsSet.has(normalizeWallKey(key, east));

        cells.push(
          <div
            key={key}
            className={[
              'zip-cell',
              isPath ? 'path' : '',
              isActive ? 'active' : '',
              number !== undefined ? 'has-number' : '',
              wallTop ? 'wall-top' : '',
              wallBottom ? 'wall-bottom' : '',
              wallLeft ? 'wall-left' : '',
              wallRight ? 'wall-right' : '',
            ]
              .filter(Boolean)
              .join(' ')}
            data-r={r}
            data-c={c}
            style={isPath ? zipPathStyle(segments) : undefined}
          >
            {number !== undefined ? (
              <div className={`zip-number ${isActive ? 'active' : ''}`}>{number}</div>
            ) : null}
          </div>
        );
      }
      rows.push(cells);
    }
    return rows;
  }, [zipPuzzle, zipPath]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-dot" />
          <div>
            <div className="brand-title">LinkedIn Games</div>
            <div className="brand-subtitle">Puzzles jouables - Queens &amp; Zip</div>
          </div>
        </div>
        <nav className="tabs">
          <button
            className={`tab ${view === 'queens' ? 'active' : ''}`}
            type="button"
            onClick={() => setView('queens')}
          >
            Queens
          </button>
          <button
            className={`tab ${view === 'zip' ? 'active' : ''}`}
            type="button"
            onClick={() => setView('zip')}
          >
            Zip
          </button>
        </nav>
      </header>

      <main className="page">
        <section className="panel">
          <div className="view-header">
            <h1>Tableau de bord</h1>
            <p>Un aperçu rapide de la progression et des actions disponibles.</p>
          </div>
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-label">Puzzles terminés</div>
              <div className="stat-value">12</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Temps moyen</div>
              <div className="stat-value">04:32</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Série actuelle</div>
              <div className="stat-value">5 jours</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Niveau</div>
              <div className="stat-value">Intermédiaire</div>
            </div>
          </div>
        </section>

        <div style={{ height: 20 }} />

        <section className="card-grid">
          <div className="card">
            <div className="card-title">Démarrer un puzzle</div>
            <div className="card-text">
              Lance immédiatement une nouvelle grille avec les dernières règles configurées.
            </div>
            <button className="cta" type="button">
              Continuer
            </button>
          </div>
          <div className="card">
            <div className="card-title">Statistiques avancées</div>
            <div className="card-text">
              Analyse tes performances et identifie les patterns de résolution.
            </div>
            <button className="secondary" type="button">
              Voir le détail
            </button>
          </div>
          <div className="card">
            <div className="card-title">Mode focus</div>
            <div className="card-text">Une interface épurée pour résoudre sans distraction.</div>
            <button className="secondary" type="button">
              Activer
            </button>
          </div>
        </section>

        <div style={{ height: 28 }} />

        {view === 'queens' ? (
          <section className="view">
            <div className="view-header">
              <h1>Queens</h1>
              <p>Place une reine par ligne, colonne et région, sans contact adjacent.</p>
            </div>
            <div className="layout">
              <section className="board" aria-label="Grille Queens">
                <div
                  className="grid"
                  style={
                    queensPuzzle
                      ? {
                          gridTemplateColumns: `repeat(${queensPuzzle.n}, 1fr)`,
                          gridTemplateRows: `repeat(${queensPuzzle.n}, 1fr)`,
                        }
                      : undefined
                  }
                  onPointerDown={handleQueensPointerDown}
                  onPointerMove={handleQueensPointerMove}
                  onPointerUp={handleQueensPointerUp}
                  onPointerCancel={handleQueensPointerUp}
                >
                  {queensGrid ?? <div className="placeholder">Chargement…</div>}
                </div>
              </section>
              <aside className="panel side-panel">
                <div className="panel-actions vertical">
                  <button className="secondary" type="button" onClick={handleQueensNext} disabled={!queensPuzzle}>
                    Puzzle suivant
                  </button>
                  <button
                    className="secondary"
                    type="button"
                    onClick={() => queensPuzzle && resetQueens(queensPuzzle)}
                    disabled={!queensPuzzle}
                  >
                    Réinitialiser
                  </button>
                  <button className="secondary" type="button" onClick={handleQueensHint} disabled={!queensPuzzle}>
                    Hint
                  </button>
                  <button className="primary" type="button" onClick={handleQueensVerify} disabled={!queensPuzzle}>
                    Vérifier
                  </button>
                </div>
                <div className={`panel-status ${queensStatusType ?? ''}`}>{queensStatus}</div>
                <div className="panel-note">
                  Clique sur une case pour placer une croix, puis reclique pour poser une reine.
                </div>
              </aside>
            </div>
          </section>
        ) : (
          <section className="view">
            <div className="view-header">
              <h1>Zip</h1>
              <p>Relie toutes les cases en respectant les murs et les numéros.</p>
            </div>
            <div className="layout">
              <section className="board" aria-label="Grille Zip">
                <div
                  className="zip-grid"
                  style={
                    zipPuzzle
                      ? {
                          gridTemplateColumns: `repeat(${zipPuzzle.n}, 1fr)`,
                          gridTemplateRows: `repeat(${zipPuzzle.n}, 1fr)`,
                        }
                      : undefined
                  }
                  onPointerDown={handleZipPointerDown}
                  onPointerMove={handleZipPointerMove}
                  onPointerUp={handleZipPointerUp}
                  onPointerCancel={handleZipPointerUp}
                >
                  {zipGrid ?? <div className="placeholder">Chargement…</div>}
                </div>
              </section>
              <aside className="panel side-panel">
                <div className="panel-actions vertical">
                  <button className="secondary" type="button" onClick={handleZipNext} disabled={!zipPuzzle}>
                    Puzzle suivant
                  </button>
                  <button className="secondary" type="button" onClick={handleZipReset} disabled={!zipPuzzle}>
                    Réinitialiser
                  </button>
                  <button className="secondary" type="button" onClick={handleZipHint} disabled={!zipPuzzle}>
                    Hint
                  </button>
                  <button className="primary" type="button" onClick={handleZipVerify} disabled={!zipPuzzle}>
                    Vérifier
                  </button>
                </div>
                <div className={`panel-status ${zipStatusType ?? ''}`}>{zipStatus}</div>
                <div className="panel-note">
                  Clique et glisse pour tracer un chemin qui passe par tous les numéros.
                </div>
              </aside>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
