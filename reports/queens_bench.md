# Queens Benchmark Report

- Generated: 2026-01-27 18:33:54
- Dataset: `data/generated/queens`
- Algorithms: baseline, heuristic_simple, heuristic_lcv

## Summary

| algo | puzzles | solved | solve_rate | avg_time_ms | median_time_ms | avg_nodes | avg_backtracks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1 | 1 | 100.00% | 0.038 | 0.038 | 45.0 | 4.0 |
| heuristic_simple | 1 | 1 | 100.00% | 0.101 | 0.101 | 10.0 | 1.0 |
| heuristic_lcv | 1 | 1 | 100.00% | 0.200 | 0.200 | 9.0 | 1.0 |

## Charts

### Average Time (ms)

```text
baseline           | ######                           |      0.038
heuristic_simple   | ################                 |      0.101
heuristic_lcv      | ################################ |      0.200
```

### Average Nodes

```text
baseline           | ################################ |     45.000
heuristic_simple   | #######                          |     10.000
heuristic_lcv      | ######                           |      9.000
```

## Notes

- Times are measured inside each solver using `perf_counter()`.
- Averages and medians are computed over solved puzzles only.
- This report is ASCII-only to stay portable in terminals and GitHub Markdown.