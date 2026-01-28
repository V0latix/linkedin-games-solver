# Queens Data Science Report

- Source: `data/benchmarks/queens_runs.jsonl`
- Runs: 5585
- Unique puzzles: 1117
- Algorithms: dlx, baseline, csp_ac3, heuristic_lcv, min_conflicts

## Summary Table

| algo          |   puzzles |   solved |   solved_rate |   avg_time_ms |   median_time_ms |   avg_nodes |   avg_backtracks |   timeout_rate |
|:--------------|----------:|---------:|--------------:|--------------:|-----------------:|------------:|-----------------:|---------------:|
| dlx           |      1117 |     1117 |        100    |       1.49108 |         0.966375 |     41.2722 |         32.2193  |           0    |
| baseline      |      1117 |     1100 |         98.48 |      29.9378  |         1.57271  |  80182.4    |       7011.98    |           1.52 |
| csp_ac3       |      1117 |     1073 |         96.06 |      46.4859  |         3.07029  |    314.793  |        305.91    |           3.94 |
| heuristic_lcv |      1117 |     1067 |         95.52 |      55.1845  |         3.82912  |   1603.68   |        873.1     |           4.48 |
| min_conflicts |      1117 |      122 |         10.92 |     395.572   |       353.752    |  26667.7    |          5.32787 |          89.08 |

## Charts

![Average Time](figures/avg_time.png)
![Solve Rate](figures/solve_rate.png)
![Time Distribution](figures/time_box.png)
![Nodes vs Backtracks](figures/nodes_backtracks.png)
![Time by Size](figures/time_by_size.png)
![Solve Rate by Size](figures/solve_rate_by_size.png)