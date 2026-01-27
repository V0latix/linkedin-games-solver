# Queens Benchmark Report (Recursive)

- Generated: 2026-01-27 19:39:27
- Dataset root: `data/generated/queens`
- Algorithms: baseline, heuristic_lcv, heuristic_simple

## Global Summary

| algo | puzzles | solved | solve_rate | avg_time_ms | median_time_ms | avg_nodes | avg_backtracks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 450 | 450 | 100.00% | 1.834 | 0.062 | 6731.3 | 582.7 |
| heuristic_lcv | 450 | 450 | 100.00% | 2.277 | 0.915 | 67.5 | 33.0 |
| heuristic_simple | 450 | 450 | 100.00% | 6.997 | 0.420 | 396.7 | 237.3 |

## Size 10

| algo | puzzles | solved | solve_rate | avg_time_ms | median_time_ms | avg_nodes | avg_backtracks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 70 | 70 | 100.00% | 1.351 | 0.168 | 4189.0 | 413.4 |
| heuristic_lcv | 70 | 70 | 100.00% | 1.787 | 1.411 | 35.3 | 14.6 |
| heuristic_simple | 70 | 70 | 100.00% | 5.217 | 0.992 | 360.7 | 207.6 |

### Average Time (ms)

```text
baseline           | ########                         |      1.351
heuristic_lcv      | ##########                       |      1.787
heuristic_simple   | ################################ |      5.217
```

### Average Nodes

```text
baseline           | ################################ |   4189.000
heuristic_lcv      | #                                |     35.343
heuristic_simple   | ##                               |    360.714
```

### Slowest Puzzles (by time)

```text
baseline:
  size_10/queens_n10_018_seed198 |     52.933 ms | nodes=141795
  size_10/queens_n10_035_seed435 |      7.830 ms | nodes=29065
  size_10/queens_n10_010_seed410 |      4.370 ms | nodes=16275
heuristic_simple:
  size_10/queens_n10_018_seed198 |    180.407 ms | nodes=13931
  size_10/queens_n10_005_seed405 |     21.125 ms | nodes=1407
  size_10/queens_n10_010_seed410 |     19.554 ms | nodes=1291
heuristic_lcv:
  size_10/queens_n10_020_seed420 |      5.572 ms | nodes=185
  size_10/queens_n10_034_seed434 |      4.764 ms | nodes=194
  size_10/queens_n10_044_seed444 |      3.958 ms | nodes=119
```

## Size 11

| algo | puzzles | solved | solve_rate | avg_time_ms | median_time_ms | avg_nodes | avg_backtracks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 50 | 50 | 100.00% | 3.196 | 0.724 | 11951.3 | 1080.5 |
| heuristic_lcv | 50 | 50 | 100.00% | 4.696 | 2.065 | 141.6 | 74.7 |
| heuristic_simple | 50 | 50 | 100.00% | 12.023 | 3.462 | 675.1 | 412.5 |

### Average Time (ms)

```text
baseline           | ########                         |      3.196
heuristic_lcv      | ############                     |      4.696
heuristic_simple   | ################################ |     12.023
```

### Average Nodes

```text
baseline           | ################################ |  11951.280
heuristic_lcv      | #                                |    141.600
heuristic_simple   | #                                |    675.060
```

### Slowest Puzzles (by time)

```text
baseline:
  size_11/queens_n11_038_seed488 |     54.665 ms | nodes=204820
  size_11/queens_n11_037_seed487 |     38.510 ms | nodes=143781
  size_11/queens_n11_020_seed470 |     13.945 ms | nodes=50963
heuristic_simple:
  size_11/queens_n11_038_seed488 |    292.845 ms | nodes=16517
  size_11/queens_n11_020_seed470 |     72.627 ms | nodes=4407
  size_11/queens_n11_039_seed489 |     51.966 ms | nodes=2988
heuristic_lcv:
  size_11/queens_n11_041_seed491 |    108.422 ms | nodes=4958
  size_11/queens_n11_011_seed461 |      8.455 ms | nodes=328
  size_11/queens_n11_044_seed494 |      8.075 ms | nodes=253
```

## Size 12

| algo | puzzles | solved | solve_rate | avg_time_ms | median_time_ms | avg_nodes | avg_backtracks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| heuristic_lcv | 50 | 50 | 100.00% | 9.783 | 2.961 | 317.5 | 170.8 |
| baseline | 50 | 50 | 100.00% | 10.971 | 1.296 | 41373.8 | 3441.3 |
| heuristic_simple | 50 | 50 | 100.00% | 41.135 | 5.430 | 2215.5 | 1354.1 |

### Average Time (ms)

```text
heuristic_lcv      | #######                          |      9.783
baseline           | ########                         |     10.971
heuristic_simple   | ################################ |     41.135
```

### Average Nodes

```text
heuristic_lcv      | #                                |    317.460
baseline           | ################################ |  41373.840
heuristic_simple   | #                                |   2215.540
```

### Slowest Puzzles (by time)

```text
baseline:
  size_12/queens_n12_047_seed547 |    180.183 ms | nodes=672570
  size_12/queens_n12_038_seed538 |     68.614 ms | nodes=255882
  size_12/queens_n12_014_seed514 |     49.450 ms | nodes=187314
heuristic_simple:
  size_12/queens_n12_047_seed547 |    500.745 ms | nodes=29147
  size_12/queens_n12_014_seed514 |    227.701 ms | nodes=14886
  size_12/queens_n12_037_seed537 |    212.965 ms | nodes=9566
heuristic_lcv:
  size_12/queens_n12_030_seed530 |    144.834 ms | nodes=6006
  size_12/queens_n12_014_seed514 |    136.791 ms | nodes=6115
  size_12/queens_n12_003_seed503 |     17.529 ms | nodes=614
```

## Size 6

| algo | puzzles | solved | solve_rate | avg_time_ms | median_time_ms | avg_nodes | avg_backtracks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 70 | 70 | 100.00% | 0.023 | 0.020 | 53.1 | 5.3 |
| heuristic_simple | 70 | 70 | 100.00% | 0.117 | 0.100 | 11.2 | 2.9 |
| heuristic_lcv | 70 | 70 | 100.00% | 0.262 | 0.227 | 12.0 | 3.2 |

### Average Time (ms)

```text
baseline           | ##                               |      0.023
heuristic_simple   | ##############                   |      0.117
heuristic_lcv      | ################################ |      0.262
```

### Average Nodes

```text
baseline           | ################################ |     53.057
heuristic_simple   | ######                           |     11.214
heuristic_lcv      | #######                          |     11.957
```

### Slowest Puzzles (by time)

```text
baseline:
  size_6/queens_n6_022_seed222 |      0.077 ms | nodes=207
  size_6/queens_n6_009_seed109 |      0.071 ms | nodes=189
  size_6/queens_n6_005_seed105 |      0.062 ms | nodes=189
heuristic_simple:
  size_6/queens_n6_009_seed109 |      0.327 ms | nodes=33
  size_6/queens_n6_005_seed105 |      0.316 ms | nodes=34
  size_6/queens_n6_022_seed222 |      0.304 ms | nodes=31
heuristic_lcv:
  size_6/queens_n6_011_seed211 |      1.329 ms | nodes=9
  size_6/queens_n6_022_seed222 |      0.744 ms | nodes=51
  size_6/queens_n6_007_seed207 |      0.576 ms | nodes=40
```

## Size 7

| algo | puzzles | solved | solve_rate | avg_time_ms | median_time_ms | avg_nodes | avg_backtracks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 70 | 70 | 100.00% | 0.041 | 0.028 | 111.2 | 11.9 |
| heuristic_simple | 70 | 70 | 100.00% | 0.228 | 0.163 | 18.9 | 6.8 |
| heuristic_lcv | 70 | 70 | 100.00% | 0.444 | 0.388 | 15.9 | 4.8 |

### Average Time (ms)

```text
baseline           | ##                               |      0.041
heuristic_simple   | ################                 |      0.228
heuristic_lcv      | ################################ |      0.444
```

### Average Nodes

```text
baseline           | ################################ |    111.200
heuristic_simple   | #####                            |     18.943
heuristic_lcv      | ####                             |     15.900
```

### Slowest Puzzles (by time)

```text
baseline:
  size_7/queens_n7_017_seed137 |      0.263 ms | nodes=847
  size_7/queens_n7_005_seed255 |      0.212 ms | nodes=665
  size_7/queens_n7_014_seed264 |      0.192 ms | nodes=616
heuristic_simple:
  size_7/queens_n7_017_seed137 |      1.199 ms | nodes=124
  size_7/queens_n7_005_seed255 |      0.877 ms | nodes=91
  size_7/queens_n7_014_seed264 |      0.874 ms | nodes=88
heuristic_lcv:
  size_7/queens_n7_026_seed276 |      1.183 ms | nodes=66
  size_7/queens_n7_018_seed138 |      1.053 ms | nodes=10
  size_7/queens_n7_017_seed137 |      0.840 ms | nodes=48
```

## Size 8

| algo | puzzles | solved | solve_rate | avg_time_ms | median_time_ms | avg_nodes | avg_backtracks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 70 | 70 | 100.00% | 0.061 | 0.039 | 182.4 | 18.3 |
| heuristic_simple | 70 | 70 | 100.00% | 0.366 | 0.249 | 24.6 | 9.6 |
| heuristic_lcv | 70 | 70 | 100.00% | 0.640 | 0.589 | 15.4 | 4.0 |

### Average Time (ms)

```text
baseline           | ###                              |      0.061
heuristic_simple   | ##################               |      0.366
heuristic_lcv      | ################################ |      0.640
```

### Average Nodes

```text
baseline           | ################################ |    182.400
heuristic_simple   | ####                             |     24.614
heuristic_lcv      | ##                               |     15.443
```

### Slowest Puzzles (by time)

```text
baseline:
  size_8/queens_n8_014_seed314 |      0.299 ms | nodes=1028
  size_8/queens_n8_046_seed346 |      0.214 ms | nodes=652
  size_8/queens_n8_019_seed159 |      0.193 ms | nodes=564
heuristic_simple:
  size_8/queens_n8_014_seed314 |      1.498 ms | nodes=127
  size_8/queens_n8_024_seed324 |      1.482 ms | nodes=49
  size_8/queens_n8_004_seed304 |      0.950 ms | nodes=59
heuristic_lcv:
  size_8/queens_n8_045_seed345 |      1.234 ms | nodes=56
  size_8/queens_n8_006_seed146 |      1.231 ms | nodes=59
  size_8/queens_n8_046_seed346 |      1.034 ms | nodes=43
```

## Size 9

| algo | puzzles | solved | solve_rate | avg_time_ms | median_time_ms | avg_nodes | avg_backtracks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 70 | 70 | 100.00% | 0.193 | 0.081 | 647.9 | 67.0 |
| heuristic_simple | 70 | 70 | 100.00% | 1.082 | 0.565 | 70.0 | 36.4 |
| heuristic_lcv | 70 | 70 | 100.00% | 1.160 | 0.947 | 27.4 | 10.2 |

### Average Time (ms)

```text
baseline           | #####                            |      0.193
heuristic_simple   | #############################    |      1.082
heuristic_lcv      | ################################ |      1.160
```

### Average Nodes

```text
baseline           | ################################ |    647.871
heuristic_simple   | ###                              |     70.043
heuristic_lcv      | #                                |     27.400
```

### Slowest Puzzles (by time)

```text
baseline:
  size_9/queens_n9_019_seed179 |      1.495 ms | nodes=5274
  size_9/queens_n9_007_seed357 |      0.908 ms | nodes=3168
  size_9/queens_n9_037_seed387 |      0.828 ms | nodes=2817
heuristic_simple:
  size_9/queens_n9_019_seed179 |      8.733 ms | nodes=569
  size_9/queens_n9_037_seed387 |      4.559 ms | nodes=311
  size_9/queens_n9_007_seed357 |      4.370 ms | nodes=331
heuristic_lcv:
  size_9/queens_n9_044_seed394 |      7.924 ms | nodes=431
  size_9/queens_n9_038_seed388 |      2.672 ms | nodes=109
  size_9/queens_n9_048_seed398 |      2.422 ms | nodes=106
```

## Notes

- Times are measured inside each solver using `perf_counter()`.
- Averages and medians are computed over solved puzzles only.
- This report is ASCII-only to stay portable in terminals and GitHub Markdown.