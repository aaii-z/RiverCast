# Phase 3 results (test: Jul-Dec 2024)

## MAE (log10) by node group

| nodes       |   persistence |   xgboost |   lstm |   graph_wavenet |
|:------------|--------------:|----------:|-------:|----------------:|
| all         |         0.541 |     0.783 |  0.377 |           0.332 |
| monitored   |         0.444 |     0.341 |  0.328 |           0.326 |
| unmonitored |         0.778 |     1.864 |  0.499 |           0.346 |

## Multi-task vs single-task Graph WaveNet (MAE log10, all nodes)

_single-task runs missing_

## 80% interval coverage vs truth

| model         |   raw |   calibrated |
|:--------------|------:|-------------:|
| xgboost       | 0.535 |        0.532 |
| lstm          | 0.635 |        0.636 |
| graph_wavenet | 0.674 |        0.675 |

## 80% interval coverage vs sensor observations

| model         |   raw |   calibrated |
|:--------------|------:|-------------:|
| xgboost       | 0.839 |        0.807 |
| lstm          | 0.826 |        0.807 |
| graph_wavenet | 0.815 |        0.823 |

## Mean interval width (log10)

| model         |   raw |   calibrated |
|:--------------|------:|-------------:|
| xgboost       | 0.825 |        0.813 |
| lstm          | 0.654 |        0.657 |
| graph_wavenet | 0.652 |        0.656 |

## Raw 80% coverage per target

Norovirus is mostly below the detection limit, so sensors report LOD/2 and models learn that value: well covered against observations, biased high against the (lower) truth.

|                                |   observations |   truth |
|:-------------------------------|---------------:|--------:|
| ('graph_wavenet', 'amr')       |          0.795 |   0.866 |
| ('graph_wavenet', 'fio')       |          0.804 |   0.859 |
| ('graph_wavenet', 'norovirus') |          0.847 |   0.295 |
| ('lstm', 'amr')                |          0.782 |   0.826 |
| ('lstm', 'fio')                |          0.784 |   0.807 |
| ('lstm', 'norovirus')          |          0.911 |   0.272 |
| ('xgboost', 'amr')             |          0.802 |   0.758 |
| ('xgboost', 'fio')             |          0.799 |   0.658 |
| ('xgboost', 'norovirus')       |          0.916 |   0.189 |

## MAE (log10) in calm vs high-pollution hours (truth above node's 95th percentile)

| model         |   calm |   high_pollution |
|:--------------|-------:|-----------------:|
| persistence   |  0.516 |            1.154 |
| xgboost       |  0.785 |            0.742 |
| lstm          |  0.361 |            0.758 |
| graph_wavenet |  0.317 |            0.691 |

## FIO > 1000 CFU/100 mL alerts, 6 h ahead

| model         | rule              |   precision |   recall |    f1 |   events |
|:--------------|:------------------|------------:|---------:|------:|---------:|
| persistence   | median > 1000     |       0.673 |    0.509 | 0.58  |    33606 |
| xgboost       | median > 1000     |       0.428 |    0.763 | 0.549 |    33606 |
| xgboost       | upper band > 1000 |       0.353 |    0.934 | 0.512 |    33606 |
| lstm          | median > 1000     |       0.789 |    0.584 | 0.671 |    33606 |
| lstm          | upper band > 1000 |       0.367 |    0.905 | 0.522 |    33606 |
| graph_wavenet | median > 1000     |       0.86  |    0.624 | 0.724 |    33606 |
| graph_wavenet | upper band > 1000 |       0.439 |    0.888 | 0.588 |    33606 |
