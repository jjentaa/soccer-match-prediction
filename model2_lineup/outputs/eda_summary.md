# Model 2 EDA summary (training split)

## Class balance (train, %)

```
result
H    45.7
A    28.8
D    25.5
```

## Result rate (%) by rating difference

```
result     H     D     A
bin                     
< -8    13.4  22.4  64.2
-8..-6  21.9  27.8  50.3
-6..-4  29.9  26.2  43.9
-4..-2  37.4  27.8  34.7
-2..0   41.9  29.9  28.2
0..2    47.7  27.4  25.0
2..4    53.8  26.0  20.2
4..6    62.3  22.8  14.9
6..8    70.0  17.8  12.2
> 8     78.8  14.5   6.8
```

## Overall rating by line

```
         count  mean  std   min   25%   50%   75%   max
line                                                   
GK     34664.0  72.4  7.3  43.0  67.0  73.0  78.0  91.0
DEF   136519.0  70.2  6.7  35.0  66.0  71.0  75.0  90.0
MID   143397.0  71.2  6.9  40.0  66.0  72.0  76.0  93.0
FWD    65013.0  72.2  7.1  41.0  67.0  72.0  77.0  94.0
```

## Attribute correlation

- mean_abs_corr: 0.384
- pairs_abs_corr_over_0.7: 85
- n_pairs: 1369

## Missing values

- matches: 25979
- matches_without_lineup: 758
- matches_with_full_22: 21374
- matches_partial_lineup: 3847
- features: 351
- train_features_with_any_missing: 351
- train_worst_missing: {'diff_FWD_defensive_work_rate': 0.081, 'diff_FWD_attacking_work_rate': 0.08, 'diff_FWD_sliding_tackle': 0.061, 'diff_FWD_curve': 0.061, 'diff_FWD_vision': 0.061}
