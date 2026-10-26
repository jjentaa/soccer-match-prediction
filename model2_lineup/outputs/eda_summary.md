# Model 2 EDA summary (training split)

## Class balance (train, %)

```
result
H    46.0
A    28.8
D    25.2
```

## Result rate (%) by rating difference

```
result     H     D     A
bin                     
< -8    14.3  21.9  63.8
-8..-6  22.6  26.1  51.2
-6..-4  30.5  26.5  43.0
-4..-2  37.5  27.6  34.9
-2..0   42.4  30.3  27.2
0..2    46.2  27.7  26.1
2..4    54.7  25.0  20.3
4..6    62.1  23.2  14.7
6..8    70.2  18.5  11.3
> 8     79.8  13.2   7.1
```

## Overall rating by line

```
         count  mean  std   min   25%   50%   75%   max
line                                                   
GK     34625.0  72.4  7.3  43.0  67.0  73.0  78.0  91.0
DEF   136450.0  70.2  6.7  35.0  66.0  71.0  75.0  90.0
MID   143268.0  71.2  6.9  40.0  66.0  72.0  76.0  93.0
FWD    65074.0  72.1  7.1  40.0  67.0  72.0  77.0  94.0
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
- train_worst_missing: {'diff_FWD_defensive_work_rate': 0.083, 'diff_FWD_attacking_work_rate': 0.081, 'diff_GK_overall_rating': 0.063, 'diff_GK_gk_diving': 0.063, 'diff_GK_gk_reflexes': 0.063}
