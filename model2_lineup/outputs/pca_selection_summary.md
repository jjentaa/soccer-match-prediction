# Model 2 PCA and feature selection summary (training split)

## Components needed for explained variance

- 80%: 52 components
- 90%: 96 components
- 95%: 135 components
- PC1-PC4 explain 20.7%, 15.3%, 3.4%, 2.5%

## Largest loadings

- PC1: home_ovr_mean, away_ovr_mean, home_pot_mean, away_pot_mean, home_MID_overall_rating
- PC2: diff_ovr_mean, diff_pot_mean, diff_MID_overall_rating, diff_DEF_overall_rating, diff_ovr_max
- PC3: diff_MID_standing_tackle, diff_MID_marking, diff_MID_sliding_tackle, diff_FWD_n, diff_MID_interceptions
- PC4: diff_DEF_long_shots, diff_DEF_crossing, diff_DEF_curve, diff_DEF_dribbling, diff_DEF_free_kick_accuracy

## CV score by reducer and number of inputs

```
             method   k  accuracy  macro_f1
                PCA   5    0.5081    0.3786
                PCA  10    0.5112    0.3814
                PCA  20    0.5133    0.3861
                PCA  40    0.5120    0.3852
                PCA  80    0.5136    0.3864
                PCA 160    0.5162    0.3897
                PCA 351    0.5159    0.3883
LightGBM importance   5    0.5175    0.3897
LightGBM importance  10    0.5178    0.3902
LightGBM importance  20    0.5149    0.3914
LightGBM importance  40    0.5134    0.3930
LightGBM importance  80    0.5158    0.3953
LightGBM importance 160    0.5137    0.3904
LightGBM importance 351    0.5166    0.3935
```

## Best k per reducer (by macro F1)

```
                       k  accuracy  macro_f1
method                                      
LightGBM importance   80    0.5158    0.3953
PCA                  160    0.5162    0.3897
```