# Model 2 PCA and feature selection summary (training split)

## Components needed for explained variance

- 80%: 52 components
- 90%: 96 components
- 95%: 135 components
- PC1-PC4 explain 20.7%, 15.2%, 3.4%, 2.5%

## Largest loadings

- PC1: away_ovr_mean, home_ovr_mean, away_pot_mean, home_pot_mean, away_MID_overall_rating
- PC2: diff_ovr_mean, diff_pot_mean, diff_MID_overall_rating, diff_DEF_overall_rating, diff_ovr_max
- PC3: diff_MID_standing_tackle, diff_MID_marking, diff_MID_sliding_tackle, diff_FWD_n, diff_MID_interceptions
- PC4: diff_DEF_long_shots, diff_DEF_crossing, diff_DEF_curve, diff_DEF_free_kick_accuracy, diff_DEF_finishing

## CV score by reducer and number of inputs

```
             method   k  accuracy  macro_f1
                PCA   5    0.5094    0.3823
                PCA  10    0.5134    0.3827
                PCA  20    0.5127    0.3872
                PCA  40    0.5129    0.3880
                PCA  80    0.5144    0.3910
                PCA 160    0.5125    0.3873
                PCA 351    0.5130    0.3876
LightGBM importance   5    0.5135    0.3884
LightGBM importance  10    0.5128    0.3878
LightGBM importance  20    0.5144    0.3948
LightGBM importance  40    0.5135    0.3967
LightGBM importance  80    0.5139    0.3966
LightGBM importance 160    0.5151    0.3957
LightGBM importance 351    0.5148    0.3943
```

## Best k per reducer (by macro F1)

```
                      k  accuracy  macro_f1
method                                     
LightGBM importance  40    0.5135    0.3967
PCA                  80    0.5144    0.3910
```