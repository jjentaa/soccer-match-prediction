# Model 3 — team play style (graph)

**Question:** does a team that plays style X tend to beat a team that plays style Y?

```bash
python model3/build_features.py   # ~40 s -> features.parquet, styles.parquet
python model3/train.py            # ~20 s -> predictions/, metrics.json, figures/
```

## Method

1. **Style.** Each team's 8 FIFA tactic settings (build-up speed/passing, chance-creation passing/crossing/shooting,
   defence pressure/aggression/width), taken as of match day, are standardised and clustered with K-means.
   K is chosen by silhouette score (K = 4).
2. **Graphs.**

   | Graph | Nodes | Edges | Features |
   |---|---|---|---|
   | Style matchup | the 4 styles | home style → away style, weighted by past H / D / A rates | `matchup_p_*`, `style_rank_*` (PageRank of a style) |
   | Style similarity | teams of a league | 3 nearest neighbours in tactic space | degree centrality, clustering coefficient, density, `*_ppg_vs_similar_to_*` |
   | Results | teams of a league | loser → winner, last 365 days | `pagerank_*`, `weakrank_*`, `results_density` |

   `home_ppg_vs_similar_to_away` is the feature closest to the question: the points per game the home team took, over
   the last two years, from the away team and its style neighbours.
3. **Model.** Gradient boosting (`HistGradientBoostingClassifier`) on the graph features plus the raw tactics and
   formation line counts. Missing values are left to the trees.

**Leakage control.** The split is random, so test matches are scattered through time. Every graph that uses match
outcomes is built only from *training* matches dated strictly before the match being described.

## Results (30 % test split, 7,794 matches)

| | Accuracy | Macro F1 | Log loss |
|---|---|---|---|
| Baseline: always home win | 0.456 | 0.209 | 1.066 |
| Style only, no graph | 0.470 | 0.289 | 1.044 |
| + style graphs | 0.487 | 0.324 | 1.030 |
| **+ results graph (final)** | **0.504** | **0.361** | **1.010** |
| Final with `class_weight="balanced"` | 0.443 | 0.429 | 1.040 |
| Reference: Bet365 odds (6,800 matches with odds) | 0.532 | 0.389 | 0.974 |

- Each graph adds signal on top of raw style.
- Without class weighting the model almost never predicts a draw; weighting trades 6 points of accuracy for 7 points of macro F1.

## Findings and limits

![style profiles](figures/style_profiles.png)
![style matchups](figures/style_matchups.png)

- **S0 (high press, fast and risky attacking play) is the dominant style**: at home it takes 1.87–1.93 points per game
  from the other three styles, and every other style scores below average against it.
- **Style is confounded with team strength.** Strong clubs are the ones that can afford to press high, so part of the
  S0 effect is squad quality, not tactics. The results-graph PageRank is there to separate the two.
- **The styles are not well-separated clusters** (silhouette 0.13): tactics vary continuously, so the 4 styles are a
  convenient discretisation rather than natural groups.
- **25 % of matches have no style** (tactics start in February 2010); for those the model falls back on the results graph.

## Outputs for the ensemble

| File | Rows | Content |
|---|---|---|
| `predictions/model3_test.parquet` | 7,794 | `match_api_id`, `p_H`, `p_D`, `p_A` for every test match |
| `predictions/model3_train_oof.parquet` | 18,185 | 5-fold out-of-fold probabilities for every training match, for training a stacking model |

Probabilities come from the unweighted model so they stay calibrated.
