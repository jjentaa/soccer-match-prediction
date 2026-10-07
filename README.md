# Soccer match prediction — Complex Data Analytics, Group 2

This README describes how the raw data was cleaned and joined into the tables in [`datasets/`](datasets/README.md).
Column descriptions and join instructions are in [`datasets/README.md`](datasets/README.md).

## Running it

```bash
pip install -r requirements.txt
```

```bash
python preprocessing/download_data.py   # step 1: raw data -> preprocessing/raw/   (~40 MB download, ~300 MB on disk)
python preprocessing/preprocess.py      # step 2: raw -> datasets/*.parquet        (~15 s)
```

`preprocessing/raw/` is git-ignored. You only need to run the two scripts if you want to change the cleaning;
the outputs are already committed in `datasets/`.

## Raw sources

| # | Source | Format | Content |
|---|---|---|---|
| 1 | [European Soccer Database](https://www.kaggle.com/datasets/hugomathien/soccer) (Kaggle) | 1 SQLite file, 7 tables | 25,979 matches, lineups, odds, FIFA player and team attributes |
| 2 | [football-data.co.uk](https://www.football-data.co.uk/) | 72 CSV files (9 leagues × 8 seasons) | per-match shots, corners, fouls, cards, half-time score |

Poland and Switzerland are not published in source 2's per-season format, so they have no match stats.

## Pipeline (`preprocessing/preprocess.py`)

```
SQLite ─ Player ────────────► clean_players ─► players
       ─ Player_Attributes ─► clean_players ─► player_attributes ─┐
       ─ Team ──────────────► clean_teams ───► teams              │
       ─ Team_Attributes ───► clean_teams ───► team_attributes    │
       ─ Match, League ─────► clean_matches ─┐                    │
72 CSV ─────► load_football_data ─► map_fd_teams ─► join_football_data ─► matches
                                                                  │         ├─► build_team_matches ─► team_matches
                                                                  └─────────┴─► build_lineups ──────► lineups
```

### 1. Cleaning

| Table | Step | Why |
|---|---|---|
| Player | parse `birthday`; `height` → `height_cm`; `weight` (lb) → `weight_kg` | consistent types and units |
| Player_Attributes | parse `date` | |
| | drop 836 rows with no `overall_rating` | these rows have every attribute empty |
| | drop duplicate (`player_api_id`, `date`) snapshots | required for the as-of join |
| | work-rate values outside `low` / `medium` / `high` → NaN | raw column contains junk tokens (`norm`, `o`, `ormal`, numbers…) |
| Team_Attributes | parse `date`; drop 1 duplicate (`team_api_id`, `date`) | |
| Match | parse `date`; attach league and team names | |
| | add `result` (H / D / A) from the goals | prediction target |
| | drop the 8 XML event columns (`goal`, `shoton`, `card`, …) | empty for 45 % of matches; replaced by source 2 |
| | keep odds of the 6 bookmakers with good coverage, drop the other 4 | the dropped ones are missing for 34–57 % of matches |
| football-data CSVs | read as latin-1, drop empty trailing rows, strip team names, parse `dd/mm/yy` and `dd/mm/yyyy` dates | file formats vary by season |

Surrogate `id` columns are dropped everywhere; the `*_api_id` columns are the keys.

### 2. Joining the two sources

The sources share no ids and spell team names differently (`Man United` vs `Manchester United`).

1. **Learn the team-name mapping from the data** (`map_fd_teams`). Rows from both sources that share
   (league, date, home goals, away goals) are candidate pairs; each football-data name is assigned the ESD team
   it co-occurs with most often. Mappings supported by fewer than 5 matches fail the run. Several spellings may
   map to one team (`Roda`, `Roda JC`).
2. **Join the matches** (`join_football_data`) on (league, season, home team, away team). Where a fixture is played
   more than once per season (e.g. Scotland), the closest date within 3 days wins, and each football-data row
   is used at most once.

Result: 22,625 of the 22,637 matches in the 9 covered leagues are matched; 1 matched pair disagrees on the score
(the ESD score is kept). `has_fd_stats` and `has_shot_stats` record the outcome per match.

### 3. Derived tables

- **`team_matches`** (`build_team_matches`) — each match duplicated into a home-team row and an away-team row with
  stats renamed to `*_for` / `*_against`, plus `points`. Sorted by team and date for rolling windows.
- **`lineups`** (`build_lineups`) — the 22 lineup slots unpivoted to one row per player, then joined to
  `player_attributes` with `merge_asof(direction="backward")`: each player gets the most recent snapshot dated on or
  before the match. This prevents future ratings leaking into past matches. Every lineup row found a snapshot.
- **`split`** (`build_split`) — random 70:30 train/test assignment per match, seed 42, shared by all models.

## Deliberately not done here

These depend on the train/test split or on the individual model, so they belong in the modelling code:

- **Imputation** — must be fitted on the training split only. Remaining missing values are listed in `datasets/README.md`.
- **Rolling / window features** (team form).
- **Class-imbalance handling** and **dimension reduction** (PCA).
