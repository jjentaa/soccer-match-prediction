# Datasets

Seven cleaned, already-joined tables plus the shared train/test split. All were produced by [`../preprocessing/`](../README.md);
do not edit them by hand — change the script and re-run it.

```python
import pandas as pd
matches = pd.read_parquet("datasets/matches.parquet")
```

Coverage: 11 European leagues, seasons 2008/2009 – 2015/2016, 25,979 matches.

| File | Rows × cols | One row is… | Primary key |
|---|---|---|---|
| [`matches`](#matchesparquet) | 25,979 × 112 | a match | `match_api_id` |
| [`team_matches`](#team_matchesparquet) | 51,958 × 27 | a team in a match | `match_api_id`, `team_api_id` |
| [`lineups`](#lineupsparquet) | 542,281 × 47 | a starting player in a match | `match_api_id`, `side`, `slot` |
| [`player_attributes`](#player_attributesparquet) | 183,142 × 40 | a FIFA rating snapshot of a player | `player_api_id`, `date` |
| [`team_attributes`](#team_attributesparquet) | 1,457 × 23 | a FIFA tactics snapshot of a team | `team_api_id`, `date` |
| [`players`](#playersparquet) | 11,060 × 6 | a player | `player_api_id` |
| [`teams`](#teamsparquet) | 299 × 4 | a team | `team_api_id` |
| [`split`](#splitparquet) | 25,979 × 2 | a match's train/test assignment | `match_api_id` |

## How the tables join

```
teams ──team_api_id──┐
                     ├── matches ──match_api_id──┬── team_matches   (2 rows per match)
team_attributes ─────┘   (home_/away_team_api_id) └── lineups        (≤ 22 rows per match)
  (as-of on date)                                      │
                                                 player_api_id
                                                       │
                                          players, player_attributes
```

| From → to | Key | Type |
|---|---|---|
| `matches` → `team_matches` | `match_api_id` | 1 : 2 |
| `matches` → `lineups` | `match_api_id` | 1 : up to 22 |
| `matches` → `teams` | `home_team_api_id` / `away_team_api_id` = `team_api_id` | many : 1 |
| `lineups` → `players` | `player_api_id` | many : 1 |
| `matches` / `team_matches` → `team_attributes` | `team_api_id` + **as-of** `date` | many : 1 |
| `lineups` → `player_attributes` | already joined (as-of) — no need to join again | — |

**Snapshot tables must be joined as-of, never on equality.** `team_attributes` and `player_attributes`
are dated snapshots; a match must only see the latest snapshot on or before its own date:

```python
team_attributes = pd.read_parquet("datasets/team_attributes.parquet")
team_matches = pd.read_parquet("datasets/team_matches.parquet")

tm = pd.merge_asof(
    team_matches.sort_values("date"),
    team_attributes.sort_values("date").rename(columns={"date": "attr_date"}),
    left_on="date", right_on="attr_date",
    by="team_api_id", direction="backward",
)
```

Bringing team-level features back to one row per match:

```python
feats = ...  # one row per (match_api_id, team_api_id), built from team_matches
home = feats[feats.is_home == 1].drop(columns="is_home").add_prefix("home_")
away = feats[feats.is_home == 0].drop(columns="is_home").add_prefix("away_")
X = (matches[["match_api_id", "date", "season", "result"]]
     .merge(home, left_on="match_api_id", right_on="home_match_api_id")
     .merge(away, left_on="match_api_id", right_on="away_match_api_id"))
```

## Rules for using the data

- **Target** is `matches.result` (`H` 45.9 %, `A` 28.7 %, `D` 25.4 %).
- **Post-match columns leak.** Goals, shots, corners, fouls, cards, points and `result` are only known after the
  final whistle. As features they must come from *earlier* matches only, e.g.
  `team_matches.groupby("team_api_id")["points"].transform(lambda s: s.shift(1).rolling(5).mean())`.
- **Pre-match columns are safe as-is:** lineups, pitch positions, betting odds, and the attribute columns in `lineups`.
- **Use the shared split** in `split.parquet` (random 70:30, seed 42) — do not re-split your own table, or the
  models' predictions will not line up for the ensemble.
- **Because the split is random, test matches are scattered through time.** Any feature built from match outcomes
  must use only matches dated *before* the row's match, and only *training* matches, so that no test label
  ever reaches a feature.
- **Missing values are left as NaN** so that imputation can be fitted on the training split only. Percentages are listed per column below.

---

## `matches.parquet`

| Column | Type | Description | Missing |
|---|---|---|---|
| `match_api_id` | int | Match id (primary key) | – |
| `league` | str | League name, 11 values | – |
| `season` | str | `"2008/2009"` … `"2015/2016"` | – |
| `stage` | int | Match-day number within the season | – |
| `date` | datetime | Match date | – |
| `home_team_api_id`, `away_team_api_id` | int | Team ids → `teams.team_api_id` | – |
| `home_team`, `away_team` | str | Team long names | – |
| `home_team_goal`, `away_team_goal` | int | Full-time goals *(post-match)* | – |
| `result` | str | **Target.** `H` home win, `D` draw, `A` away win *(post-match)* | – |
| `home_player_1` … `home_player_11`, `away_player_1` … `away_player_11` | float | `player_api_id` of each starter. Slot 1 is the goalkeeper | 4.7–6 % per slot; 17.7 % of matches miss at least one |
| `home_player_X1` … `X11`, `away_player_X1` … `X11` | float | Left–right pitch position of the slot, 1–9 | ~7 % |
| `home_player_Y1` … `Y11`, `away_player_Y1` … `Y11` | float | Back–front pitch position, 1 = goalkeeper … 11 = striker | ~7 % |
| `B365H/D/A`, `BWH/D/A`, `IWH/D/A`, `LBH/D/A`, `WHH/D/A`, `VCH/D/A` | float | Pre-match decimal odds for home / draw / away from Bet365, Bet&Win, Interwetten, Ladbrokes, William Hill, VC Bet | ~13 % |
| `h_ht_goals`, `a_ht_goals` | float | Half-time goals *(post-match)* | 12.9 % |
| `h_shots`, `a_shots` | float | Total shots *(post-match)* | 36.8 % |
| `h_shots_on_target`, `a_shots_on_target` | float | Shots on target *(post-match)* | 36.8 % |
| `h_fouls`, `a_fouls` | float | Fouls committed *(post-match)* | 36.8 % |
| `h_corners`, `a_corners` | float | Corners *(post-match)* | 36.8 % |
| `h_yellows`, `a_yellows` | float | Yellow cards *(post-match)* | 36.8 % |
| `h_reds`, `a_reds` | float | Red cards *(post-match)* | 36.8 % |
| `has_fd_stats` | bool | Match was found in football-data.co.uk (22,625 matches) | – |
| `has_shot_stats` | bool | Shots/fouls/corners/cards are present (16,408 matches) | – |

`h_` = home team, `a_` = away team. Stat coverage by league:

| Leagues | `h_ht_goals` | shots, fouls, corners, cards |
|---|---|---|
| England, France, Germany, Italy, Spain, Scotland | yes | yes |
| Belgium, Netherlands, Portugal | yes | no |
| Poland, Switzerland | no | no |

## `team_matches.parquet`

The same matches reshaped to the team's point of view: each match appears twice, once per team.
Sorted by `team_api_id`, `date` so rolling windows work directly. All `*_for` / `*_against` / `points`
columns are *post-match*.

| Column | Type | Description | Missing |
|---|---|---|---|
| `match_api_id` | int | → `matches` | – |
| `team_api_id` | int | The team this row is about | – |
| `opponent_api_id` | int | The other team | – |
| `is_home` | int | 1 if `team_api_id` played at home | – |
| `league`, `season`, `stage`, `date` | | Same as in `matches` | – |
| `goals_for`, `goals_against` | int | Full-time goals scored / conceded | – |
| `points` | int | 3 win, 1 draw, 0 loss | – |
| `ht_goals_for`, `ht_goals_against` | float | Half-time goals | 12.9 % |
| `shots_for`, `shots_against` | float | Total shots | 36.8 % |
| `shots_on_target_for`, `shots_on_target_against` | float | Shots on target | 36.8 % |
| `corners_for`, `corners_against` | float | Corners | 36.8 % |
| `fouls_for`, `fouls_against` | float | Fouls committed by the team / by the opponent | 36.8 % |
| `yellows_for`, `yellows_against` | float | Yellow cards | 36.8 % |
| `reds_for`, `reds_against` | float | Red cards | 36.8 % |
| `has_fd_stats`, `has_shot_stats` | bool | Same as in `matches` | – |

## `lineups.parquet`

`matches` lineup slots unpivoted to long form, with each player's FIFA attributes attached as-of the match date
(`attr_date <= date` always holds, so every column here is safe to use as a pre-match feature).
Empty lineup slots are dropped, so a match can have fewer than 22 rows.

| Column | Type | Description | Missing |
|---|---|---|---|
| `match_api_id` | int | → `matches` | – |
| `date` | datetime | Match date | – |
| `team_api_id` | int | Team the player started for | – |
| `player_api_id` | int | → `players` | – |
| `side` | str | `home` / `away` | – |
| `slot` | int | 1–11, lineup slot in `matches` (1 = goalkeeper) | – |
| `pos_x` | float | Left–right pitch position, 1–9 | 2.7 % |
| `pos_y` | float | Back–front pitch position, 1 = goalkeeper … 11 = striker | 2.7 % |
| `attr_date` | datetime | Date of the attribute snapshot used | – |
| `overall_rating` | float | FIFA overall rating, 0–100 | – |
| `potential` | float | FIFA potential rating, 0–100 | – |
| `preferred_foot` | str | `left` / `right` | – |
| `attacking_work_rate`, `defensive_work_rate` | str | `low` / `medium` / `high` | 4.1 % |
| *Attacking:* `crossing`, `finishing`, `heading_accuracy`, `short_passing`, `volleys` | float | 0–100 | `volleys` 1.2 % |
| *Skill:* `dribbling`, `curve`, `free_kick_accuracy`, `long_passing`, `ball_control` | float | 0–100 | `curve` 1.2 % |
| *Movement:* `acceleration`, `sprint_speed`, `agility`, `reactions`, `balance` | float | 0–100 | `agility`, `balance` 1.2 % |
| *Power:* `shot_power`, `jumping`, `stamina`, `strength`, `long_shots` | float | 0–100 | `jumping` 1.2 % |
| *Mentality:* `aggression`, `interceptions`, `positioning`, `vision`, `penalties` | float | 0–100 | `vision` 1.2 % |
| *Defending:* `marking`, `standing_tackle`, `sliding_tackle` | float | 0–100 | `sliding_tackle` 1.2 % |
| *Goalkeeping:* `gk_diving`, `gk_handling`, `gk_kicking`, `gk_positioning`, `gk_reflexes` | float | 0–100 | – |

A few `pos_x` / `pos_y` values are 0; treat them as unknown.

## `player_attributes.parquet`

Every FIFA snapshot of every player, 2007-02 to 2016-07 (about 17 per player). Same attribute columns as
`lineups` (`overall_rating` … `gk_reflexes`), keyed by `player_api_id` + `date`. Only needed for
player-over-time analysis; for match features use `lineups`.

Missing: work rates 3.6 %, `volleys` / `curve` / `agility` / `balance` / `jumping` / `vision` / `sliding_tackle` 1.0 %.

## `team_attributes.parquet`

FIFA team tactics, roughly one snapshot per team per year (2010–2015), 288 teams. Not joined to matches yet — use the
as-of join shown above. Matches before a team's first snapshot (all of 2008/09 and most of 2009/10) get no attributes.

| Column | Type | Description |
|---|---|---|
| `team_api_id` | int | → `teams` |
| `date` | datetime | Snapshot date |
| `buildUpPlaySpeed` | int | 0–100, how fast the team moves the ball forward |
| `buildUpPlayDribbling` | float | 0–100, tendency to dribble in build-up (**66.5 % missing**, only recorded from 2014) |
| `buildUpPlayPassing` | int | 0–100, passing distance in build-up (high = long balls) |
| `chanceCreationPassing` | int | 0–100, risk taken in the final pass |
| `chanceCreationCrossing` | int | 0–100, tendency to cross |
| `chanceCreationShooting` | int | 0–100, tendency to shoot |
| `defencePressure` | int | 0–100, how high up the pitch the team presses |
| `defenceAggression` | int | 0–100, aggressiveness in tackling |
| `defenceTeamWidth` | int | 0–100, defensive width |
| `buildUpPlaySpeedClass` | str | `Slow` / `Balanced` / `Fast` |
| `buildUpPlayDribblingClass` | str | `Little` / `Normal` / `Lots` |
| `buildUpPlayPassingClass` | str | `Short` / `Mixed` / `Long` |
| `buildUpPlayPositioningClass` | str | `Organised` / `Free Form` |
| `chanceCreationPassingClass` | str | `Safe` / `Normal` / `Risky` |
| `chanceCreationCrossingClass` | str | `Little` / `Normal` / `Lots` |
| `chanceCreationShootingClass` | str | `Little` / `Normal` / `Lots` |
| `chanceCreationPositioningClass` | str | `Organised` / `Free Form` |
| `defencePressureClass` | str | `Deep` / `Medium` / `High` |
| `defenceAggressionClass` | str | `Contain` / `Press` / `Double` |
| `defenceTeamWidthClass` | str | `Narrow` / `Normal` / `Wide` |
| `defenceDefenderLineClass` | str | `Cover` / `Offside Trap` |

## `players.parquet`

| Column | Type | Description |
|---|---|---|
| `player_api_id` | int | Player id (primary key) |
| `player_name` | str | Full name |
| `player_fifa_api_id` | int | Id in the FIFA game database |
| `birthday` | datetime | Date of birth — age at match = `match date − birthday` |
| `height_cm` | float | Height in cm |
| `weight_kg` | float | Weight in kg |

## `teams.parquet`

| Column | Type | Description |
|---|---|---|
| `team_api_id` | int | Team id (primary key) |
| `team_fifa_api_id` | float | Id in the FIFA game database (3.7 % missing) |
| `team_long_name` | str | e.g. `Manchester United` |
| `team_short_name` | str | e.g. `MUN` |

## `split.parquet`

The train/test assignment every model must use: `train_test_split(sorted match ids, test_size=0.3, random_state=42)`.
18,185 train and 7,794 test matches.

| Column | Type | Description |
|---|---|---|
| `match_api_id` | int | → `matches` |
| `split` | str | `train` / `test` |

```python
split = pd.read_parquet("datasets/split.parquet")
matches = matches.merge(split, on="match_api_id")
train, test = matches[matches.split == "train"], matches[matches.split == "test"]
```
