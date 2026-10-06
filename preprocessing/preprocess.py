"""Step 2 - clean and join the raw data into the tables in datasets/.

Outputs (parquet):
  teams.parquet             one row per team
  players.parquet           one row per player (birthday, height, weight)
  player_attributes.parquet FIFA attribute snapshots per player over time (cleaned)
  team_attributes.parquet   FIFA team tactic snapshots per team over time (cleaned)
  matches.parquet           one row per match: result, lineups, odds + football-data.co.uk match stats
  team_matches.parquet      matches in long form, one row per (match, team) - base for rolling "form" features
  lineups.parquet           one row per (match, side, slot) with the player's latest attributes *before* the match

Usage: python preprocessing/preprocess.py
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ESD_SQLITE = HERE / "raw" / "european_soccer" / "database.sqlite"
FOOTBALL_DATA_DIR = HERE / "raw" / "football_data"
DATASETS = HERE.parent / "datasets"

FD_LEAGUES = {
    "E0": "England Premier League",
    "F1": "France Ligue 1",
    "D1": "Germany 1. Bundesliga",
    "I1": "Italy Serie A",
    "SP1": "Spain LIGA BBVA",
    "N1": "Netherlands Eredivisie",
    "B1": "Belgium Jupiler League",
    "P1": "Portugal Liga ZON Sagres",
    "SC0": "Scotland Premier League",
}
# football-data.co.uk column -> our name (h_/a_ = home/away)
FD_STATS = {
    "HTHG": "h_ht_goals", "HTAG": "a_ht_goals",
    "HS": "h_shots", "AS": "a_shots",
    "HST": "h_shots_on_target", "AST": "a_shots_on_target",
    "HF": "h_fouls", "AF": "a_fouls",
    "HC": "h_corners", "AC": "a_corners",
    "HY": "h_yellows", "AY": "a_yellows",
    "HR": "h_reds", "AR": "a_reds",
}
ODDS = ["B365H", "B365D", "B365A", "BWH", "BWD", "BWA", "IWH", "IWD", "IWA",
        "LBH", "LBD", "LBA", "WHH", "WHD", "WHA", "VCH", "VCD", "VCA"]
WORK_RATES = ["low", "medium", "high"]


def load_esd() -> dict[str, pd.DataFrame]:
    con = sqlite3.connect(ESD_SQLITE)
    tables = ["Country", "League", "Team", "Team_Attributes", "Player", "Player_Attributes", "Match"]
    out = {t: pd.read_sql(f"SELECT * FROM {t}", con) for t in tables}
    con.close()
    return out


def clean_players(esd):
    players = esd["Player"].drop(columns="id")
    players["birthday"] = pd.to_datetime(players["birthday"])
    players["height_cm"] = players.pop("height")
    players["weight_kg"] = (players.pop("weight") * 0.453592).round(1)

    pa = esd["Player_Attributes"].drop(columns=["id", "player_fifa_api_id"])
    pa["date"] = pd.to_datetime(pa["date"])
    pa = pa.dropna(subset=["overall_rating"])
    pa = pa.drop_duplicates(["player_api_id", "date"], keep="last")
    # work-rate columns contain junk tokens ("norm", "o", "ormal", numbers, ...)
    for c in ["attacking_work_rate", "defensive_work_rate"]:
        pa[c] = pa[c].where(pa[c].isin(WORK_RATES))
    pa = pa.sort_values(["player_api_id", "date"]).reset_index(drop=True)
    return players, pa


def clean_teams(esd):
    teams = esd["Team"].drop(columns="id")
    ta = esd["Team_Attributes"].drop(columns=["id", "team_fifa_api_id"])
    ta["date"] = pd.to_datetime(ta["date"])
    ta = ta.drop_duplicates(["team_api_id", "date"], keep="last")
    ta = ta.sort_values(["team_api_id", "date"]).reset_index(drop=True)
    return teams, ta


def clean_matches(esd, teams):
    m = esd["Match"]
    slot_cols = [c for c in m.columns if "_player_" in c]
    keep = ["match_api_id", "league_id", "season", "stage", "date", "home_team_api_id",
            "away_team_api_id", "home_team_goal", "away_team_goal", *slot_cols, *ODDS]
    m = m[keep].copy()
    m["date"] = pd.to_datetime(m["date"])
    league = esd["League"].rename(columns={"id": "league_id", "name": "league"})[["league_id", "league"]]
    m = m.merge(league, on="league_id").drop(columns="league_id")
    names = teams.set_index("team_api_id")["team_long_name"]
    m["home_team"] = m["home_team_api_id"].map(names)
    m["away_team"] = m["away_team_api_id"].map(names)
    diff = m["home_team_goal"] - m["away_team_goal"]
    m["result"] = np.select([diff > 0, diff < 0], ["H", "A"], "D")
    return m


def load_football_data() -> pd.DataFrame:
    frames = []
    for path in sorted(FOOTBALL_DATA_DIR.glob("*.csv")):
        season, code = path.stem.split("_")
        df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
        df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG"])
        cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", *[c for c in FD_STATS if c in df.columns]]
        df = df[cols].copy()
        df["HomeTeam"] = df["HomeTeam"].str.strip()
        df["AwayTeam"] = df["AwayTeam"].str.strip()
        df["date"] = pd.to_datetime(df.pop("Date"), dayfirst=True, format="mixed")
        df["season"] = f"20{season[:2]}/20{season[2:]}"
        df["league"] = FD_LEAGUES[code]
        frames.append(df)
    return pd.concat(frames, ignore_index=True).rename(columns=FD_STATS)


def map_fd_teams(fd: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Attach ESD team ids to football-data rows.

    Team names differ between sources, so the mapping is learned from the data:
    rows sharing (league, date, score) are candidate pairs and each
    football-data name is assigned the ESD team it co-occurs with most often.
    """
    key = ["league", "date", "FTHG", "FTAG"]
    esd = matches.rename(columns={"home_team_goal": "FTHG", "away_team_goal": "FTAG"})
    cand = fd.merge(esd[[*key, "home_team_api_id", "away_team_api_id"]], on=key)
    votes = pd.concat([
        cand[["league", "HomeTeam", "home_team_api_id"]].set_axis(["league", "name", "team_api_id"], axis=1),
        cand[["league", "AwayTeam", "away_team_api_id"]].set_axis(["league", "name", "team_api_id"], axis=1),
    ])
    counts = votes.value_counts().reset_index(name="n").sort_values("n", ascending=False)
    mapping = counts.drop_duplicates(["league", "name"])
    # several spellings may point at one team (e.g. "Roda" / "Roda JC"), but a
    # mapping backed by only a handful of coincidental score matches is unsafe
    weak = mapping[mapping["n"] < 5]
    assert weak.empty, f"weakly supported team mappings:\n{weak}"
    mapping = mapping.set_index(["league", "name"])["team_api_id"]

    fd = fd.copy()
    for side in ["Home", "Away"]:
        idx = pd.MultiIndex.from_frame(fd[["league", f"{side}Team"]])
        fd[f"{side.lower()}_team_api_id"] = mapping.reindex(idx).to_numpy()
    unmapped = fd["home_team_api_id"].isna() | fd["away_team_api_id"].isna()
    print(f"  football-data rows without a team mapping: {unmapped.sum()}")
    return fd[~unmapped].astype({"home_team_api_id": int, "away_team_api_id": int})


def join_football_data(matches: pd.DataFrame, fd: pd.DataFrame) -> pd.DataFrame:
    """Left-join football-data stats onto ESD matches.

    Key is (league, season, home, away); when a fixture is played more than
    once per season (e.g. Scotland) the closest date wins.
    """
    key = ["league", "season", "home_team_api_id", "away_team_api_id"]
    stat_cols = [c for c in FD_STATS.values() if c in fd.columns]
    cand = matches[["match_api_id", "date", *key]].merge(
        fd[[*key, "date", "FTHG", "FTAG", *stat_cols]], on=key, suffixes=("", "_fd"))
    cand["gap"] = (cand["date"] - cand["date_fd"]).abs()
    cand = cand[cand["gap"] <= pd.Timedelta(days=3)].sort_values("gap")
    cand = cand.drop_duplicates("match_api_id").drop_duplicates([*key, "date_fd"])
    out = matches.merge(cand[["match_api_id", "FTHG", "FTAG", *stat_cols]], on="match_api_id", how="left")

    matched = out["FTHG"].notna()
    score_ok = (out["FTHG"] == out["home_team_goal"]) & (out["FTAG"] == out["away_team_goal"])
    in_scope = out["league"].isin(FD_LEAGUES.values())
    print(f"  matched {matched.sum()}/{in_scope.sum()} matches in the 9 covered leagues; "
          f"score disagreements: {(matched & ~score_ok).sum()}")
    out["has_fd_stats"] = matched
    # Belgium, Netherlands and Portugal only publish half-time scores, no shot/corner/card stats
    out["has_shot_stats"] = out["h_shots"].notna()
    return out.drop(columns=["FTHG", "FTAG"])


def build_lineups(matches: pd.DataFrame, pa: pd.DataFrame) -> pd.DataFrame:
    """Long lineup table with each player's most recent attributes on or before match day."""
    parts = []
    for side in ["home", "away"]:
        for slot in range(1, 12):
            part = matches[["match_api_id", "date", f"{side}_team_api_id", f"{side}_player_{slot}",
                            f"{side}_player_X{slot}", f"{side}_player_Y{slot}"]]
            part = part.set_axis(["match_api_id", "date", "team_api_id", "player_api_id", "pos_x", "pos_y"], axis=1)
            parts.append(part.assign(side=side, slot=slot))
    lu = pd.concat(parts, ignore_index=True).dropna(subset=["player_api_id"])
    lu["player_api_id"] = lu["player_api_id"].astype("int64")
    lu = pd.merge_asof(
        lu.sort_values("date"),
        pa.rename(columns={"date": "attr_date"}).sort_values("attr_date"),
        left_on="date", right_on="attr_date", by="player_api_id", direction="backward")
    return lu.sort_values(["date", "match_api_id", "side", "slot"]).reset_index(drop=True)


def build_team_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, team): own stats as *_for, opponent's as *_against."""
    base = ["match_api_id", "league", "season", "stage", "date", "has_fd_stats", "has_shot_stats"]
    stats = sorted({c[2:] for c in matches.columns if c.startswith("h_")})
    frames = []
    for me, opp, flag in [("home", "away", 1), ("away", "home", 0)]:
        df = matches[base].copy()
        df["team_api_id"] = matches[f"{me}_team_api_id"]
        df["opponent_api_id"] = matches[f"{opp}_team_api_id"]
        df["is_home"] = flag
        df["goals_for"] = matches[f"{me}_team_goal"]
        df["goals_against"] = matches[f"{opp}_team_goal"]
        for s in stats:
            df[f"{s}_for"] = matches[f"{me[0]}_{s}"]
            df[f"{s}_against"] = matches[f"{opp[0]}_{s}"]
        frames.append(df)
    tm = pd.concat(frames, ignore_index=True)
    gd = tm["goals_for"] - tm["goals_against"]
    tm["points"] = np.select([gd > 0, gd == 0], [3, 1], 0)
    return tm.sort_values(["team_api_id", "date"]).reset_index(drop=True)


def main() -> None:
    DATASETS.mkdir(parents=True, exist_ok=True)
    print("loading European Soccer Database")
    esd = load_esd()
    players, pa = clean_players(esd)
    teams, ta = clean_teams(esd)
    matches = clean_matches(esd, teams)

    print("joining football-data.co.uk")
    fd = map_fd_teams(load_football_data(), matches)
    matches = join_football_data(matches, fd)
    matches = matches.sort_values(["date", "match_api_id"]).reset_index(drop=True)

    lineups = build_lineups(matches, pa)
    team_matches = build_team_matches(matches)

    outputs = {
        "teams": teams, "players": players, "player_attributes": pa, "team_attributes": ta,
        "matches": matches, "team_matches": team_matches, "lineups": lineups,
    }
    for name, df in outputs.items():
        df.to_parquet(DATASETS / f"{name}.parquet", index=False)
        print(f"  {name:18s} {df.shape[0]:>7,} rows x {df.shape[1]:>3} cols")


if __name__ == "__main__":
    main()
