"""Model 2 features: one row per match describing the two starting lineups.

Every input column is pre-match (lineups carry FIFA attributes joined as-of the match date),
so these features are safe to use regardless of how the data is split.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "datasets"

SEED = 42

GK_ATTRS = ["gk_diving", "gk_handling", "gk_kicking", "gk_positioning", "gk_reflexes"]
OUTFIELD_ATTRS = [
    "overall_rating", "potential",
    "crossing", "finishing", "heading_accuracy", "short_passing", "volleys",
    "dribbling", "curve", "free_kick_accuracy", "long_passing", "ball_control",
    "acceleration", "sprint_speed", "agility", "reactions", "balance",
    "shot_power", "jumping", "stamina", "strength", "long_shots",
    "aggression", "interceptions", "positioning", "vision", "penalties",
    "marking", "standing_tackle", "sliding_tackle",
    "attacking_work_rate", "defensive_work_rate",
]
WORK_RATE = {"low": 0, "medium": 1, "high": 2}
LINES = ["DEF", "MID", "FWD"]


def split_ids(matches: pd.DataFrame) -> pd.Series:
    """Team-agreed split, read from datasets/split.parquet so every model uses the same matches.

    Re-running train_test_split here would not reproduce it: the shuffle depends on row order, and
    split.parquet was drawn over match ids in sorted order, not in matches.parquet file order.
    """
    split = pd.read_parquet(DATA / "split.parquet").set_index("match_api_id")["split"]
    return split.reindex(matches["match_api_id"].values).rename("set")


def assign_line(df: pd.DataFrame) -> pd.Series:
    """GK / DEF / MID / FWD from the back-front pitch position; fall back to the slot when it is unknown."""
    y = df["pos_y"].where(df["pos_y"] > 0)
    by_pos = pd.cut(y, bins=[0, 1, 4, 8, 11], labels=["GK", "DEF", "MID", "FWD"]).astype(object)
    by_slot = pd.cut(df["slot"], bins=[0, 1, 5, 9, 11], labels=["GK", "DEF", "MID", "FWD"]).astype(object)
    return by_pos.fillna(by_slot)


def prepare_players(lineups: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Encode categoricals and add body / age columns to the long lineup table."""
    df = lineups.merge(players[["player_api_id", "birthday", "height_cm", "weight_kg"]],
                       on="player_api_id", how="left")
    for col in ["attacking_work_rate", "defensive_work_rate"]:
        df[col] = df[col].map(WORK_RATE)
    df["left_foot"] = (df["preferred_foot"] == "left").astype(float)
    df["age"] = (df["date"] - df["birthday"]).dt.days / 365.25
    df["line"] = assign_line(df)
    return df


def side_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate players to one row per (match, side)."""
    keys = ["match_api_id", "side"]
    g = df.groupby(keys)
    parts = [
        g.size().rename("n_players"),
        g["overall_rating"].agg(["mean", "max", "min", "std"]).add_prefix("ovr_"),
        g["potential"].agg(["mean", "max"]).add_prefix("pot_"),
        g[["age", "height_cm", "weight_kg", "left_foot"]].mean().add_suffix("_mean"),
        g["age"].std().rename("age_std"),
    ]

    gk = df[df["line"] == "GK"].groupby(keys)[GK_ATTRS + ["overall_rating"]].mean()
    parts.append(gk.add_prefix("GK_"))

    outfield = df[df["line"] != "GK"]
    per_line = outfield.groupby(keys + ["line"])[OUTFIELD_ATTRS].mean().unstack("line")
    per_line.columns = [f"{line}_{attr}" for attr, line in per_line.columns]
    parts.append(per_line)

    # formation shape, e.g. 4-4-2 -> DEF_n=4, MID_n=4, FWD_n=2
    counts = outfield.groupby(keys + ["line"]).size().unstack("line", fill_value=0)
    parts.append(counts.reindex(columns=LINES, fill_value=0).add_suffix("_n"))

    return pd.concat(parts, axis=1)


def match_features(df: pd.DataFrame) -> pd.DataFrame:
    """Home / away side features side by side, plus home-minus-away differences."""
    sides = side_features(df).unstack("side")
    home = sides.xs("home", axis=1, level="side")
    away = sides.xs("away", axis=1, level="side")
    out = pd.concat([home.add_prefix("home_"), away.add_prefix("away_"), (home - away).add_prefix("diff_")], axis=1)
    return out


def build_dataset() -> pd.DataFrame:
    """match_api_id, set, result + lineup features for every match (NaN where a lineup is missing)."""
    matches = pd.read_parquet(DATA / "matches.parquet")
    lineups = pd.read_parquet(DATA / "lineups.parquet")
    players = pd.read_parquet(DATA / "players.parquet")

    feats = match_features(prepare_players(lineups, players))
    base = matches[["match_api_id", "league", "season", "date", "result"]].copy()
    base["set"] = base["match_api_id"].map(split_ids(matches))
    return base.merge(feats, left_on="match_api_id", right_index=True, how="left")


if __name__ == "__main__":
    ds = build_dataset()
    print(ds.shape)
    print(ds["set"].value_counts())
    print(ds.filter(like="diff_").columns.size, "diff features")
