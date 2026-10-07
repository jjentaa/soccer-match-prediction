"""Model 2 answer to "team A (home) vs team B (away), who wins?" from the two expected starting lineups.

The real lineup is unknown before kick-off, so each team is assumed to field the XI (and formation) of its most
recent match before `date`. Player attributes are taken as-of `date` from player_attributes, so a later date
picks up newer FIFA ratings for the same players.

Usage:
    python model2_lineup/predict.py "Manchester United" "Chelsea"
    python model2_lineup/predict.py "Barcelona" "Real Madrid" --date 2015-11-21
Run train.py first: it writes outputs/model2.joblib.
"""
import argparse

import joblib
import numpy as np
import pandas as pd

from features import DATA, match_features, prepare_players
from train import CLASSES, OUT, reweight_draw

LINEUP_KEYS = ["team_api_id", "player_api_id", "pos_x", "pos_y", "slot"]


def find_team(query, teams):
    """team_api_id from an id, an exact name or a unique case-insensitive substring of the long or short name."""
    if str(query).isdigit():
        return int(query)
    q = str(query).lower()
    exact = teams[(teams["team_long_name"].str.lower() == q) | (teams["team_short_name"].str.lower() == q)]
    hits = exact if len(exact) else teams[teams["team_long_name"].str.lower().str.contains(q, regex=False)]
    if len(hits) != 1:
        names = ", ".join(hits["team_long_name"].head(10)) or "none"
        raise ValueError(f"'{query}' matches {len(hits)} teams ({names}); be more specific")
    return int(hits["team_api_id"].iloc[0])


def latest_xi(team_id, date, lineups):
    """The team's starting XI and pitch positions in its last match before `date`."""
    past = lineups[(lineups["team_api_id"] == team_id) & (lineups["date"] < date)]
    if past.empty:
        raise ValueError(f"team {team_id} has no lineup before {date.date()}")
    last = past[past["date"] == past["date"].max()]
    return last[LINEUP_KEYS], last["date"].iloc[0]


def lineup_rows(home_id, away_id, date, lineups, attributes):
    """Long lineup table for one hypothetical match, with attributes joined as-of `date`."""
    home, home_from = latest_xi(home_id, date, lineups)
    away, away_from = latest_xi(away_id, date, lineups)
    rows = pd.concat([home.assign(side="home"), away.assign(side="away")])
    rows = rows.assign(match_api_id=-1, date=date)

    snaps = attributes[attributes["date"] <= date].sort_values("date").groupby("player_api_id").tail(1)
    rows = rows.merge(snaps.rename(columns={"date": "attr_date"}), on="player_api_id", how="left")
    return rows, home_from, away_from


def predict(team_a, team_b, date=None):
    """Probabilities that team_a (at home) beats, draws with, or loses to team_b."""
    bundle = joblib.load(OUT / "model2.joblib")
    teams = pd.read_parquet(DATA / "teams.parquet")
    lineups = pd.read_parquet(DATA / "lineups.parquet", columns=["match_api_id", "date"] + LINEUP_KEYS)
    attributes = pd.read_parquet(DATA / "player_attributes.parquet")
    players = pd.read_parquet(DATA / "players.parquet")

    home_id, away_id = find_team(team_a, teams), find_team(team_b, teams)
    if home_id == away_id:
        raise ValueError("a team cannot play itself")
    last_seen = lineups.loc[lineups["team_api_id"].isin([home_id, away_id]), "date"].max()
    date = pd.Timestamp(date) if date is not None else last_seen + pd.Timedelta(days=1)

    rows, home_from, away_from = lineup_rows(home_id, away_id, date, lineups, attributes)
    X = match_features(prepare_players(rows, players)).reindex(columns=bundle["features"])

    raw = bundle["model"].predict_proba(X)
    adjusted = reweight_draw(raw, bundle["draw_weight"])[0]
    names = teams.set_index("team_api_id")["team_long_name"]
    return {
        "home": names[home_id], "away": names[away_id], "date": date.date().isoformat(),
        "home_xi_from": home_from.date().isoformat(), "away_xi_from": away_from.date().isoformat(),
        "home_avg_rating": round(float(X["home_ovr_mean"].iloc[0]), 1),
        "away_avg_rating": round(float(X["away_ovr_mean"].iloc[0]), 1),
        "p_H": round(float(raw[0, 2]), 3), "p_D": round(float(raw[0, 1]), 3), "p_A": round(float(raw[0, 0]), 3),
        "prediction": {"H": "home win", "D": "draw", "A": "away win"}[CLASSES[int(np.argmax(adjusted))]],
        "model": bundle["name"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("home", help="home team: name, short name or team_api_id")
    parser.add_argument("away", help="away team: name, short name or team_api_id")
    parser.add_argument("--date", help="match date YYYY-MM-DD (default: day after the later team's last match)")
    args = parser.parse_args()

    r = predict(args.home, args.away, args.date)
    print(f"{r['home']} (home) vs {r['away']} (away) on {r['date']}  [model: {r['model']}]")
    print(f"  expected XI from: home {r['home_xi_from']}, away {r['away_xi_from']}")
    print(f"  average overall rating: home {r['home_avg_rating']}, away {r['away_avg_rating']}")
    print(f"  P(home win) {r['p_H']:.1%}   P(draw) {r['p_D']:.1%}   P(away win) {r['p_A']:.1%}")
    print(f"  prediction (draw-weighted): {r['prediction']}")


if __name__ == "__main__":
    main()
