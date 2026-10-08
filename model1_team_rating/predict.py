"""Model 1 answer to "team A (home) vs team B (away), who wins?" from the two teams' ratings.

Ratings are fitted on the training matches of the teams' league dated before `date`, with the
half-life and draw weight chosen by train.py. Both teams must play in the same league: ratings are
relative to the league average, so they cannot be compared across leagues.

Usage:
    python model1_team_rating/predict.py "Barcelona" "Real Madrid CF"
    python model1_team_rating/predict.py "Manchester United" "Chelsea" --date 2015-11-21
Run train.py first: it writes outputs/metrics.json.
"""
import argparse
import json

import numpy as np
import pandas as pd

from rating import DATA, WINDOW_DAYS, fit_league, expected_goals, load_matches, outcome_probs, score_grid
from train import CLASSES, OUT, apply_draw_weight


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


def team_league(team_id, matches):
    """League of the team's most recent match in the data."""
    played = matches[(matches["home_team_api_id"] == team_id) | (matches["away_team_api_id"] == team_id)]
    return played.sort_values("date")["league"].iloc[-1], played["date"].max()


def predict(team_a, team_b, date=None):
    """Probabilities that team_a (at home) beats, draws with, or loses to team_b."""
    settings = json.loads((OUT / "metrics.json").read_text())
    teams = pd.read_parquet(DATA / "teams.parquet")
    names = teams.set_index("team_api_id")["team_long_name"]
    matches = load_matches()

    home_id, away_id = find_team(team_a, teams), find_team(team_b, teams)
    if home_id == away_id:
        raise ValueError("a team cannot play itself")
    home_league, home_last = team_league(home_id, matches)
    away_league, away_last = team_league(away_id, matches)
    if home_league != away_league:
        raise ValueError(f"{names[home_id]} plays in {home_league}, {names[away_id]} in {away_league}; "
                         "ratings are only comparable within one league")
    date = pd.Timestamp(date) if date is not None else max(home_last, away_last) + pd.Timedelta(days=1)

    # same data rule as in training: training matches of the league, dated before `date`
    train = matches[(matches["split"] == "train") & (matches["league"] == home_league)]
    history = train[(train["date"] < date) & (train["date"] >= date - pd.Timedelta(days=WINDOW_DAYS))]
    seen = set(history["home_team_api_id"]) | set(history["away_team_api_id"])
    for team_id in (home_id, away_id):
        if team_id not in seen:
            print(f"  warning: {names[team_id]} has no match in the history window; rated as league average")
    league_teams = sorted(seen | {home_id, away_id})
    params = fit_league(history, league_teams, date, settings["half_life_days"], use_rho=True)

    lam, mu = expected_goals(params, [home_id], [away_id])
    P = np.column_stack(outcome_probs(lam, mu, params["rho"]))
    weighted = apply_draw_weight(P, settings["draw_weight"])[0]

    # five most likely scores from the score grid
    grid = score_grid(lam, mu, params["rho"])[0]
    top = np.argsort(grid, axis=None)[::-1][:5]
    scores = [(f"{x}-{y}", round(float(grid[x, y]), 3)) for x, y in zip(*np.unravel_index(top, grid.shape))]

    # rank among the teams of the latest season before `date`, so relegated teams do not count
    # (uses only who played whom, no results)
    before = matches[(matches["league"] == home_league) & (matches["date"] < date)]
    season = before[before["season"] == before["season"].max()]
    current = sorted((set(season["home_team_api_id"]) | set(season["away_team_api_id"]) | {home_id, away_id}) & seen)
    rating = (params["attack"] + params["defence"])[current].sort_values(ascending=False)
    rank = {team_id: k + 1 for k, team_id in enumerate(rating.index)}

    def team_info(team_id):
        return {"name": names[team_id], "attack": round(float(params["attack"][team_id]), 2),
                "defence": round(float(params["defence"][team_id]), 2),
                "rating": round(float(params["attack"][team_id] + params["defence"][team_id]), 2),
                "rank": rank.get(team_id), "league_size": len(rating)}

    return {
        "home": team_info(home_id), "away": team_info(away_id),
        "league": home_league, "date": date.date().isoformat(), "matches_used": len(history),
        "expected_goals": (round(float(lam[0]), 2), round(float(mu[0]), 2)),
        "p_H": round(float(P[0, 0]), 3), "p_D": round(float(P[0, 1]), 3), "p_A": round(float(P[0, 2]), 3),
        "top_scores": scores,
        "prediction": {"H": "home win", "D": "draw", "A": "away win"}[CLASSES[int(np.argmax(weighted))]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("home", help="home team: name, short name or team_api_id")
    parser.add_argument("away", help="away team: name, short name or team_api_id")
    parser.add_argument("--date", help="match date YYYY-MM-DD (default: day after the later team's last match)")
    args = parser.parse_args()

    r = predict(args.home, args.away, args.date)
    print(f"{r['home']['name']} (home) vs {r['away']['name']} (away) on {r['date']}")
    print(f"  {r['league']}, ratings from {r['matches_used']} training matches")
    for side in ("home", "away"):
        t = r[side]
        print(f"  {side}: attack {t['attack']:+.2f}  defence {t['defence']:+.2f}  "
              f"rating {t['rating']:+.2f}  (rank {t['rank']} of {t['league_size']})")
    print(f"  expected goals: {r['expected_goals'][0]} - {r['expected_goals'][1]}")
    print(f"  P(home win) {r['p_H']:.1%}   P(draw) {r['p_D']:.1%}   P(away win) {r['p_A']:.1%}")
    print("  most likely scores: " + ", ".join(f"{s} ({p:.1%})" for s, p in r["top_scores"]))
    print(f"  prediction (draw-weighted): {r['prediction']}")


if __name__ == "__main__":
    main()
