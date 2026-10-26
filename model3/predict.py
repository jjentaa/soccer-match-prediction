"""Model 3 answer to "team A (home) vs team B (away), who wins?" from the two teams' play styles.

The hypothetical match is appended to the real ones and run through the same feature builder, so the
graphs it sees are built exactly as in training: from training matches dated before `date`. Each team
is assumed to line up in the formation of its last match. Both teams must play in the same league,
because the similarity and results graphs are per league.

Usage:
    python model3/predict.py "Barcelona" "Real Madrid CF"
    python model3/predict.py "Manchester United" "Chelsea" --date 2015-11-21
Run build_features.py and train.py first: they write styles.joblib and model3.joblib.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from build_features import DATA, LINES, TACTICS, assemble, load_matches

HERE = Path(__file__).resolve().parent
HYPOTHETICAL = -1
OUTCOME = {"H": "home win", "D": "draw", "A": "away win"}


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


def last_match(team_id, matches, date):
    played = matches[((matches["home_team_api_id"] == team_id) | (matches["away_team_api_id"] == team_id))
                     & (matches["date"] < date)]
    if played.empty:
        raise ValueError(f"team {team_id} has no match before {date.date()}")
    return played.sort_values("date").iloc[-1]


def last_formation(team_id, side, lineups, date) -> dict:
    """Formation line counts of the team's last lineup before `date`, named as the given side's features."""
    past = lineups[(lineups["team_api_id"] == team_id) & (lineups["date"] < date) & (lineups["pos_y"] > 1)]
    if past.empty:
        return {}
    xi = past[past["date"] == past["date"].max()]
    return {f"{side}_n_{line}": int(xi["pos_y"].between(lo, hi).sum()) for line, (lo, hi) in LINES.items()}


def describe_style(vector) -> str:
    """The two tactics that set a team furthest from the average, e.g. '+defencePressure, −buildUpPlaySpeed'."""
    z = pd.Series(vector, index=TACTICS)
    top = z.abs().sort_values(ascending=False).index[:2]
    return ", ".join(f"{'+' if z[t] > 0 else '−'}{t}" for t in top)


def predict(team_a, team_b, date=None):
    """Probabilities that team_a (at home) beats, draws with, or loses to team_b."""
    styles, bundle = joblib.load(HERE / "styles.joblib"), joblib.load(HERE / "model3.joblib")
    teams = pd.read_parquet(DATA / "teams.parquet")
    names = teams.set_index("team_api_id")["team_long_name"]
    matches = load_matches()

    home_id, away_id = find_team(team_a, teams), find_team(team_b, teams)
    if home_id == away_id:
        raise ValueError("a team cannot play itself")
    latest = pd.Timestamp(date) if date is not None else matches["date"].max() + pd.Timedelta(days=1)
    home_last, away_last = last_match(home_id, matches, latest), last_match(away_id, matches, latest)
    if home_last["league"] != away_last["league"]:
        raise ValueError(f"{names[home_id]} plays in {home_last['league']}, {names[away_id]} in "
                         f"{away_last['league']}; the style graphs are built per league")
    date = pd.Timestamp(date) if date is not None else max(home_last["date"], away_last["date"]) + pd.Timedelta(days=1)

    # the hypothetical match is marked "test" so that it never feeds the graphs; its result is a placeholder
    row = {"match_api_id": HYPOTHETICAL, "league": home_last["league"], "season": home_last["season"], "date": date,
           "home_team_api_id": home_id, "away_team_api_id": away_id, "result": "H", "split": "test"}
    matches = pd.concat([matches, pd.DataFrame([row])], ignore_index=True)

    lineups = pd.read_parquet(DATA / "lineups.parquet", columns=["team_api_id", "date", "pos_y"])
    formation = {"match_api_id": HYPOTHETICAL, **last_formation(home_id, "home", lineups, date),
                 **last_formation(away_id, "away", lineups, date)}
    feats = assemble(matches, pd.read_parquet(DATA / "team_attributes.parquet"), pd.DataFrame([formation]),
                     styles["scaler"], styles["kmeans"], only={HYPOTHETICAL})
    x = feats[feats["match_api_id"] == HYPOTHETICAL].iloc[[0]]

    model = bundle["model"]
    proba = dict(zip(model.classes_, model.predict_proba(x.reindex(columns=bundle["features"]))[0]))
    f = x.iloc[0]

    def side(prefix):
        has_style = not np.isnan(f[f"{prefix}_style"])
        return {"style": f"S{int(f[f'{prefix}_style'])}" if has_style else None,
                "style_traits": describe_style([f[f"{prefix}_{t}"] for t in TACTICS]) if has_style else None,
                "pagerank": None if np.isnan(f[f"pagerank_{prefix}"]) else round(float(f[f"pagerank_{prefix}"]), 2)}

    def number(col, digits=2):
        if col not in f or pd.isna(f[col]):
            return None
        return int(f[col]) if digits == 0 else round(float(f[col]), digits)

    return {
        "home": names[home_id], "away": names[away_id], "league": home_last["league"], "date": date.date().isoformat(),
        "home_side": side("home"), "away_side": side("away"),
        "matchup": {"p_H": number("matchup_p_home", 3), "p_D": number("matchup_p_draw", 3),
                    "p_A": number("matchup_p_away", 3), "matches": number("matchup_n", 0)},
        "home_ppg_vs_similar_to_away": number("home_ppg_vs_similar_to_away"),
        "home_n_vs_similar_to_away": number("home_n_vs_similar_to_away", 0),
        "away_ppg_vs_similar_to_home": number("away_ppg_vs_similar_to_home"),
        "away_n_vs_similar_to_home": number("away_n_vs_similar_to_home", 0),
        "style_distance": number("style_distance"),
        "p_H": round(float(proba["H"]), 3), "p_D": round(float(proba["D"]), 3), "p_A": round(float(proba["A"]), 3),
        "prediction": OUTCOME[max(proba, key=proba.get)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("home", help="home team: name, short name or team_api_id")
    parser.add_argument("away", help="away team: name, short name or team_api_id")
    parser.add_argument("--date", help="match date YYYY-MM-DD (default: day after the later team's last match)")
    args = parser.parse_args()

    r = predict(args.home, args.away, args.date)
    show = lambda v, unit="": "n/a" if v is None else f"{v}{unit}"
    print(f"{r['home']} (home) vs {r['away']} (away) on {r['date']}")
    print(f"  {r['league']}")
    for label, s in (("home", r["home_side"]), ("away", r["away_side"])):
        style = f"{s['style']} ({s['style_traits']})" if s["style"] else "no tactic snapshot yet"
        print(f"  {label}: style {style};  results-graph PageRank {show(s['pagerank'])} (1.00 = league average)")
    m = r["matchup"]
    if m["matches"] is not None:
        print(f"  style matchup, {r['home_side']['style']} at home vs {r['away_side']['style']} away, over "
              f"{m['matches']:,} earlier matches: home {m['p_H']:.1%}  draw {m['p_D']:.1%}  away {m['p_A']:.1%}")
        print(f"  points per game against the opponent and teams that play like it: "
              f"home {show(r['home_ppg_vs_similar_to_away'])} ({show(r['home_n_vs_similar_to_away'])} matches), "
              f"away {show(r['away_ppg_vs_similar_to_home'])} ({show(r['away_n_vs_similar_to_home'])} matches)")
    print(f"  P(home win) {r['p_H']:.1%}   P(draw) {r['p_D']:.1%}   P(away win) {r['p_A']:.1%}")
    print(f"  prediction: {r['prediction']}")


if __name__ == "__main__":
    main()
