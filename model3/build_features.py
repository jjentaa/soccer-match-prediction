"""Model 3, step 1 - turn team play style into graph features, one row per match.

Question: does a team that plays style X tend to beat a team that plays style Y?

  1. Style      each team's FIFA tactic vector (as of match day) is clustered into K styles
  2. Graphs     a) style matchup graph   nodes = styles, edges = how style X fares against style Y
                b) style similarity graph nodes = teams,  edges = k nearest neighbours in tactic space
                c) results graph          nodes = teams,  edges = loser -> winner
  3. Features   edge weights, PageRank, degree centrality, clustering coefficient, density

Leakage rule: every graph that carries match outcomes is built only from *training* matches
dated strictly before the match being described, so no test label reaches any feature.

Usage: python model3/build_features.py     -> model3/features.parquet, model3/styles.parquet
"""

from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "datasets"

SEED = 42
TACTICS = ["buildUpPlaySpeed", "buildUpPlayPassing", "chanceCreationPassing", "chanceCreationCrossing",
           "chanceCreationShooting", "defencePressure", "defenceAggression", "defenceTeamWidth"]
K_RANGE = range(3, 9)       # candidate numbers of styles, chosen by silhouette score
KNN = 3                     # neighbours per team in the style similarity graph
RESULTS_WINDOW_DAYS = 365   # history used for the results graph
SIMILAR_WINDOW_DAYS = 730   # history used for "record against similar-style teams"
PRIOR = 5                   # pseudo-matches pulling sparse style matchup rates toward the overall rate
# pitch rows (pos_y) -> line of the formation; row 1 is the goalkeeper
LINES = {"def": (2, 4), "dm": (5, 6), "mid": (7, 7), "am": (8, 9), "fwd": (10, 11)}


def fit_styles(team_attributes: pd.DataFrame):
    """Cluster tactic snapshots into K styles; K maximises the silhouette score."""
    scaler = StandardScaler().fit(team_attributes[TACTICS])
    z = scaler.transform(team_attributes[TACTICS])
    scores = {}
    for k in K_RANGE:
        labels = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(z)
        scores[k] = silhouette_score(z, labels)
    best_k = max(scores, key=scores.get)
    print("  silhouette by K:", {k: round(s, 3) for k, s in scores.items()}, "-> K =", best_k)
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(z)
    return scaler, kmeans


def attach_tactics(matches, team_attributes, scaler, kmeans) -> pd.DataFrame:
    """As-of join: each side gets its latest tactic snapshot on or before match day, and its style."""
    snaps = team_attributes[["team_api_id", "date", *TACTICS]].rename(columns={"date": "snap_date"})
    snaps[TACTICS] = scaler.transform(snaps[TACTICS])   # z-scores, so distances weigh tactics equally
    snaps["style"] = kmeans.predict(snaps[TACTICS].to_numpy())
    snaps = snaps.sort_values("snap_date")
    out = matches.sort_values("date")
    for side in ["home", "away"]:
        side_snaps = snaps.rename(columns={"team_api_id": f"{side}_team_api_id"}).add_prefix(f"{side}_")
        side_snaps = side_snaps.rename(columns={f"{side}_{side}_team_api_id": f"{side}_team_api_id"})
        out = pd.merge_asof(out, side_snaps, left_on="date", right_on=f"{side}_snap_date",
                            by=f"{side}_team_api_id", direction="backward")
    return out


def formation_features(lineups: pd.DataFrame) -> pd.DataFrame:
    """Number of starters in each line of the formation, per match and side (known before kick-off)."""
    lu = lineups[lineups["pos_y"] > 1]
    counts = {name: lu[lu["pos_y"].between(lo, hi)].groupby(["match_api_id", "side"]).size()
              for name, (lo, hi) in LINES.items()}
    wide = pd.DataFrame(counts).fillna(0).unstack("side")
    wide.columns = [f"{side}_n_{line}" for line, side in wide.columns]
    return wide.reset_index()


def results_graph_features(history: list, teams: set) -> dict:
    """PageRank on the loser -> winner graph (draws point both ways with half weight)."""
    g = nx.DiGraph()
    for home, away, result in history:
        if result == "H":
            edges = [(away, home, 1.0)]
        elif result == "A":
            edges = [(home, away, 1.0)]
        else:
            edges = [(home, away, 0.5), (away, home, 0.5)]
        for u, v, w in edges:
            g.add_edge(u, v, weight=g.get_edge_data(u, v, {"weight": 0})["weight"] + w)
    if g.number_of_nodes() < 3:
        return {}
    n = g.number_of_nodes()
    strength = nx.pagerank(g, weight="weight")              # rank flows toward teams that win
    weakness = nx.pagerank(g.reverse(), weight="weight")    # rank flows toward teams that lose
    density = nx.density(g)
    return {t: (strength[t] * n, weakness[t] * n, density) for t in teams if t in g}


def similarity_graph(vectors: dict) -> nx.Graph:
    """Undirected kNN graph over the teams of one league, in tactic space."""
    g = nx.Graph()
    teams = list(vectors)
    g.add_nodes_from(teams)
    if len(teams) <= KNN:
        return g
    x = np.array([vectors[t] for t in teams])
    dist = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    for i, team in enumerate(teams):
        for j in np.argsort(dist[i])[1:KNN + 1]:
            g.add_edge(team, teams[j], distance=dist[i, j])
    return g


def record_against(team, opponents, pair_history, since) -> tuple[float, int]:
    """Points per game `team` took from `opponents` in training matches since `since`."""
    points = [p for opp in opponents for d, p in pair_history.get((team, opp), ()) if d >= since]
    return (float(np.mean(points)), len(points)) if points else (np.nan, 0)


def latest_snapshot(snapshots: dict, team, date):
    """(snapshot date, tactic vector) of the team's latest snapshot on or before `date`, or None."""
    dated = [s for s in snapshots.get(team, ()) if s[0] <= date]
    return dated[-1] if dated else None


def build(matches: pd.DataFrame, season_teams: dict, snapshots: dict) -> pd.DataFrame:
    k = int(np.nanmax(matches[["home_style", "away_style"]].to_numpy())) + 1
    is_train = matches["split"].eq("train").to_numpy()

    matchup = np.zeros((k, k, 3))                 # [home style, away style] -> counts of H, D, A
    overall = np.zeros(3)
    style_rank = {}
    league_history = defaultdict(list)            # league -> [(date, home, away, result)]
    pair_history = defaultdict(list)              # (team, opponent) -> [(date, points for team)]
    sim_cache = {}                                # league -> (snapshot key, graph, centralities)

    rows = []
    for date, day in matches.groupby("date", sort=True):
        for league, lg in day.groupby("league"):
            teams = set(lg["home_team_api_id"]) | set(lg["away_team_api_id"])
            since = date - pd.Timedelta(days=RESULTS_WINDOW_DAYS)
            league_history[league] = [h for h in league_history[league] if h[0] >= since]
            res = results_graph_features([h[1:] for h in league_history[league]], teams)

            # similarity graph over this league-season's teams; rebuilt only when a tactic snapshot changes
            current = {t: snap for t in season_teams[(league, lg["season"].iloc[0])]
                       if (snap := latest_snapshot(snapshots, t, date)) is not None}
            vectors = {t: vec for t, (_, vec) in current.items()}
            key = tuple(sorted((t, snap_date) for t, (snap_date, _) in current.items()))
            if sim_cache.get(league, (None,))[0] != key:
                g = similarity_graph(vectors)
                degree = nx.degree_centrality(g) if g.number_of_nodes() > 1 else {}
                sim_cache[league] = (key, g, degree, nx.clustering(g), nx.density(g))
            _, sim, sim_degree, sim_clustering, sim_density = sim_cache[league]
            similar_since = date - pd.Timedelta(days=SIMILAR_WINDOW_DAYS)

            for m in lg.itertuples():
                h, a = m.home_team_api_id, m.away_team_api_id
                row = {"match_api_id": m.match_api_id}

                # a) style matchup graph: edge weight home style -> away style, and style centrality
                if not (np.isnan(m.home_style) or np.isnan(m.away_style)):
                    hs, as_ = int(m.home_style), int(m.away_style)
                    counts = matchup[hs, as_]
                    base = overall / overall.sum() if overall.sum() else np.full(3, 1 / 3)
                    rate = (counts + PRIOR * base) / (counts.sum() + PRIOR)
                    row |= {"matchup_p_home": rate[0], "matchup_p_draw": rate[1], "matchup_p_away": rate[2],
                            "matchup_n": counts.sum(),
                            "style_rank_home": style_rank.get(hs, np.nan),
                            "style_rank_away": style_rank.get(as_, np.nan)}

                # b) style similarity graph
                if h in sim and a in sim:
                    row |= {"sim_degree_home": sim_degree[h], "sim_degree_away": sim_degree[a],
                            "sim_clustering_home": sim_clustering[h], "sim_clustering_away": sim_clustering[a],
                            "sim_density": sim_density,
                            "sim_are_neighbours": int(sim.has_edge(h, a)),
                            "style_distance": float(np.linalg.norm(vectors[h] - vectors[a]))}
                    # how did each side do against teams that play like its opponent today?
                    ppg, n = record_against(h, [a, *sim.neighbors(a)], pair_history, similar_since)
                    row |= {"home_ppg_vs_similar_to_away": ppg, "home_n_vs_similar_to_away": n}
                    ppg, n = record_against(a, [h, *sim.neighbors(h)], pair_history, similar_since)
                    row |= {"away_ppg_vs_similar_to_home": ppg, "away_n_vs_similar_to_home": n}

                # c) results graph
                if h in res:
                    row |= {"pagerank_home": res[h][0], "weakrank_home": res[h][1], "results_density": res[h][2]}
                if a in res:
                    row |= {"pagerank_away": res[a][0], "weakrank_away": res[a][1], "results_density": res[a][2]}
                rows.append(row)

        # only now add today's *training* matches to the graphs
        changed = False
        for m in day[is_train[day.index]].itertuples():
            h, a, r = m.home_team_api_id, m.away_team_api_id, m.result
            league_history[m.league].append((date, h, a, r))
            pair_history[(h, a)].append((date, {"H": 3, "D": 1, "A": 0}[r]))
            pair_history[(a, h)].append((date, {"H": 0, "D": 1, "A": 3}[r]))
            if not (np.isnan(m.home_style) or np.isnan(m.away_style)):
                matchup[int(m.home_style), int(m.away_style), "HDA".index(r)] += 1
                overall["HDA".index(r)] += 1
                changed = True
        if changed:
            style_rank = style_pagerank(matchup)

    return pd.DataFrame(rows)


def style_pagerank(matchup: np.ndarray) -> dict:
    """PageRank over styles: an edge runs from the losing style to the winning style."""
    k = matchup.shape[0]
    g = nx.DiGraph()
    g.add_nodes_from(range(k))
    for hs in range(k):
        for as_ in range(k):
            home_wins, draws, away_wins = matchup[hs, as_]
            for u, v, w in [(as_, hs, home_wins + draws / 2), (hs, as_, away_wins + draws / 2)]:
                if w > 0 and u != v:
                    g.add_edge(u, v, weight=g.get_edge_data(u, v, {"weight": 0})["weight"] + w)
    if g.number_of_edges() == 0:
        return {}
    return {s: r * k for s, r in nx.pagerank(g, weight="weight").items()}


def main() -> None:
    matches = pd.read_parquet(DATA / "matches.parquet")
    matches = matches[["match_api_id", "league", "season", "date", "home_team_api_id", "away_team_api_id", "result"]]
    matches = matches.merge(pd.read_parquet(DATA / "split.parquet"), on="match_api_id")
    team_attributes = pd.read_parquet(DATA / "team_attributes.parquet")

    print("clustering tactic snapshots into styles")
    scaler, kmeans = fit_styles(team_attributes)
    matches = attach_tactics(matches, team_attributes, scaler, kmeans).reset_index(drop=True)

    # lookups used while walking through the matches in date order
    long = pd.concat([matches[["league", "season", f"{s}_team_api_id"]].set_axis(["league", "season", "team"], axis=1)
                      for s in ["home", "away"]])
    season_teams = long.groupby(["league", "season"])["team"].agg(set).to_dict()
    snaps = team_attributes.sort_values("date")
    snapshots = defaultdict(list)                 # team -> [(snapshot date, z-scored tactic vector)]
    for team, date, vec in zip(snaps["team_api_id"], snaps["date"], scaler.transform(snaps[TACTICS])):
        snapshots[team].append((date, vec))

    print("walking through matches in date order to build graph features")
    graph = build(matches, season_teams, snapshots)

    tactic_cols = [f"{s}_{t}" for s in ["home", "away"] for t in TACTICS]
    feats = matches[["match_api_id", "date", "league", "season", "split", "result",
                     "home_style", "away_style", *tactic_cols]].merge(graph, on="match_api_id", how="left")
    for t in TACTICS:
        feats[f"diff_{t}"] = feats[f"home_{t}"] - feats[f"away_{t}"]
    feats["pagerank_diff"] = feats["pagerank_home"] - feats["pagerank_away"]
    feats["weakrank_diff"] = feats["weakrank_home"] - feats["weakrank_away"]
    feats["style_rank_diff"] = feats["style_rank_home"] - feats["style_rank_away"]
    feats["similar_ppg_diff"] = feats["home_ppg_vs_similar_to_away"] - feats["away_ppg_vs_similar_to_home"]
    feats = feats.merge(formation_features(pd.read_parquet(DATA / "lineups.parquet")), on="match_api_id", how="left")

    feats.to_parquet(HERE / "features.parquet", index=False)
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=TACTICS).rename_axis("style").reset_index()
    centroids["n_snapshots"] = np.bincount(kmeans.labels_, minlength=len(centroids))
    centroids.to_parquet(HERE / "styles.parquet", index=False)
    print(f"  features.parquet {feats.shape[0]:,} rows x {feats.shape[1]} cols")
    print("  share of matches with a style on both sides:",
          round(feats[["home_style", "away_style"]].notna().all(axis=1).mean(), 3))


if __name__ == "__main__":
    main()
