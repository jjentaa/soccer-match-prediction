"""Model 3 EDA: what do the tactic data and the graphs built from them look like?

Anything that uses match results is computed on the training split only.
Run build_features.py first.

Usage: python model3/eda.py   -> figures/eda_*.png, eda_summary.md
"""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from build_features import DATA, KNN, TACTICS, similarity_graph  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"

SURFACE, INK, MUTED, GRID, GREY = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # categorical slots 1-4, fixed order
RESULT_COLOR = {"H": SERIES[0], "D": SERIES[1], "A": SERIES[2]}
RESULT_NAME = {"H": "Home win", "D": "Draw", "A": "Away win"}
DIVERGING = LinearSegmentedColormap.from_list("blue_red", ["#2a78d6", "#f0efec", "#e34948"])
SHORT = {"buildUpPlaySpeed": "build-up speed", "buildUpPlayPassing": "build-up passing",
         "chanceCreationPassing": "chance passing", "chanceCreationCrossing": "chance crossing",
         "chanceCreationShooting": "chance shooting", "defencePressure": "def. pressure",
         "defenceAggression": "def. aggression", "defenceTeamWidth": "def. width"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "legend.frameon": False, "lines.linewidth": 2, "font.size": 9.5, "savefig.dpi": 160, "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("saved", FIG / name)


def tactic_correlation(z: pd.DataFrame) -> pd.DataFrame:
    corr = z.corr()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.imshow(corr, cmap=DIVERGING, norm=TwoSlopeNorm(0, -1, 1))
    labels = [SHORT[t] for t in TACTICS]
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color=INK, fontsize=8.5)
    ax.set_xticks(np.arange(-0.5, len(labels)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels)), minor=True)
    ax.grid(False)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Tactics are only loosely correlated; the defence settings are the tightest group")
    save(fig, "eda_tactic_correlation.png")
    return corr


def style_map(z: pd.DataFrame, styles: np.ndarray):
    """Tactic snapshots projected on their first two principal components, one panel per style."""
    pca = PCA(n_components=2, random_state=42).fit(z)
    xy = pca.transform(z)
    k = int(styles.max()) + 1
    fig, axes = plt.subplots(1, k, figsize=(3.1 * k, 3.3), sharex=True, sharey=True)
    for s, ax in enumerate(axes):
        ax.scatter(*xy[styles != s].T, s=7, color=GREY, alpha=0.5, linewidths=0)
        ax.scatter(*xy[styles == s].T, s=9, color=SERIES[0], alpha=0.8, linewidths=0)
        ax.set_title(f"S{s}  ({(styles == s).sum()} snapshots)", fontsize=10)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} of variance)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    fig.suptitle("Styles are regions of one continuous cloud, not separate islands", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", y=1.04)
    save(fig, "eda_style_map.png")
    return pca


def tactics_vs_points(train: pd.DataFrame) -> pd.Series:
    """Correlation between a team's tactic setting and the points it took, over training matches."""
    points = train["result"].map({"H": 3, "D": 1, "A": 0})
    long = pd.concat([
        train[[f"home_{t}" for t in TACTICS]].set_axis(TACTICS, axis=1).assign(points=points),
        train[[f"away_{t}" for t in TACTICS]].set_axis(TACTICS, axis=1).assign(
            points=train["result"].map({"H": 0, "D": 1, "A": 3})),
    ]).dropna()
    corr = long[TACTICS].corrwith(long["points"]).sort_values()
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.barh([SHORT[t] for t in corr.index], corr.values, color=SERIES[0], height=0.62)
    for i, v in enumerate(corr.values):
        ax.text(v + (0.003 if v >= 0 else -0.003), i, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", color=INK, fontsize=9)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_xlim(corr.min() - 0.03, corr.max() + 0.03)
    ax.set_xlabel("Correlation with points won in the match")
    ax.grid(axis="y", visible=False)
    ax.set_title("Single tactics relate only weakly to results (|r| ≤ 0.07): pressing helps, long balls hurt")
    save(fig, "eda_tactics_vs_points.png")
    return corr


def rank_gap_vs_result(train: pd.DataFrame) -> pd.DataFrame:
    bins = [-np.inf, -0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9, np.inf]
    labels = ["< −0.9", "−0.9..−0.6", "−0.6..−0.3", "−0.3..0", "0..0.3", "0.3..0.6", "0.6..0.9", "> 0.9"]
    d = train.dropna(subset=["pagerank_diff"])
    d = d.assign(bin=pd.cut(d["pagerank_diff"], bins=bins, labels=labels))
    rates = pd.crosstab(d["bin"], d["result"], normalize="index")[["H", "D", "A"]] * 100
    counts = d["bin"].value_counts().reindex(labels)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(labels))
    for r in ["H", "D", "A"]:
        ax.plot(x, rates[r].values, marker="o", markersize=6, color=RESULT_COLOR[r], label=RESULT_NAME[r])
        ax.annotate(RESULT_NAME[r], (x[-1], rates[r].values[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=MUTED)
    ax.set_xticks(x, [f"{lab}\nn={n:,}" for lab, n in zip(labels, counts)], fontsize=8)
    ax.set_xlabel("Home minus away PageRank in the results graph (1.0 = league average)")
    ax.set_ylabel("% of matches")
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.5, len(labels) + 0.6)
    ax.legend(loc="upper left", ncol=3)
    ax.set_title("PageRank on the loser → winner graph orders teams by strength")
    save(fig, "eda_pagerank_diff_vs_result.png")
    return rates.round(1)


def league_similarity_graph(snapshots: pd.DataFrame, matches: pd.DataFrame, teams: pd.DataFrame,
                            league: str, season: str) -> dict:
    """Draw one league-season's style similarity graph."""
    season_matches = matches[(matches["league"] == league) & (matches["season"] == season)]
    ids = sorted(set(season_matches["home_team_api_id"]) | set(season_matches["away_team_api_id"]))
    as_of = season_matches["date"].max()
    latest = snapshots[snapshots["date"] <= as_of].sort_values("date").groupby("team_api_id").tail(1)
    latest = latest[latest["team_api_id"].isin(ids)].set_index("team_api_id")
    vectors = {t: latest.loc[t, TACTICS].to_numpy(dtype=float) for t in latest.index}
    g = similarity_graph(vectors)
    pos = nx.kamada_kawai_layout(g)
    names = teams.set_index("team_api_id")["team_short_name"]

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=GREY, width=1.2)
    for s in sorted(latest["style"].unique()):
        nodes = [t for t in g if latest.loc[t, "style"] == s]
        nx.draw_networkx_nodes(g, pos, nodelist=nodes, node_color=SERIES[int(s)], node_size=210, ax=ax,
                               edgecolors=SURFACE, linewidths=2, label=f"S{int(s)}")
    for t, (x, y) in pos.items():
        ax.annotate(f"{names[t]} · S{int(latest.loc[t, 'style'])}", (x, y), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.legend(title="style", loc="upper left", ncol=4, handletextpad=0.2, columnspacing=1.0)
    ax.set_axis_off()
    ax.set_title(f"Style similarity graph, {league} {season}: each team is linked to the {KNN} teams "
                 "that play most like it")
    save(fig, "eda_similarity_graph.png")
    degree = dict(g.degree())
    return {"teams": g.number_of_nodes(), "edges": g.number_of_edges(), "density": round(nx.density(g), 3),
            "average_clustering": round(nx.average_clustering(g), 3),
            "most_connected": ", ".join(f"{teams.set_index('team_api_id')['team_long_name'][t]} ({d})"
                                        for t, d in sorted(degree.items(), key=lambda kv: -kv[1])[:3])}


def main() -> None:
    FIG.mkdir(exist_ok=True)
    fitted = joblib.load(HERE / "styles.joblib")
    snapshots = pd.read_parquet(DATA / "team_attributes.parquet")
    raw = snapshots[TACTICS].copy()
    snapshots[TACTICS] = fitted["scaler"].transform(snapshots[TACTICS])
    snapshots["style"] = fitted["kmeans"].predict(snapshots[TACTICS].to_numpy())
    feats = pd.read_parquet(HERE / "features.parquet")
    train = feats[feats["split"] == "train"]
    matches = pd.read_parquet(DATA / "matches.parquet")
    teams = pd.read_parquet(DATA / "teams.parquet")

    corr = tactic_correlation(snapshots[TACTICS])
    pca = style_map(snapshots[TACTICS], snapshots["style"].to_numpy())
    points_corr = tactics_vs_points(train)
    rates = rank_gap_vs_result(train)
    graph = league_similarity_graph(snapshots, matches, teams, "England Premier League", "2015/2016")

    # how stable is a team's style from one snapshot to the next?
    ordered = snapshots.sort_values(["team_api_id", "date"])
    same_team = ordered["team_api_id"].eq(ordered["team_api_id"].shift())
    kept = (ordered["style"].eq(ordered["style"].shift()) & same_team).sum() / same_team.sum()
    coverage = feats.assign(has=feats[["home_style", "away_style"]].notna().all(axis=1)).groupby("season")["has"].mean()
    off_diag = corr.where(~np.eye(len(TACTICS), dtype=bool)).abs()

    lines = [
        "# Model 3 EDA summary\n",
        "## Tactic data\n",
        f"- snapshots: {len(snapshots):,} for {snapshots['team_api_id'].nunique()} teams, "
        f"{snapshots['date'].dt.year.min()}-{snapshots['date'].dt.year.max()}, one per team per year",
        f"- raw scale: {raw.min().min():.0f}-{raw.max().max():.0f}, mean {raw.mean().mean():.1f}, "
        f"std {raw.std().mean():.1f}",
        f"- mean |correlation| between tactics: {np.nanmean(off_diag.to_numpy()):.2f}; "
        f"largest: {off_diag.stack().idxmax()[0]} & {off_diag.stack().idxmax()[1]} = {np.nanmax(off_diag.to_numpy()):.2f}",
        f"- first two principal components explain {pca.explained_variance_ratio_[:2].sum():.0%} of the variance",
        f"- a team keeps its style from one snapshot to the next {kept:.0%} of the time\n",
        "## Share of matches with a style on both sides, by season\n",
        "```", coverage.round(3).to_string(), "```\n",
        "## Correlation of a team's tactic with points won (training matches)\n",
        "```", points_corr.round(3).to_string(), "```\n",
        "## Result rate (%) by PageRank gap (training matches)\n",
        "```", rates.to_string(), "```\n",
        "## Style similarity graph, England Premier League 2015/2016\n",
        *[f"- {k}: {v}" for k, v in graph.items()],
    ]
    (HERE / "eda_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
