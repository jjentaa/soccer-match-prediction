"""Model 1 EDA on the training split: do goals behave like the Poisson model assumes?

  1. Goal counts vs a Poisson distribution with the same mean
  2. Low scores (0-0, 1-0, 0-1, 1-1) and draws: observed vs what the Poisson model expects
  3. Home advantage by league
  4. Result rates by pre-match team-rating difference

Only training matches are used. Figures 2 and 4 use the pre-match predictions and ratings
written by train.py, so run train.py first.

Usage: python model1_team_rating/eda.py     -> figures/eda_*.png, outputs/eda_summary.md
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import poisson

from rating import load_matches
from train import CLASSES, FIG, OUT

RESULT_COLOR = {"H": "tab:blue", "D": "tab:grey", "A": "tab:red"}
RESULT_NAME = {"H": "home win", "D": "draw", "A": "away win"}
LOW_SCORES = [(0, 0), (1, 0), (0, 1), (1, 1)]


def goal_distribution(train):
    """Share of matches with 0, 1, 2, ... goals for each side, next to Poisson with the same mean."""
    goals = np.arange(8)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    table = {}
    for ax, side in zip(axes, ["home", "away"]):
        g = train[f"{side}_team_goal"]
        observed = g.value_counts(normalize=True).reindex(goals, fill_value=0)
        expected = poisson.pmf(goals, g.mean())
        ax.bar(goals - 0.2, observed * 100, width=0.4, label="observed")
        ax.bar(goals + 0.2, expected * 100, width=0.4, label=f"Poisson (mean {g.mean():.2f})")
        ax.set(title=f"{side.capitalize()} goals", xlabel="goals in the match", xticks=goals)
        ax.legend()
        table[side] = pd.DataFrame({"observed %": observed * 100, "Poisson %": expected * 100})
    axes[0].set_ylabel("% of matches")
    fig.suptitle("Goal counts are close to Poisson")
    fig.tight_layout()
    fig.savefig(FIG / "eda_goal_distribution.png", dpi=150)
    plt.close(fig)
    return pd.concat(table, axis=1).round(1)


def low_scores(train_rated):
    """Observed share of each low score vs the share the plain Poisson model expects.

    Expected share = mean over matches of Pois(x; lambda) * Pois(y; mu), using each match's own
    pre-match expected goals - so it compares like with like.
    """
    lam, mu = train_rated["lambda_home"].to_numpy(), train_rated["lambda_away"].to_numpy()
    hg, ag = train_rated["home_team_goal"].to_numpy(), train_rated["away_team_goal"].to_numpy()
    rows = []
    for x, y in LOW_SCORES:
        rows.append({"score": f"{x}-{y}", "observed %": np.mean((hg == x) & (ag == y)) * 100,
                     "Poisson %": np.mean(poisson.pmf(x, lam) * poisson.pmf(y, mu)) * 100})
    goals = np.arange(11)
    p_draw = sum(poisson.pmf(g, lam) * poisson.pmf(g, mu) for g in goals)
    rows.append({"score": "all draws", "observed %": np.mean(hg == ag) * 100, "Poisson %": p_draw.mean() * 100})
    table = pd.DataFrame(rows).set_index("score")

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(table))
    ax.bar(x - 0.2, table["observed %"], width=0.4, label="observed")
    ax.bar(x + 0.2, table["Poisson %"], width=0.4, label="Poisson expects")
    ax.set_xticks(x, table.index)
    ax.set(ylabel="% of matches", title="Low scores and draws: observed vs independent Poisson")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "eda_low_scores.png", dpi=150)
    plt.close(fig)
    return table.round(2)


def home_advantage(train):
    """Result rates and average goals per league."""
    by_league = train.groupby("league")
    table = pd.crosstab(train["league"], train["result"], normalize="index")[CLASSES] * 100
    table["home goals"] = by_league["home_team_goal"].mean()
    table["away goals"] = by_league["away_team_goal"].mean()
    table = table.sort_values("H")

    fig, ax = plt.subplots(figsize=(9, 5))
    left = np.zeros(len(table))
    for r in CLASSES:
        ax.barh(table.index, table[r], left=left, color=RESULT_COLOR[r], label=RESULT_NAME[r])
        left += table[r].to_numpy()
    ax.set(xlabel="% of matches", title="Home teams win most often in every league")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout()
    fig.savefig(FIG / "eda_home_advantage.png", dpi=150)
    plt.close(fig)
    return table.round(2)


def rating_diff_vs_result(train, ratings):
    """Result rates by home rating minus away rating, both taken on match day (pre-match)."""
    r = ratings[["team_api_id", "date", "rating"]]
    d = (train.merge(r.rename(columns={"team_api_id": "home_team_api_id", "rating": "home_rating"}),
                     on=["home_team_api_id", "date"])
              .merge(r.rename(columns={"team_api_id": "away_team_api_id", "rating": "away_rating"}),
                     on=["away_team_api_id", "date"]))
    d["diff"] = d["home_rating"] - d["away_rating"]
    bins = [-np.inf, -1, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 1, np.inf]
    labels = ["< -1", "-1..-0.6", "-0.6..-0.3", "-0.3..-0.1", "-0.1..0.1", "0.1..0.3", "0.3..0.6", "0.6..1", "> 1"]
    d["bin"] = pd.cut(d["diff"], bins=bins, labels=labels)
    rates = pd.crosstab(d["bin"], d["result"], normalize="index")[CLASSES] * 100
    counts = d["bin"].value_counts().reindex(labels)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(labels))
    for r in CLASSES:
        ax.plot(x, rates[r], marker="o", color=RESULT_COLOR[r], label=RESULT_NAME[r])
    ax.set_xticks(x, [f"{lab}\nn={n:,}" for lab, n in zip(labels, counts)], fontsize=8)
    ax.set(xlabel="home rating minus away rating (pre-match)", ylabel="% of matches", ylim=(0, 100),
           title="The higher-rated team wins more often; draws peak when the teams are evenly matched")
    ax.legend(loc="upper left", ncol=3)
    fig.tight_layout()
    fig.savefig(FIG / "eda_rating_diff_vs_result.png", dpi=150)
    plt.close(fig)
    rates["matches"] = counts
    return rates.round(1)


def main():
    FIG.mkdir(exist_ok=True)
    matches = load_matches()
    train = matches[matches["split"] == "train"]
    preds = pd.read_parquet(OUT / "pred_model1.parquet")
    ratings = pd.read_parquet(OUT / "ratings.parquet")
    train_rated = train.merge(preds[preds["has_rating"]][["match_api_id", "lambda_home", "lambda_away"]],
                              on="match_api_id")

    sections = {
        "Class balance (train, %)": (train["result"].value_counts(normalize=True)[CLASSES] * 100).round(1),
        "Goal counts vs Poisson (%)": goal_distribution(train),
        "Low scores and draws vs Poisson (%)": low_scores(train_rated),
        "Home advantage by league": home_advantage(train),
        "Result rate (%) by pre-match rating difference": rating_diff_vs_result(train, ratings),
    }
    lines = ["# Model 1 EDA summary (training split)\n"]
    for title, table in sections.items():
        lines += [f"## {title}\n", "```", table.to_string(), "```\n"]
    (OUT / "eda_summary.md").write_text("\n".join(lines))
    for title, table in sections.items():
        print(f"\n== {title}\n{table.to_string()}")


if __name__ == "__main__":
    main()
