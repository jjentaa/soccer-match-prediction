"""Model 2 EDA on the training split: how lineup ratings relate to match results.

Figures go to model2_lineup/figures/, a text summary to model2_lineup/outputs/eda_summary.md.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

import plot_style as ps
from features import DATA, GK_ATTRS, OUTFIELD_ATTRS, build_dataset, prepare_players
from train import OUT

LINE_ORDER = ["GK", "DEF", "MID", "FWD"]


def rating_diff_vs_result(train, plt):
    bins = [-np.inf, -8, -6, -4, -2, 0, 2, 4, 6, 8, np.inf]
    labels = ["< -8", "-8..-6", "-6..-4", "-4..-2", "-2..0", "0..2", "2..4", "4..6", "6..8", "> 8"]
    d = train.dropna(subset=["diff_ovr_mean"])
    d = d.assign(bin=pd.cut(d["diff_ovr_mean"], bins=bins, labels=labels))
    rates = pd.crosstab(d["bin"], d["result"], normalize="index")[["H", "D", "A"]] * 100
    counts = d["bin"].value_counts().reindex(labels)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    for r in ["H", "D", "A"]:
        ax.plot(x, rates[r].values, marker="o", markersize=6, color=ps.RESULT_COLOR[r], label=ps.RESULT_NAME[r])
        ax.annotate(ps.RESULT_NAME[r], (x[-1], rates[r].values[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", color=ps.TEXT_2)
    ax.set_xticks(x, [f"{lab}\nn={n:,}" for lab, n in zip(labels, counts)], fontsize=8)
    ax.set_xlabel("Home minus away average overall rating of the starting XI")
    ax.set_ylabel("% of matches")
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.5, len(labels) + 0.8)
    ax.legend(loc="upper left", ncol=3)
    ax.set_title("A stronger starting XI wins more often; the draw rate barely moves")
    ps.save(fig, "eda_rating_diff_vs_result.png")
    return rates.round(1)


def rating_by_line(players_tr, plt):
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [players_tr.loc[players_tr["line"] == ln, "overall_rating"].dropna() for ln in LINE_ORDER]
    bp = ax.boxplot(data, tick_labels=LINE_ORDER, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops={"color": ps.TEXT, "linewidth": 1.5})
    for box in bp["boxes"]:
        box.set(facecolor="#b7d3f6", edgecolor=ps.ACCENT)
    for part in ["whiskers", "caps"]:
        for line in bp[part]:
            line.set_color(ps.ACCENT)
    ax.set_ylabel("FIFA overall rating")
    ax.set_title("Overall rating of starters by line (whiskers 1.5 IQR)")
    ps.save(fig, "eda_rating_by_line.png")
    return players_tr.groupby("line")["overall_rating"].describe().loc[LINE_ORDER].round(1)


def attribute_correlation(players_tr, plt):
    attrs = OUTFIELD_ATTRS + GK_ATTRS
    corr = players_tr[attrs].corr()
    order = leaves_list(linkage(squareform(1 - corr.abs().values, checks=False), method="average"))
    corr = corr.iloc[order, order]

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr.values, cmap=ps.DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Pearson correlation")
    ax.set_title("Player attributes are strongly correlated (clustered order)")
    ps.save(fig, "eda_attribute_correlation.png")

    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    return {
        "mean_abs_corr": round(float(upper.abs().mean()), 3),
        "pairs_abs_corr_over_0.7": int((upper.abs() > 0.7).sum()),
        "n_pairs": int(len(upper)),
    }


def missing_summary(ds, train):
    feat_cols = [c for c in ds.columns if c.startswith(("home_", "away_", "diff_"))]
    no_lineup = ds["home_n_players"].isna() & ds["away_n_players"].isna()
    full = (ds["home_n_players"] == 11) & (ds["away_n_players"] == 11)
    miss = train[feat_cols].isna().mean().sort_values(ascending=False)
    return {
        "matches": len(ds),
        "matches_without_lineup": int(no_lineup.sum()),
        "matches_with_full_22": int(full.sum()),
        "matches_partial_lineup": int((~no_lineup & ~full).sum()),
        "features": len(feat_cols),
        "train_features_with_any_missing": int((miss > 0).sum()),
        "train_worst_missing": miss.head(5).round(3).to_dict(),
    }


def main():
    ps.setup()
    import matplotlib.pyplot as plt

    ds = build_dataset()
    train = ds[ds["set"] == "train"]
    lineups = pd.read_parquet(DATA / "lineups.parquet")
    players = pd.read_parquet(DATA / "players.parquet")
    players_tr = prepare_players(lineups[lineups["match_api_id"].isin(train["match_api_id"])], players)

    sections = {
        "Class balance (train, %)": (train["result"].value_counts(normalize=True) * 100).round(1),
        "Result rate (%) by rating difference": rating_diff_vs_result(train, plt),
        "Overall rating by line": rating_by_line(players_tr, plt),
        "Attribute correlation": attribute_correlation(players_tr, plt),
        "Missing values": missing_summary(ds, train),
    }

    OUT.mkdir(exist_ok=True)
    lines = ["# Model 2 EDA summary (training split)\n"]
    for title, value in sections.items():
        body = f"```\n{value.to_string()}\n```" if isinstance(value, (pd.DataFrame, pd.Series)) else \
            "\n".join(f"- {k}: {v}" for k, v in value.items())
        lines += [f"## {title}\n", body, ""]
    (OUT / "eda_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
