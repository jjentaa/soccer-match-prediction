"""Model 3, step 2 - gradient boosting on the style graph features.

Trains on the shared 70 % split, evaluates on the 30 % test split, and writes:
  predictions/model3_test.parquet        p_H, p_D, p_A for every test match
  predictions/model3_train_oof.parquet   5-fold out-of-fold probabilities for every training match (for stacking)
  metrics.json                           accuracy / macro F1 of the model, its ablations and the baselines
  figures/style_profiles.png, figures/style_matchups.png

Usage: python model3/train.py        (run build_features.py first)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "datasets"
SEED = 42
CLASSES = ["H", "D", "A"]

TACTICS = ["buildUpPlaySpeed", "buildUpPlayPassing", "chanceCreationPassing", "chanceCreationCrossing",
           "chanceCreationShooting", "defencePressure", "defenceAggression", "defenceTeamWidth"]
FORMATION = [f"{s}_n_{line}" for s in ["home", "away"] for line in ["def", "dm", "mid", "am", "fwd"]]
STYLE = ["home_style", "away_style", *FORMATION,
         *[f"{p}_{t}" for p in ["home", "away", "diff"] for t in TACTICS]]
STYLE_GRAPHS = ["matchup_p_home", "matchup_p_draw", "matchup_p_away", "matchup_n",
                "style_rank_home", "style_rank_away", "style_rank_diff",
                "sim_degree_home", "sim_degree_away", "sim_clustering_home", "sim_clustering_away",
                "sim_density", "sim_are_neighbours", "style_distance",
                "home_ppg_vs_similar_to_away", "home_n_vs_similar_to_away",
                "away_ppg_vs_similar_to_home", "away_n_vs_similar_to_home", "similar_ppg_diff"]
RESULTS_GRAPH = ["pagerank_home", "pagerank_away", "pagerank_diff",
                 "weakrank_home", "weakrank_away", "weakrank_diff", "results_density"]
FEATURE_SETS = {
    "style only (no graph)": STYLE,
    "style + style graphs": STYLE + STYLE_GRAPHS,
    "style + style graphs + results graph": STYLE + STYLE_GRAPHS + RESULTS_GRAPH,
}
FINAL = "style + style graphs + results graph"


def make_model(columns: list[str], class_weight=None) -> HistGradientBoostingClassifier:
    # missing values (matches without a tactic snapshot or lineup) are handled natively by the trees
    return HistGradientBoostingClassifier(
        learning_rate=0.03, max_iter=500, max_depth=3, min_samples_leaf=40, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=30,
        categorical_features=[c in ("home_style", "away_style") for c in columns],
        class_weight=class_weight, random_state=SEED)


def score(y_true, proba, classes) -> dict:
    order = np.argsort(classes)                       # sklearn metrics expect lexicographic class order
    classes, proba = np.asarray(classes)[order], np.asarray(proba)[:, order]
    pred = classes[proba.argmax(axis=1)]
    return {"accuracy": round(accuracy_score(y_true, pred), 4),
            "macro_f1": round(f1_score(y_true, pred, average="macro"), 4),
            "log_loss": round(log_loss(y_true, proba, labels=list(classes)), 4)}


def save_predictions(ids, proba, classes, path: Path) -> None:
    out = pd.DataFrame(proba, columns=[f"p_{c}" for c in classes])
    out.insert(0, "match_api_id", ids.to_numpy())
    out[["match_api_id", *[f"p_{c}" for c in CLASSES]]].to_parquet(path, index=False)


# ---- figures -------------------------------------------------------------------------------
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("blue_red", ["#2a78d6", "#f0efec", "#e34948"])


def style_names(styles: pd.DataFrame) -> list[str]:
    """Label each style by its two most distinctive tactics, e.g. 'S0: +defencePressure, +chanceCreationPassing'."""
    names = []
    for row in styles.itertuples():
        z = pd.Series({t: getattr(row, t) for t in TACTICS})
        top = z.abs().sort_values(ascending=False).index[:2]
        names.append(f"S{row.style}: " + ", ".join(f"{'+' if z[t] > 0 else '−'}{t}" for t in top))
    return names


def heatmap(ax, values, row_labels, col_labels, fmt, norm):
    ax.imshow(values, cmap=DIVERGING, norm=norm, aspect="auto")
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if not np.isnan(values[i, j]):
                ax.text(j, i, fmt(values[i, j]), ha="center", va="center", color=INK, fontsize=9)
    # 2px surface gaps between cells
    ax.set_xticks(np.arange(-0.5, values.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, values.shape[0]), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0, colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_style_profiles(styles: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 3.2), facecolor=SURFACE)
    labels = [f"S{s}  (n={n})" for s, n in zip(styles["style"], styles["n_snapshots"])]
    heatmap(ax, styles[TACTICS].to_numpy(), labels, TACTICS, lambda v: f"{v:+.1f}", TwoSlopeNorm(0, -1.2, 1.2))
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax.set_title("Style profiles: cluster centre of each tactic, in standard deviations from the average team\n"
                 "red = more than average, blue = less", loc="left", fontsize=10, color=INK)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_style_matchups(train: pd.DataFrame, k: int, path: Path) -> None:
    played = train.dropna(subset=["home_style", "away_style"])
    points = played["result"].map({"H": 3, "D": 1, "A": 0})
    table = points.groupby([played["home_style"].astype(int), played["away_style"].astype(int)]).mean().unstack()
    table = table.reindex(index=range(k), columns=range(k))
    centre = points.mean()
    fig, ax = plt.subplots(figsize=(5.6, 3.6), facecolor=SURFACE)
    heatmap(ax, table.to_numpy(), [f"S{i}" for i in range(k)], [f"S{i}" for i in range(k)],
            lambda v: f"{v:.2f}", TwoSlopeNorm(centre, centre - 0.45, centre + 0.45))
    ax.set_xlabel("away team style", color=MUTED, fontsize=9)
    ax.set_ylabel("home team style", color=MUTED, fontsize=9)
    ax.set_title(f"Home points per game by style matchup (training matches)\n"
                 f"red = home side does better than the {centre:.2f} average, blue = worse",
                 loc="left", fontsize=10, color=INK)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    feats = pd.read_parquet(HERE / "features.parquet")
    train, test = feats[feats["split"] == "train"], feats[feats["split"] == "test"]
    (HERE / "predictions").mkdir(exist_ok=True)
    (HERE / "figures").mkdir(exist_ok=True)
    metrics = {}

    # baselines
    prior = train["result"].value_counts(normalize=True).reindex(CLASSES).to_numpy()
    metrics["baseline: always home win"] = score(test["result"], np.tile(prior, (len(test), 1)), CLASSES)
    odds = pd.read_parquet(DATA / "matches.parquet")[["match_api_id", "B365H", "B365D", "B365A"]]
    odds = test[["match_api_id", "result"]].merge(odds, on="match_api_id").dropna()
    implied = 1 / odds[["B365H", "B365D", "B365A"]].to_numpy()
    metrics[f"reference: Bet365 odds ({len(odds):,} test matches with odds)"] = score(
        odds["result"], implied / implied.sum(axis=1, keepdims=True), CLASSES)

    # ablation: what do the graphs add on top of raw style?
    for name, cols in FEATURE_SETS.items():
        model = make_model(cols).fit(train[cols], train["result"])
        metrics[f"model 3: {name}"] = score(test["result"], model.predict_proba(test[cols]), model.classes_)
    cols = FEATURE_SETS[FINAL]
    balanced = make_model(cols, class_weight="balanced").fit(train[cols], train["result"])
    metrics[f"model 3: {FINAL}, class_weight=balanced"] = score(
        test["result"], balanced.predict_proba(test[cols]), balanced.classes_)

    for name, m in metrics.items():
        print(f"  {name:<62s} acc {m['accuracy']:.3f}   macro-F1 {m['macro_f1']:.3f}   log-loss {m['log_loss']:.3f}")
    (HERE / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # predictions come from the unweighted final model: its probabilities are calibrated, which the
    # ensemble needs; class re-weighting can be applied once, at the ensemble stage
    final = make_model(cols).fit(train[cols], train["result"])
    save_predictions(test["match_api_id"], final.predict_proba(test[cols]), final.classes_,
                     HERE / "predictions" / "model3_test.parquet")
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(make_model(cols), train[cols], train["result"], cv=folds, method="predict_proba")
    save_predictions(train["match_api_id"], oof, sorted(CLASSES), HERE / "predictions" / "model3_train_oof.parquet")

    styles = pd.read_parquet(HERE / "styles.parquet")
    print("  styles:", *style_names(styles), sep="\n    ")
    plot_style_profiles(styles, HERE / "figures" / "style_profiles.png")
    plot_style_matchups(train, len(styles), HERE / "figures" / "style_matchups.png")


if __name__ == "__main__":
    main()
