"""Model 1, step 2 - tune, compare and score the team-rating model.

  1. Tune the time-decay half-life on the training split (log loss, first season left out)
  2. Ablation: baselines, plain Poisson, + time decay, + Dixon-Coles rho (final), + draw weight
  3. Tune the draw weight on the training split (macro F1)
  4. Score every row of the ablation once on the test split
  5. Save predictions for the ensemble, team ratings over time, metrics and figures

Training rows are predicted only from training matches dated before them, so their scores are
out-of-sample and can be used for tuning; the test split is only scored at the end.

Usage: python model1_team_rating/train.py     (~6 min)
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss

from rating import DATA, HERE, load_matches, rolling_predictions

HALF_LIVES = [90, 180, 365, 730, None]          # days; None = no time decay
DRAW_WEIGHTS = np.round(np.arange(1.0, 2.01, 0.05), 2)
CLASSES = ["H", "D", "A"]
PROBS = ["p_H", "p_D", "p_A"]
FIRST_SEASON = "2008/2009"                       # no history yet -> left out of tuning
TOP_TEAMS = ["FC Barcelona", "Real Madrid CF", "FC Bayern Munich", "Manchester City",
             "Chelsea", "Juventus", "Paris Saint-Germain"]

OUT = HERE / "outputs"
FIG = HERE / "figures"


# ---------------------------------------------------------------- helpers

def scores(y, P) -> dict:
    """Accuracy, macro F1 and log loss of probabilities P (columns H, D, A) against labels y."""
    pred = np.array(CLASSES)[P.argmax(axis=1)]
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "macro_f1": round(f1_score(y, pred, average="macro"), 4),
        # log_loss expects columns in sorted label order (A, D, H), ours are H, D, A
        "log_loss": round(log_loss(y, P[:, ::-1], labels=sorted(CLASSES)), 4),
    }


def apply_draw_weight(P, weight):
    """Multiply P(draw) by `weight` and renormalise, so draws are predicted more often."""
    P = P.copy()
    P[:, 1] *= weight
    return P / P.sum(axis=1, keepdims=True)


def bet365_probs(match_ids) -> pd.DataFrame:
    """Bookmaker reference: 1 / decimal odds, normalised to remove the margin."""
    odds = pd.read_parquet(DATA / "matches.parquet", columns=["match_api_id", "B365H", "B365D", "B365A"])
    odds = odds[odds["match_api_id"].isin(match_ids)].dropna()
    inv = 1 / odds[["B365H", "B365D", "B365A"]].to_numpy()
    odds[PROBS] = inv / inv.sum(axis=1, keepdims=True)
    return odds[["match_api_id", *PROBS]]


def tune_half_life(matches) -> tuple:
    """Run the final model (with rho) for every half-life; pick the lowest training log loss."""
    rows = []
    for half_life in HALF_LIVES:
        print(f"   half-life {half_life} days ...", flush=True)
        preds = rolling_predictions(matches, half_life=half_life, use_rho=True)
        tr = preds[(preds["split"] == "train") & (preds["season"] != FIRST_SEASON)]
        rows.append({"half_life": half_life or "none", **scores(tr["result"], tr[PROBS].to_numpy())})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    best = HALF_LIVES[table["log_loss"].idxmin()]
    return best, table


def tune_draw_weight(train_preds) -> tuple[float, pd.DataFrame]:
    """Pick the draw weight with the best macro F1 on the training rows."""
    y, P = train_preds["result"], train_preds[PROBS].to_numpy()
    table = pd.DataFrame([{"draw_weight": w, **scores(y, apply_draw_weight(P, w))} for w in DRAW_WEIGHTS])
    best = float(table.loc[table["macro_f1"].idxmax(), "draw_weight"])
    return best, table


# ---------------------------------------------------------------- figures

def plot_tuning(half_life_table, draw_table, best_half_life, best_weight):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(half_life_table["half_life"].astype(str), half_life_table["log_loss"], color="tab:blue")
    axes[0].set_ylim(half_life_table["log_loss"].min() - 0.005, half_life_table["log_loss"].max() + 0.005)
    axes[0].set(title=f"Time decay (chosen: {best_half_life} days)", xlabel="half-life (days)",
                ylabel="training log loss")
    axes[1].plot(draw_table["draw_weight"], draw_table["macro_f1"], marker="o", label="macro F1")
    axes[1].plot(draw_table["draw_weight"], draw_table["accuracy"], marker="o", label="accuracy")
    axes[1].axvline(best_weight, color="grey", linestyle="--")
    axes[1].set(title=f"Draw weight (chosen: {best_weight})", xlabel="weight on P(draw)",
                ylabel="training score")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG / "tuning.png", dpi=150)
    plt.close(fig)


def plot_ratings(ratings):
    teams = pd.read_parquet(DATA / "teams.parquet", columns=["team_api_id", "team_long_name"])
    top = ratings.merge(teams, on="team_api_id")
    # the first season has little history, so its ratings swing wildly; start from the second
    top = top[top["team_long_name"].isin(TOP_TEAMS) & (top["date"] >= "2009-07-01")]
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, rows in top.groupby("team_long_name"):
        ax.plot(rows["date"], rows["rating"], label=name, linewidth=1.2)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set(title="Team rating (attack + defence) over time; 0 = league average",
           xlabel="date", ylabel="rating")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "ratings_top_teams.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    matches = load_matches()
    train = matches[matches["split"] == "train"]
    print(f"matches: {len(matches)}  train: {len(train)}  test: {len(matches) - len(train)}")

    print("1. tune half-life (training split)")
    best_half_life, half_life_table = tune_half_life(matches)
    print(f"   -> half-life = {best_half_life} days")

    print("2. ablation runs")
    no_decay = rolling_predictions(matches, half_life=None, use_rho=False)
    decay = rolling_predictions(matches, half_life=best_half_life, use_rho=False)
    final, ratings = rolling_predictions(matches, half_life=best_half_life, use_rho=True, keep_ratings=True)
    rho = final.groupby("league")["rho"].mean()

    print("3. tune draw weight (training split)")
    final_train = final[final["split"] == "train"]
    best_weight, draw_table = tune_draw_weight(final_train)
    print(f"   -> draw weight = {best_weight}")

    print("4. score on the test split")
    test_ids = matches.loc[matches["split"] == "test", "match_api_id"]

    def on_test(preds):
        return preds.set_index("match_api_id").loc[test_ids]

    y_test = on_test(final)["result"]
    # training class rates: argmax is always H, so this is "always home win" with a usable log loss
    train_rates = train["result"].value_counts(normalize=True)[CLASSES].to_numpy()
    ablation = {
        "baseline: always home win (training class rates)": np.tile(train_rates, (len(y_test), 1)),
        "Poisson, no time decay": on_test(no_decay)[PROBS].to_numpy(),
        f"Poisson + time decay (half-life {best_half_life or 'none'})": on_test(decay)[PROBS].to_numpy(),
        "Dixon-Coles + time decay (final)": on_test(final)[PROBS].to_numpy(),
        f"final + draw weight {best_weight}": apply_draw_weight(on_test(final)[PROBS].to_numpy(), best_weight),
    }
    test_scores = {name: scores(y_test, P) for name, P in ablation.items()}

    bet = bet365_probs(test_ids)
    y_bet = matches.set_index("match_api_id").loc[bet["match_api_id"], "result"]
    test_scores[f"reference: Bet365 odds ({len(bet)} test matches with odds)"] = \
        scores(y_bet, bet[PROBS].to_numpy())
    print(pd.DataFrame(test_scores).T.to_string())

    P_weighted = ablation[f"final + draw weight {best_weight}"]
    pred_weighted = np.array(CLASSES)[P_weighted.argmax(axis=1)]
    f1_per_class = f1_score(y_test, pred_weighted, labels=CLASSES, average=None)

    print("5. save outputs")
    metrics = {
        "split": "datasets/split.parquet (random 70:30, random_state=42)",
        "half_life_days": best_half_life,
        "draw_weight": best_weight,
        "mean_rho_by_league": rho.round(4).to_dict(),
        "test": test_scores,
        "test_f1_per_class_draw_weighted": dict(zip(CLASSES, np.round(f1_per_class, 4).tolist())),
        "test_confusion_draw_weighted_rows_true_cols_pred_HDA":
            confusion_matrix(y_test, pred_weighted, labels=CLASSES).tolist(),
        "test_rows_without_rating": int((~on_test(final)["has_rating"]).sum()),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    half_life_table.to_csv(OUT / "tuning_half_life.csv", index=False)
    draw_table.to_csv(OUT / "tuning_draw_weight.csv", index=False)

    final["set"] = final["split"]
    final[["match_api_id", "set", *PROBS, "lambda_home", "lambda_away", "has_rating"]] \
        .to_parquet(OUT / "pred_model1.parquet", index=False)
    ratings.to_parquet(OUT / "ratings.parquet", index=False)

    plot_tuning(half_life_table, draw_table, best_half_life, best_weight)
    plot_ratings(ratings)
    print(f"   -> {OUT.relative_to(HERE.parent)}/, {FIG.relative_to(HERE.parent)}/")


if __name__ == "__main__":
    main()
