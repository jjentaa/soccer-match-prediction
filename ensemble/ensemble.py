"""Combine Model 1 (team rating), Model 2 (lineup) and Model 3 (style graph) into one prediction.

Every model gives out-of-sample p_H, p_D, p_A for all 25,979 matches. Five ways of combining them are
compared by 5-fold cross-validation on the training split; the best one by CV log loss is refit on the
whole training split and scored once on the test split.

  average          mean of the three probability vectors (soft voting)
  geometric mean   mean of the log-probabilities, renormalised
  weighted average weights on the simplex, minimising log loss
  stacking         multinomial logistic regression on the nine log-probabilities
  majority vote    each model votes for its top class (ties -> the average's top class)

Usage: python ensemble/ensemble.py   -> predictions/ensemble_test.parquet, metrics.json, figures/
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEED = 42
CLASSES = np.array(["H", "D", "A"])
P = ["p_H", "p_D", "p_A"]
MODELS = ["model 1: team rating", "model 2: lineup", "model 3: style graph"]


def reweight_draw(proba: np.ndarray, w: float) -> np.ndarray:
    out = proba.copy()
    out[..., 1] *= w
    return out / out.sum(axis=-1, keepdims=True)


def load() -> tuple[pd.DataFrame, np.ndarray]:
    """One row per match, plus an array [match, model, class] of probabilities."""
    base = pd.read_parquet(ROOT / "datasets" / "split.parquet").merge(
        pd.read_parquet(ROOT / "datasets" / "matches.parquet")[["match_api_id", "result", "B365H", "B365D", "B365A"]])
    m1 = pd.read_parquet(ROOT / "model1_team_rating" / "outputs" / "pred_model1.parquet")
    m2 = pd.read_parquet(ROOT / "model2_lineup" / "outputs" / "pred_model2.parquet")
    m3 = pd.concat([pd.read_parquet(ROOT / "model3" / "predictions" / f"model3_{part}.parquet")
                    for part in ["train_oof", "test"]])
    # Model 2 saved its probabilities with the draw weight applied; undo it so all three are calibrated
    w2 = json.loads((ROOT / "model2_lineup" / "outputs" / "metrics.json").read_text())["draw_weight"]
    m2[P] = reweight_draw(m2[P].to_numpy(), 1 / w2)

    stack = []
    for preds in [m1, m2, m3]:
        aligned = base[["match_api_id"]].merge(preds[["match_api_id", *P]], on="match_api_id", how="left")
        assert aligned[P].notna().all().all(), "a model is missing predictions for some matches"
        stack.append(aligned[P].to_numpy())
    return base, np.stack(stack, axis=1)


# ---- combination rules: each returns a function  probs[match, model, class] -> proba[match, class] ----
def fit_average(x, y):
    return lambda x: x.mean(axis=1)


def fit_geometric(x, y):
    def predict(x):
        g = np.exp(np.log(x).mean(axis=1))
        return g / g.sum(axis=1, keepdims=True)
    return predict


def fit_weighted(x, y):
    target = (y[:, None] == CLASSES[None, :])

    def loss(theta):
        w = np.exp(theta) / np.exp(theta).sum()
        return -np.log((x * w[None, :, None]).sum(axis=1)[target]).mean()

    theta = minimize(loss, np.zeros(x.shape[1]), method="Nelder-Mead").x
    w = np.exp(theta) / np.exp(theta).sum()
    predict = lambda x: (x * w[None, :, None]).sum(axis=1)
    predict.weights = w
    return predict


def fit_stacking(x, y):
    meta = LogisticRegression(C=1.0, max_iter=2000).fit(np.log(x).reshape(len(x), -1), y)
    order = [list(meta.classes_).index(c) for c in CLASSES]
    return lambda x: meta.predict_proba(np.log(x).reshape(len(x), -1))[:, order]


def fit_vote(x, y):
    def predict(x):
        votes = np.stack([(x.argmax(axis=2) == c).mean(axis=1) for c in range(3)], axis=1)
        return votes + 1e-3 * x.mean(axis=1)      # the average breaks ties; not a calibrated probability
    return predict


METHODS = {"average": fit_average, "geometric mean": fit_geometric, "weighted average": fit_weighted,
           "stacking (logistic regression)": fit_stacking, "majority vote": fit_vote}


def score(y, proba) -> dict:
    proba = proba / proba.sum(axis=1, keepdims=True)
    pred = CLASSES[proba.argmax(axis=1)]
    order = np.argsort(CLASSES)                   # sklearn's log_loss expects lexicographic class order
    return {"accuracy": round(accuracy_score(y, pred), 4),
            "macro_f1": round(f1_score(y, pred, average="macro"), 4),
            "log_loss": round(log_loss(y, proba[:, order], labels=list(CLASSES[order])), 4)}


def cross_val_proba(fit, x, y) -> np.ndarray:
    oof = np.zeros((len(x), 3))
    for tr, va in StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(x, y):
        oof[va] = fit(x[tr], y[tr])(x[va])
    return oof


def tune_draw_weight(y, proba) -> float:
    grid = np.round(np.arange(1.0, 2.01, 0.05), 2)
    f1 = [f1_score(y, CLASSES[reweight_draw(proba, w).argmax(axis=1)], average="macro") for w in grid]
    return float(grid[int(np.argmax(f1))])


def plot_comparison(rows: dict, path: Path) -> None:
    ink, muted, surface, blue, grey = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6", "#8a8980"
    names = list(rows)
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.9), facecolor=surface, sharey=True)
    for ax, (metric, title) in zip(axes, [("accuracy", "Accuracy (higher is better)"),
                                          ("macro_f1", "Macro F1 (higher is better)"),
                                          ("log_loss", "Log loss (lower is better)")]):
        values = [rows[n][metric] for n in names]
        colors = [blue if n.startswith("ensemble") else grey for n in names]
        # dots, not bars: the differences are small and the scale does not start at zero
        span = max(values) - min(values)
        ax.hlines(range(len(names)), min(values) - span * 0.3, values, color="#e4e3df", linewidth=2)
        ax.scatter(values, range(len(names)), color=colors, s=90, zorder=3)
        ax.set_xlim(min(values) - span * 0.3, max(values) + span * 0.45)
        ax.set_yticks(range(len(names)), names)
        ax.set_ylim(len(names) - 0.5, -0.5)
        for i, v in enumerate(values):
            ax.text(v + span * 0.07, i, f"{v:.3f}", va="center", ha="left", color=ink, fontsize=9)
        ax.set_title(title, loc="left", fontsize=10, color=ink)
        ax.set_facecolor(surface)
        ax.tick_params(length=0, colors=muted, labelsize=9)
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(0.01, -0.04, "Test split, 7,794 matches. Each panel has its own scale.",
             color=muted, fontsize=8)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=surface)
    plt.close(fig)


def main() -> None:
    base, probs = load()
    y = base["result"].to_numpy()
    tr, te = (base["split"] == "train").to_numpy(), (base["split"] == "test").to_numpy()
    (HERE / "predictions").mkdir(exist_ok=True)
    (HERE / "figures").mkdir(exist_ok=True)

    print("single models")
    singles = {name: {"train": score(y[tr], probs[tr, i]), "test": score(y[te], probs[te, i])}
               for i, name in enumerate(MODELS)}
    for name, m in singles.items():
        print(f"  {name:<34s} test acc {m['test']['accuracy']:.3f}  macro-F1 {m['test']['macro_f1']:.3f}  "
              f"log-loss {m['test']['log_loss']:.3f}")
    corr = np.corrcoef(probs[tr, :, 0].T).round(2)
    print("  correlation of P(home win) between models (train):", corr[0, 1], corr[0, 2], corr[1, 2])

    print("combination rules")
    results, oofs = {}, {}
    for name, fit in METHODS.items():
        oofs[name] = cross_val_proba(fit, probs[tr], y[tr])
        results[name] = {"cv_train": score(y[tr], oofs[name]), "test": score(y[te], fit(probs[tr], y[tr])(probs[te]))}
        if name == "majority vote":               # votes are not probabilities, so log loss is undefined
            results[name]["cv_train"]["log_loss"] = results[name]["test"]["log_loss"] = None
        cv, test = results[name]["cv_train"], results[name]["test"]
        fmt = lambda v: "  –  " if v is None else f"{v:.3f}"
        print(f"  {name:<34s} CV   acc {cv['accuracy']:.3f}  macro-F1 {cv['macro_f1']:.3f}  log-loss {fmt(cv['log_loss'])}"
              f"   | test acc {test['accuracy']:.3f}  macro-F1 {test['macro_f1']:.3f}  log-loss {fmt(test['log_loss'])}")

    # choose on the training split only; majority vote has no probabilities, so it cannot win on log loss
    best = min((n for n in METHODS if n != "majority vote"), key=lambda n: results[n]["cv_train"]["log_loss"])
    final = METHODS[best](probs[tr], y[tr])
    proba_te = final(probs[te])
    w = tune_draw_weight(y[tr], oofs[best])
    weights = fit_weighted(probs[tr], y[tr]).weights
    print(f"chosen by CV log loss: {best};  draw weight tuned on training CV: {w}")
    print("  weighted-average weights:", dict(zip(MODELS, weights.round(3))))

    odds = base.loc[te, ["B365H", "B365D", "B365A"]].to_numpy()
    has_odds = ~np.isnan(odds).any(axis=1)
    metrics = {
        "single_models": singles,
        "combination_rules": results,
        "chosen": best,
        "weighted_average_weights": dict(zip(MODELS, weights.round(4).tolist())),
        "draw_weight": w,
        "test_final": score(y[te], proba_te),
        "test_final_draw_weighted": score(y[te], reweight_draw(proba_te, w)),
        "test_reference_bet365": score(y[te][has_odds], 1 / odds[has_odds]),
        "test_final_on_matches_with_odds": score(y[te][has_odds], proba_te[has_odds]),
    }
    (HERE / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("  final on test:              ", metrics["test_final"])
    print("  final + draw weight on test:", metrics["test_final_draw_weighted"])
    print("  Bet365 on test with odds:   ", metrics["test_reference_bet365"],
          "\n  final on the same matches:  ", metrics["test_final_on_matches_with_odds"])

    out = pd.DataFrame(proba_te, columns=P)
    out.insert(0, "match_api_id", base.loc[te, "match_api_id"].to_numpy())
    out["prediction"] = CLASSES[proba_te.argmax(axis=1)]
    out["prediction_draw_weighted"] = CLASSES[reweight_draw(proba_te, w).argmax(axis=1)]
    out.to_parquet(HERE / "predictions" / "ensemble_test.parquet", index=False)

    plot_comparison({**{n: m["test"] for n, m in singles.items()}, f"ensemble: {best}": metrics["test_final"]},
                    HERE / "figures" / "ensemble_vs_models.png")


if __name__ == "__main__":
    main()
