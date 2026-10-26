"""A closer look at the final ensemble on the test split: where is it right, where is it wrong?

  confusion matrices and per-class precision / recall / F1, with and without the draw weight
  calibration: when the ensemble says 60 %, is it right 60 % of the time?
  accuracy by league
  paired bootstrap: is the ensemble's gain over the best single model more than noise?

Run ensemble.py first.

Usage: python ensemble/evaluate.py   -> evaluation.json, figures/confusion_matrices.png, figures/calibration.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from ensemble import CLASSES, MODELS, ROOT, SEED, load

HERE = Path(__file__).resolve().parent
LABELS = ["H", "D", "A"]
NAMES = ["Home win", "Draw", "Away win"]
SURFACE, INK, MUTED, GRID, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#f0efec", "#cde2fb", "#6da7ec", "#2a78d6", "#104281"])
CONFIDENCE_BINS = [0, 0.4, 0.5, 0.6, 0.7, 1.0]


def per_class(y, pred) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
    return {label: {"precision": round(float(p), 3), "recall": round(float(r), 3), "f1": round(float(f), 3),
                    "matches": int(n)} for label, p, r, f, n in zip(LABELS, precision, recall, f1, support)}


def paired_bootstrap(y, ensemble_proba, single_probas: dict, n_boot: int = 2000) -> dict:
    """Ensemble minus each single model on the same resampled test matches: mean difference and 95 % interval."""
    rng = np.random.default_rng(SEED)
    target = (np.asarray(y)[:, None] == CLASSES[None, :])
    hit = lambda proba: (proba.argmax(axis=1) == target.argmax(axis=1)).astype(float)
    nll = lambda proba: -np.log(proba[target])
    draws = rng.integers(0, len(y), size=(n_boot, len(y)))
    out = {}
    for name, proba in single_probas.items():
        row = {}
        for metric, fn in (("accuracy", hit), ("log_loss", nll)):
            diff = fn(ensemble_proba) - fn(proba)
            boot = diff[draws].mean(axis=1)
            row[metric] = {"difference": round(float(diff.mean()), 4),
                           "ci95": [round(float(np.quantile(boot, 0.025)), 4), round(float(np.quantile(boot, 0.975)), 4)]}
        out[name] = row
    return out


def plot_confusion(matrices: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, len(matrices), figsize=(9.6, 3.9), facecolor=SURFACE)
    for ax, (title, cm) in zip(axes, matrices.items()):
        share = cm / cm.sum(axis=1, keepdims=True) * 100
        ax.imshow(share, cmap=SEQUENTIAL, vmin=0, vmax=100)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{share[i, j]:.0f}%\n{cm[i, j]:,}", ha="center", va="center", fontsize=9.5,
                        color="white" if share[i, j] > 55 else INK, fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(3), NAMES)
        ax.set_yticks(range(3), NAMES)
        ax.set_xlabel("predicted", color=MUTED)
        ax.set_xticks(np.arange(-0.5, 3), minor=True)
        ax.set_yticks(np.arange(-0.5, 3), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="both", length=0, colors=MUTED, labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=INK)
    axes[0].set_ylabel("actual result", color=MUTED)
    fig.text(0.01, -0.03, "Each row adds to 100 %: of the matches that ended this way, how were they predicted?",
             fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_calibration(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot([0.3, 0.85], [0.3, 0.85], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(0.80, 0.815, "perfectly\ncalibrated", fontsize=8.5, color=MUTED, ha="right", va="bottom")
    ax.plot(table["mean_confidence"], table["accuracy"], color=BLUE, linewidth=2, marker="o", markersize=8)
    for row in table.itertuples():
        ax.annotate(f"{row.accuracy:.0%}\nn={row.matches:,}", (row.mean_confidence, row.accuracy), xytext=(9, -16),
                    textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xlim(0.3, 0.88)
    ax.set_ylim(0.3, 0.85)
    ax.set_xlabel("ensemble's probability for its predicted result", color=MUTED)
    ax.set_ylabel("share of those predictions that were right", color=MUTED)
    ax.grid(color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("The ensemble's confidence can be trusted", loc="left", fontsize=11, fontweight="bold", color=INK)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    preds = pd.read_parquet(HERE / "predictions" / "ensemble_test.parquet").merge(
        pd.read_parquet(ROOT / "datasets" / "matches.parquet")[["match_api_id", "result", "league"]], on="match_api_id")
    y = preds["result"]
    variants = {"Most likely result": preds["prediction"], "With the draw weight": preds["prediction_draw_weighted"]}

    report, matrices = {"test_matches": int(len(preds))}, {}
    for name, pred in variants.items():
        matrices[name] = confusion_matrix(y, pred, labels=LABELS)
        report[name] = {
            "accuracy": round(accuracy_score(y, pred), 4), "macro_f1": round(f1_score(y, pred, average="macro"), 4),
            "predicted_counts": {label: int((pred == label).sum()) for label in LABELS},
            "per_class": per_class(y, pred),
            "confusion_rows_actual_cols_predicted_HDA": matrices[name].tolist(),
        }

    preds["confidence"] = preds[["p_H", "p_D", "p_A"]].max(axis=1)
    preds["correct"] = preds["prediction"] == y
    preds["bin"] = pd.cut(preds["confidence"], CONFIDENCE_BINS)
    calibration = preds.groupby("bin", observed=True).agg(
        matches=("correct", "size"), mean_confidence=("confidence", "mean"), accuracy=("correct", "mean")).reset_index()
    calibration["bin"] = calibration["bin"].astype(str)
    report["calibration"] = calibration.round(3).to_dict("records")

    by_league = preds.groupby("league").apply(lambda d: pd.Series({
        "matches": len(d), "accuracy": accuracy_score(d["result"], d["prediction"]),
        "macro_f1_draw_weighted": f1_score(d["result"], d["prediction_draw_weighted"], average="macro"),
    }), include_groups=False).sort_values("accuracy", ascending=False)
    report["by_league"] = {league: {"matches": int(r["matches"]), "accuracy": round(r["accuracy"], 3),
                                    "macro_f1_draw_weighted": round(r["macro_f1_draw_weighted"], 3)}
                           for league, r in by_league.iterrows()}

    # ensemble vs each single model, resampling the same test matches for both
    base, probs = load()
    test = base[base["split"] == "test"].reset_index(drop=True)
    assert (test["match_api_id"].to_numpy() == preds["match_api_id"].to_numpy()).all()
    singles = {name: probs[(base["split"] == "test").to_numpy(), i] for i, name in enumerate(MODELS)}
    report["ensemble_minus_single_model_paired_bootstrap"] = paired_bootstrap(
        y, preds[["p_H", "p_D", "p_A"]].to_numpy(), singles)

    (HERE / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    plot_confusion(matrices, HERE / "figures" / "confusion_matrices.png")
    plot_calibration(calibration, HERE / "figures" / "calibration.png")


if __name__ == "__main__":
    main()
