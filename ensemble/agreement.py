"""Do Models 1, 2 and 3 point the same way on the test matches, and what happens when they do not?

Each model's prediction is its most likely class (no draw weight). Run ensemble.py first.

Usage: python ensemble/agreement.py   -> agreement.json, agreement_sample.csv, figures/agreement_accuracy.png
"""

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ensemble import CLASSES, MODELS, ROOT, SEED, load

HERE = Path(__file__).resolve().parent
SHORT = ["model 1", "model 2", "model 3"]


def plot_accuracy(groups: dict, path: Path) -> None:
    ink, muted, surface, blue, grey = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6", "#8a8980"
    fig, axes = plt.subplots(1, len(groups), figsize=(9.5, 2.6), facecolor=surface, sharex=True, sharey=True)
    for ax, (title, acc) in zip(axes, groups.items()):
        names = list(acc)
        ax.barh(names, list(acc.values()), height=0.6,
                color=[blue if n == "ensemble" else grey for n in names])
        for i, v in enumerate(acc.values()):
            ax.text(v + 0.008, i, f"{v:.1%}", va="center", ha="left", color=ink, fontsize=9)
        ax.set_title(title, loc="left", fontsize=10, color=ink)
        ax.set_xlim(0, 0.68)
        ax.set_xticks([])
        ax.set_facecolor(surface)
        ax.tick_params(length=0, colors=muted, labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle("Share of test matches predicted correctly", x=0.01, ha="left", fontsize=10, color=muted, y=1.08)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=surface)
    plt.close(fig)


def main() -> None:
    base, probs = load()
    test = (base["split"] == "test").to_numpy()
    base, probs = base[test].reset_index(drop=True), probs[test]
    y = base["result"].to_numpy()
    pred = CLASSES[probs.argmax(axis=2)]                     # [match, model]
    ens = pd.read_parquet(HERE / "predictions" / "ensemble_test.parquet").set_index("match_api_id")
    ens = ens.loc[base["match_api_id"], "prediction"].to_numpy()

    distinct = np.array([len(set(row)) for row in pred])
    agree, split, scattered = distinct == 1, distinct == 2, distinct == 3

    def accuracy(mask) -> dict:
        acc = {name: float((pred[mask, i] == y[mask]).mean()) for i, name in enumerate(SHORT)}
        return acc | {"ensemble": float((ens[mask] == y[mask]).mean())}

    def outcomes(mask) -> dict:
        return {c: round(float((y[mask] == c).mean()), 3) for c in CLASSES}

    odd_one_out = {}
    for k, name in enumerate(SHORT):
        a, b = [i for i in range(3) if i != k]
        mask = split & (pred[:, a] == pred[:, b])
        odd_one_out[name] = {
            "matches": int(mask.sum()),
            "odd_model_correct": round(float((pred[mask, k] == y[mask]).mean()), 3),
            "other_two_correct": round(float((pred[mask, a] == y[mask]).mean()), 3),
            "ensemble_sides_with_odd_model": round(float((ens[mask] == pred[mask, k]).mean()), 3),
        }

    groups = {"all three agree": agree, "models disagree": ~agree}
    report = {
        "test_matches": int(len(y)),
        "share": {"all three agree": round(float(agree.mean()), 3), "two against one": round(float(split.mean()), 3),
                  "three different predictions": round(float(scattered.mean()), 3)},
        "matches": {"all three agree": int(agree.sum()), "two against one": int(split.sum()),
                    "three different predictions": int(scattered.sum())},
        "pairwise_agreement": {f"{SHORT[i]} & {SHORT[j]}": round(float((pred[:, i] == pred[:, j]).mean()), 3)
                               for i, j in combinations(range(3), 2)},
        "agreed_prediction": {c: int((pred[agree, 0] == c).sum()) for c in CLASSES},
        "accuracy": {name: {k: round(v, 3) for k, v in accuracy(mask).items()} for name, mask in groups.items()},
        "actual_results": {name: outcomes(mask) for name, mask in groups.items()},
        "two_against_one": odd_one_out,
        "mean_spread_of_p_home": round(float((probs[:, :, 0].max(axis=1) - probs[:, :, 0].min(axis=1)).mean()), 3),
    }
    (HERE / "agreement.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    # a reproducible sample of test matches to read through
    matches = pd.read_parquet(ROOT / "datasets" / "matches.parquet").set_index("match_api_id").loc[base["match_api_id"]]
    idx = np.sort(np.random.default_rng(SEED).choice(len(y), 15, replace=False))
    sample = pd.DataFrame({
        "date": matches["date"].dt.date.to_numpy()[idx],
        "home_team": matches["home_team"].to_numpy()[idx],
        "away_team": matches["away_team"].to_numpy()[idx],
        **{f"{name}": [f"{pred[i, k]} {probs[i, k].max():.2f}" for i in idx] for k, name in enumerate(SHORT)},
        "ensemble": ens[idx],
        "actual": y[idx],
        "models_agree": agree[idx],
        "ensemble_correct": ens[idx] == y[idx],
    })
    sample.to_csv(HERE / "agreement_sample.csv", index=False)
    print(sample.to_string(index=False))

    titles = {f"All three agree ({agree.sum():,} matches)": accuracy(agree),
              f"Models disagree ({(~agree).sum():,} matches)": accuracy(~agree)}
    plot_accuracy(titles, HERE / "figures" / "agreement_accuracy.png")


if __name__ == "__main__":
    main()
