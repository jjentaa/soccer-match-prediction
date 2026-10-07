"""Model 2 dimension reduction on the training split: PCA vs feature selection.

1. PCA explained variance and what the leading components mean (loadings).
2. 5-fold CV of LightGBM on k inputs, k from 5 to all 351, for two reducers:
     PCA (k components) and LightGBM gain importance (k best features).
   Every reducer is fitted inside the CV folds, so no fold sees its own validation rows.

Figures go to model2_lineup/figures/, tables to model2_lineup/outputs/.
"""
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import plot_style as ps
from features import SEED, build_dataset
from train import OUT, lgbm

K_GRID = [5, 10, 20, 40, 80, 160, 351]
VARIANCE_MARKS = [0.80, 0.90, 0.95]
SCORING = {"accuracy": "accuracy", "macro_f1": "f1_macro"}
METHODS = {
    "PCA": lambda k: [StandardScaler(), PCA(n_components=k, random_state=SEED)],
    "LightGBM importance": lambda k: [SelectFromModel(lgbm(), threshold=-np.inf, max_features=k)],
}


def explained_variance(X, feat_cols, plt):
    pca = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PCA(random_state=SEED)).fit(X)[-1]
    cum = np.cumsum(pca.explained_variance_ratio_)
    needed = {f"{int(v * 100)}%": int(np.searchsorted(cum, v) + 1) for v in VARIANCE_MARKS}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    n = np.arange(1, len(cum) + 1)
    ax.plot(n, cum * 100, color=ps.ACCENT)
    for label, k in needed.items():
        ax.plot(k, cum[k - 1] * 100, "o", markersize=7, color=ps.ACCENT,
                markeredgecolor=ps.SURFACE, markeredgewidth=2)
        ax.annotate(f"{label} at {k} components", (k, cum[k - 1] * 100), xytext=(10, -14),
                    textcoords="offset points", fontsize=9, color=ps.TEXT_2)
    ax.set_xscale("log")
    ax.set_xlabel(f"Number of principal components (of {len(feat_cols)} features, log scale)")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_ylim(0, 102)
    ax.set_title(f"PC1+PC2 hold {cum[1] * 100:.0f}% of the variance; 95% needs {needed['95%']} components")
    ps.save(fig, "pca_explained_variance.png")

    loadings = pd.DataFrame(pca.components_[:10].T, index=feat_cols,
                            columns=[f"PC{i + 1}" for i in range(10)])
    loadings.to_csv(OUT / "pca_loadings.csv")
    return pca.explained_variance_ratio_, needed, loadings


def plot_loadings(loadings, ratio, plt, n_pc=4, top=8):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, ax in enumerate(axes.flat[:n_pc]):
        pc = f"PC{i + 1}"
        s = loadings[pc]
        s = s.reindex(s.abs().sort_values(ascending=False).index[:top])[::-1]
        ax.barh(np.arange(top), s.values, height=0.6,
                color=[ps.SERIES[0] if v > 0 else ps.SERIES[1] for v in s.values])
        ax.set_yticks(np.arange(top), s.index, fontsize=8)
        ax.axvline(0, color=ps.TEXT_2, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{pc}  ({ratio[i] * 100:.1f}% of variance)", fontsize=11)
    fig.suptitle("Top loadings of the first four components (blue positive, orange negative)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    ps.save(fig, "pca_loadings.png")


def selection_curves(X, y, plt):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for method, steps in METHODS.items():
        for k in K_GRID:
            t = time.time()
            pipe = make_pipeline(SimpleImputer(strategy="median"), *steps(k), lgbm())
            res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING)
            row = {"method": method, "k": k}
            row.update({m: float(np.mean(res[f"test_{m}"])) for m in SCORING})
            rows.append(row)
            print(f"{method:20s} k={k:3d}  " + "  ".join(f"{m}={row[m]:.4f}" for m in SCORING)
                  + f"  ({time.time() - t:.0f}s)")
    curve = pd.DataFrame(rows)
    curve.to_csv(OUT / "selection_curve.csv", index=False)
    plot_selection_curve(curve, plt)
    return curve


def plot_selection_curve(curve, plt):
    """Also runnable alone from the saved CSV: plot_selection_curve(pd.read_csv(OUT / "selection_curve.csv"), plt)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    titles = {"accuracy": "Accuracy", "macro_f1": "Macro F1"}
    for ax, metric in zip(axes, SCORING):
        for i, method in enumerate(METHODS):
            d = curve[curve["method"] == method]
            ax.plot(d["k"], d[metric], marker="o", markersize=6, color=ps.SERIES[i], label=method)
        ax.set_xscale("log")
        ax.set_xticks(K_GRID, [str(k) for k in K_GRID])
        ax.minorticks_off()
        ax.set_xlabel("Inputs kept (components or features)")
        ax.set_title(titles[metric], fontsize=11)
    axes[0].legend(loc="lower right")
    spread = {m: curve[m].max() - curve[m].min() for m in SCORING}
    fig.suptitle("5-fold CV of LightGBM after each reducer (training split)\naxes are zoomed: whole range is "
                 f"{spread['accuracy'] * 100:.1f} pt accuracy, {spread['macro_f1'] * 100:.1f} pt macro F1",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    ps.save(fig, "selection_curve.png")


def main():
    ps.setup()
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    ds = build_dataset()
    feat_cols = [c for c in ds.columns if c.startswith(("home_", "away_", "diff_"))]
    train = ds[ds["set"] == "train"]
    X, y = train[feat_cols], train["result"].to_numpy(dtype=str)

    ratio, needed, loadings = explained_variance(X, feat_cols, plt)
    plot_loadings(loadings, ratio, plt)
    curve = selection_curves(X, y, plt)

    best = curve.loc[curve.groupby("method")["macro_f1"].idxmax()].set_index("method")
    top_loadings = {pc: ", ".join(loadings[pc].abs().sort_values(ascending=False).index[:5])
                    for pc in ["PC1", "PC2", "PC3", "PC4"]}
    lines = [
        "# Model 2 PCA and feature selection summary (training split)\n",
        "## Components needed for explained variance\n",
        "\n".join(f"- {k}: {v} components" for k, v in needed.items()),
        f"- PC1-PC4 explain {', '.join(f'{r * 100:.1f}%' for r in ratio[:4])}\n",
        "## Largest loadings\n",
        "\n".join(f"- {pc}: {v}" for pc, v in top_loadings.items()), "",
        "## CV score by reducer and number of inputs\n",
        f"```\n{curve.round(4).to_string(index=False)}\n```\n",
        "## Best k per reducer (by macro F1)\n",
        f"```\n{best.round(4).to_string()}\n```",
    ]
    (OUT / "pca_selection_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
