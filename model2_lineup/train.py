"""Model 2 (lineup evaluation): compare tree-based lineup classifiers and export H/D/A probabilities.

Experiments, all scored by 5-fold CV on the training split:
    lgbm_all    LightGBM on every feature
    lgbm_pca    impute + scale + PCA (95 % variance) + LightGBM
    lgbm_select feature selection by LightGBM importance + LightGBM
The best one by CV macro F1 is refit on the whole training split, its draw probability is reweighted
(weight tuned on the out-of-fold predictions) and it is scored once on the test split.

Outputs in model2_lineup/outputs/:
    pred_model2.parquet   match_api_id, set, p_H, p_D, p_A  (train rows are out-of-fold)
    metrics.json          CV and test scores
    feature_importance.csv
    model2.joblib         fitted best model, feature list and draw weight (used by predict.py)
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import SEED, build_dataset

OUT = Path(__file__).resolve().parent / "outputs"
CLASSES = np.array(["A", "D", "H"])  # sklearn sorts labels; probability columns follow this order


def lgbm():
    return LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
        random_state=SEED, verbose=-1,
    )


EXPERIMENTS = {
    "lgbm_all": lgbm,
    "lgbm_pca": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                      PCA(n_components=0.95, random_state=SEED), lgbm()),
    "lgbm_select": lambda: make_pipeline(SelectFromModel(lgbm(), threshold="median"), lgbm()),
}


def scores(y, proba):
    pred = CLASSES[proba.argmax(1)]
    return {"accuracy": round(accuracy_score(y, pred), 4), "macro_f1": round(f1_score(y, pred, average="macro"), 4)}


def reweight_draw(proba, w):
    p = proba.copy()
    p[:, 1] *= w
    return p / p.sum(1, keepdims=True)


def tune_draw_weight(y, proba):
    grid = np.round(np.arange(1.0, 2.01, 0.05), 2)
    f1s = [f1_score(y, CLASSES[reweight_draw(proba, w).argmax(1)], average="macro") for w in grid]
    return float(grid[int(np.argmax(f1s))])


def main():
    OUT.mkdir(exist_ok=True)
    ds = build_dataset()
    feat_cols = [c for c in ds.columns if c.startswith(("home_", "away_", "diff_"))]
    train, test = ds[ds["set"] == "train"], ds[ds["set"] == "test"]
    X_tr, y_tr = train[feat_cols], train["result"].values
    X_te, y_te = test[feat_cols], test["result"].values
    print(f"features: {len(feat_cols)}  train: {len(train)}  test: {len(test)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_results, oof = {}, {}
    for name, make in EXPERIMENTS.items():
        oof[name] = cross_val_predict(make(), X_tr, y_tr, cv=cv, method="predict_proba")
        cv_results[name] = scores(y_tr, oof[name])
        print(f"cv  {name:12s} {cv_results[name]}")

    best = max(cv_results, key=lambda k: cv_results[k]["macro_f1"])
    w = tune_draw_weight(y_tr, oof[best])
    print(f"best: {best}  draw weight: {w}  cv after reweight: {scores(y_tr, reweight_draw(oof[best], w))}")

    model = EXPERIMENTS[best]().fit(X_tr, y_tr)
    joblib.dump({"name": best, "model": model, "features": feat_cols, "draw_weight": w}, OUT / "model2.joblib")
    raw_test = model.predict_proba(X_te)
    test_proba = reweight_draw(raw_test, w)
    pred = CLASSES[test_proba.argmax(1)]

    metrics = {
        "split": "datasets/split.parquet (random 70:30, random_state=42)",
        "cv_train": cv_results,
        "best": best,
        "draw_weight": w,
        "test_raw": scores(y_te, raw_test),
        "test": scores(y_te, test_proba),
        "test_f1_per_class": dict(zip(CLASSES, np.round(f1_score(y_te, pred, average=None, labels=CLASSES), 4))),
        "test_confusion_rows_true_cols_pred_ADH": confusion_matrix(y_te, pred, labels=CLASSES).tolist(),
    }
    print(json.dumps(metrics, indent=2))
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    train_proba = reweight_draw(oof[best], w)
    pred_df = pd.concat([
        pd.DataFrame(train_proba, columns=["p_A", "p_D", "p_H"]).assign(match_api_id=train["match_api_id"].values, set="train"),
        pd.DataFrame(test_proba, columns=["p_A", "p_D", "p_H"]).assign(match_api_id=test["match_api_id"].values, set="test"),
    ])[["match_api_id", "set", "p_H", "p_D", "p_A"]]
    pred_df.to_parquet(OUT / "pred_model2.parquet", index=False)

    full = lgbm().fit(X_tr, y_tr)
    (pd.Series(full.booster_.feature_importance("gain"), index=feat_cols, name="gain")
       .sort_values(ascending=False).to_csv(OUT / "feature_importance.csv"))


if __name__ == "__main__":
    main()
