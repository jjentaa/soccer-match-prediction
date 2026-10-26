"""Whole-system answer to "team A (home) vs team B (away), who wins?".

Asks Model 1 (team rating), Model 2 (lineup) and Model 3 (style graph) for their probabilities and
combines them with the stacking model chosen in ensemble.py. Both teams must play in the same league.

Usage:
    python ensemble/predict.py "Barcelona" "Real Madrid CF"
    python ensemble/predict.py "Manchester United" "Chelsea" --date 2015-11-21
Run every model's train.py and ensemble/ensemble.py first.
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_DIRS = {"model 1: team rating": "model1_team_rating", "model 2: lineup": "model2_lineup",
              "model 3: style graph": "model3"}
OUTCOME = {"H": "home win", "D": "draw", "A": "away win"}


def ensure_openmp() -> None:
    """LightGBM (Model 2) needs libomp, which macOS lacks. scikit-learn ships a copy, but the loader only
    looks there if told so before Python starts - so restart once with the path set."""
    if sys.platform != "darwin" or os.environ.get("MODEL_OPENMP_SET"):
        return
    try:
        import lightgbm  # noqa: F401
    except OSError:
        import sklearn
        libs = Path(sklearn.__file__).parent / ".dylibs"
        if (libs / "libomp.dylib").exists():
            env = dict(os.environ, DYLD_LIBRARY_PATH=str(libs), MODEL_OPENMP_SET="1")
            os.execve(sys.executable, [sys.executable, *sys.argv], env)


def load_predict(folder: str):
    """Import <folder>/predict.py. The model folders reuse module names (train, features, predict),
    so each one is imported with only its own folder on the path and the shared names cleared first."""
    for name in ("train", "features", "rating", "build_features", "plot_style"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(ROOT / folder))
    try:
        spec = importlib.util.spec_from_file_location(f"{folder}_predict", ROOT / folder / "predict.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(ROOT / folder))
    return module.predict


def predict(team_a, team_b, date=None) -> dict:
    """Each model's probabilities and the stacked ensemble's, for team_a at home against team_b."""
    import numpy as np

    singles = {}
    for name, folder in MODEL_DIRS.items():
        singles[name] = load_predict(folder)(team_a, team_b, date)
        date = date or singles[name]["date"]          # all three models describe the same day

    for name in ("ensemble",):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(HERE))
    from ensemble import CLASSES, fit_stacking, load, reweight_draw

    # the stacking model is a small logistic regression: refit it from the stored out-of-sample predictions
    base, probs = load()
    train = (base["split"] == "train").to_numpy()
    stack = fit_stacking(probs[train], base["result"].to_numpy()[train])
    x = np.array([[[singles[name][f"p_{c}"] for c in CLASSES] for name in MODEL_DIRS]])
    proba = stack(x)[0]
    draw_weight = json.loads((HERE / "metrics.json").read_text())["draw_weight"]
    weighted = reweight_draw(proba, draw_weight)

    first = next(iter(singles.values()))
    home = first["home"]["name"] if isinstance(first["home"], dict) else first["home"]
    away = first["away"]["name"] if isinstance(first["away"], dict) else first["away"]
    calls = {name: CLASSES[int(np.argmax([r[f"p_{c}"] for c in CLASSES]))] for name, r in singles.items()}
    return {
        "home": home, "away": away, "date": date, "models": singles, "model_calls": calls,
        "models_agree": len(set(calls.values())) == 1,
        "p_H": round(float(proba[0]), 3), "p_D": round(float(proba[1]), 3), "p_A": round(float(proba[2]), 3),
        "prediction": OUTCOME[CLASSES[int(np.argmax(proba))]],
        "prediction_draw_weighted": OUTCOME[CLASSES[int(np.argmax(weighted))]],
        "draw_weight": draw_weight,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("home", help="home team: name, short name or team_api_id")
    parser.add_argument("away", help="away team: name, short name or team_api_id")
    parser.add_argument("--date", help="match date YYYY-MM-DD (default: day after the later team's last match)")
    args = parser.parse_args()
    ensure_openmp()

    r = predict(args.home, args.away, args.date)
    line = lambda p: f"home {p['p_H']:6.1%}   draw {p['p_D']:6.1%}   away {p['p_A']:6.1%}"
    print(f"{r['home']} (home) vs {r['away']} (away) on {r['date']}\n")
    for name, p in r["models"].items():
        print(f"  {name:<22s} {line(p)}   -> {OUTCOME[r['model_calls'][name]]}")
    print(f"  {'-' * 78}")
    print(f"  {'ensemble (stacking)':<22s} {line(r)}   -> {r['prediction']}")
    agreement = "all three models agree" if r["models_agree"] else "the models disagree: a close match"
    print(f"\n  {agreement}")
    print(f"  prediction: {r['prediction']}"
          f"   (with draw weight {r['draw_weight']}: {r['prediction_draw_weighted']})")


if __name__ == "__main__":
    main()
