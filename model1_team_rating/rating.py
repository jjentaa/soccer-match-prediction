"""Model 1 - team rating with a Poisson regression (Dixon-Coles style).

Question: team A plays team B - who wins? Answered from team results only (no players, no tactics).

Every team gets two numbers, refitted on every match day:
    attack   how many goals it scores compared with an average team of its league
    defence  how few goals it concedes compared with an average team of its league

For a match, home team i vs away team j:
    expected home goals  lambda = exp(intercept + home + attack_i - defence_j)
    expected away goals  mu     = exp(intercept        + attack_j - defence_i)

The two goal counts are Poisson. Summing the score grid gives P(home win), P(draw), P(away win).
The Dixon-Coles factor rho corrects the 0-0, 1-0, 0-1 and 1-1 scores, which plain Poisson gets wrong.

Fitting, per league and per match day d:
  - only *training* matches dated strictly before d are used, so no test label reaches a prediction
  - older matches count less: weight = 0.5 ** (age in days / half_life)
  - attack and defence are fitted with sklearn's PoissonRegressor (L2 penalty keeps teams with
    little history close to the league average), then rho is fitted on the same matches with
    attack and defence held fixed

Usage: imported by train.py and predict.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "datasets"

MAX_GOALS = 10          # score grid is 0..10 goals per team
WINDOW_DAYS = 3 * 365   # history used for one fit; older matches have ~0 weight anyway
MIN_HISTORY = 50        # fewer training matches than this -> fall back to base rates
RIDGE = 0.001           # L2 penalty on attack / defence (PoissonRegressor's alpha)


def load_matches() -> pd.DataFrame:
    """Matches with only the columns Model 1 needs, plus the shared 70:30 split (seed 42)."""
    cols = ["match_api_id", "league", "season", "date", "home_team_api_id", "away_team_api_id",
            "home_team_goal", "away_team_goal", "result"]
    matches = pd.read_parquet(DATA / "matches.parquet", columns=cols)
    split = pd.read_parquet(DATA / "split.parquet")
    return matches.merge(split, on="match_api_id").sort_values("date", ignore_index=True)


# ---------------------------------------------------------------- fitting

def design_matrix(home_idx, away_idx, n_teams):
    """Two rows per match: the home side's goals, then the away side's goals.

    Columns: attack of each team (+1 for the scoring team), defence of each team (-1 for the
    conceding team), and one home-advantage column.
    """
    n = len(home_idx)
    rows = np.arange(n)
    X_home = np.zeros((n, 2 * n_teams + 1))
    X_home[rows, home_idx] = 1               # home attack
    X_home[rows, n_teams + away_idx] = -1    # minus away defence
    X_home[:, -1] = 1                        # home advantage
    X_away = np.zeros((n, 2 * n_teams + 1))
    X_away[rows, away_idx] = 1               # away attack
    X_away[rows, n_teams + home_idx] = -1    # minus home defence
    return np.vstack([X_home, X_away])


def tau(home_goals, away_goals, lam, mu, rho):
    """Dixon-Coles correction factor for low scores; 1 for every other score."""
    t = np.ones_like(lam)
    t = np.where((home_goals == 0) & (away_goals == 0), 1 - lam * mu * rho, t)
    t = np.where((home_goals == 0) & (away_goals == 1), 1 + lam * rho, t)
    t = np.where((home_goals == 1) & (away_goals == 0), 1 + mu * rho, t)
    t = np.where((home_goals == 1) & (away_goals == 1), 1 - rho, t)
    return t


def fit_rho(home_goals, away_goals, lam, mu, weights):
    """Best rho for fixed expected goals: maximise the weighted log of the tau factor."""
    def loss(rho):
        t = tau(home_goals, away_goals, lam, mu, rho)
        return -np.sum(weights * np.log(np.clip(t, 1e-10, None)))
    return minimize_scalar(loss, bounds=(-0.3, 0.3), method="bounded").x


def fit_league(history: pd.DataFrame, teams: list, date, half_life=None, use_rho=True) -> dict:
    """Fit one league's ratings from the matches in `history` (all dated before `date`).

    `teams` fixes the order of the attack / defence columns; a team with no match in `history`
    gets attack = defence = 0, i.e. the league average.
    """
    team_pos = {team: k for k, team in enumerate(teams)}
    home_idx = history["home_team_api_id"].map(team_pos).to_numpy()
    away_idx = history["away_team_api_id"].map(team_pos).to_numpy()
    home_goals = history["home_team_goal"].to_numpy()
    away_goals = history["away_team_goal"].to_numpy()

    age_days = (date - history["date"]).dt.days.to_numpy()
    weights = np.ones(len(history)) if half_life is None else 0.5 ** (age_days / half_life)

    X = design_matrix(home_idx, away_idx, len(teams))
    y = np.concatenate([home_goals, away_goals])
    reg = PoissonRegressor(alpha=RIDGE, max_iter=300)
    reg.fit(X, y, sample_weight=np.concatenate([weights, weights]))

    n = len(teams)
    params = {
        "intercept": reg.intercept_,
        "home": reg.coef_[-1],
        "attack": pd.Series(reg.coef_[:n], index=teams),
        "defence": pd.Series(reg.coef_[n:2 * n], index=teams),
        "rho": 0.0,
    }
    if use_rho:
        lam, mu = expected_goals(params, history["home_team_api_id"], history["away_team_api_id"])
        params["rho"] = fit_rho(home_goals, away_goals, lam, mu, weights)
    return params


# ---------------------------------------------------------------- predicting

def expected_goals(params, home_teams, away_teams):
    """Expected goals (lambda, mu) for each home / away pair."""
    att, dfc = params["attack"], params["defence"]
    lam = np.exp(params["intercept"] + params["home"]
                 + att.loc[home_teams].to_numpy() - dfc.loc[away_teams].to_numpy())
    mu = np.exp(params["intercept"]
                + att.loc[away_teams].to_numpy() - dfc.loc[home_teams].to_numpy())
    return lam, mu


def score_grid(lam, mu, rho):
    """P(home scores x, away scores y) for x, y = 0..MAX_GOALS; shape (matches, 11, 11)."""
    goals = np.arange(MAX_GOALS + 1)
    p_home = poisson.pmf(goals[None, :], np.asarray(lam)[:, None])   # (matches, 11)
    p_away = poisson.pmf(goals[None, :], np.asarray(mu)[:, None])
    grid = p_home[:, :, None] * p_away[:, None, :]
    grid[:, :2, :2] *= tau(goals[:2, None], goals[None, :2],
                           np.asarray(lam)[:, None, None], np.asarray(mu)[:, None, None], rho)
    return grid / grid.sum(axis=(1, 2), keepdims=True)


def outcome_probs(lam, mu, rho):
    """P(H), P(D), P(A) from the score grid: below / on / above the diagonal."""
    grid = score_grid(lam, mu, rho)
    p_H = np.tril(grid, k=-1).sum(axis=(1, 2))   # home goals > away goals
    p_D = np.trace(grid, axis1=1, axis2=2)
    p_A = np.triu(grid, k=1).sum(axis=(1, 2))
    return p_H, p_D, p_A


# ---------------------------------------------------------------- refit every match day

def rolling_predictions(matches: pd.DataFrame, half_life=None, use_rho=True, keep_ratings=False):
    """Predict every match (train and test) from training matches dated before its match day.

    Returns one row per match: p_H, p_D, p_A, expected goals, rho, has_rating.
    With keep_ratings=True also returns every team's attack / defence on every match day.
    """
    train = matches[matches["split"] == "train"]
    base_rates = train["result"].value_counts(normalize=True)   # fallback during burn-in

    preds, ratings = [], []
    for league, games in matches.groupby("league"):
        league_train = train[train["league"] == league]
        teams = sorted(set(games["home_team_api_id"]) | set(games["away_team_api_id"]))

        for date, today in games.groupby("date"):
            in_window = (league_train["date"] < date) & \
                        (league_train["date"] >= date - pd.Timedelta(days=WINDOW_DAYS))
            history = league_train[in_window]

            out = today[["match_api_id"]].copy()
            if len(history) < MIN_HISTORY:
                out["p_H"], out["p_D"], out["p_A"] = base_rates["H"], base_rates["D"], base_rates["A"]
                out["lambda_home"] = out["lambda_away"] = out["rho"] = np.nan
                out["has_rating"] = False
            else:
                params = fit_league(history, teams, date, half_life, use_rho)
                lam, mu = expected_goals(params, today["home_team_api_id"], today["away_team_api_id"])
                out["p_H"], out["p_D"], out["p_A"] = outcome_probs(lam, mu, params["rho"])
                out["lambda_home"], out["lambda_away"] = lam, mu
                out["rho"] = params["rho"]
                out["has_rating"] = True
                if keep_ratings:
                    # only teams that have played in the history window have a real rating
                    seen = sorted(set(history["home_team_api_id"]) | set(history["away_team_api_id"]))
                    ratings.append(pd.DataFrame({
                        "team_api_id": seen, "league": league, "date": date,
                        "attack": params["attack"].loc[seen].to_numpy(),
                        "defence": params["defence"].loc[seen].to_numpy(),
                    }))
            preds.append(out)

    preds = matches.merge(pd.concat(preds), on="match_api_id")
    if not keep_ratings:
        return preds
    ratings = pd.concat(ratings, ignore_index=True)
    ratings["rating"] = ratings["attack"] + ratings["defence"]
    return preds, ratings
