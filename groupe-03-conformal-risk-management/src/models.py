"""Baseline predictive models for SPY return forecasting.

Three families for the comparison section (Excellent objective):
  - Ridge              : linear, fast, stable baseline
  - GradientBoosting   : non-linear point predictor (used as conformal base)
  - BayesianRidgeInterval  : Bayesian credible intervals (distributional assumption)
  - QuantileRegressionInterval : direct quantile regression intervals (pinball loss)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Model factories (return sklearn Pipelines for consistent scaling)
# ---------------------------------------------------------------------------


def make_ridge_pipeline(alpha: float = 1.0) -> Pipeline:
    """Regularised linear regression (Ridge)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])


def make_rf_pipeline(
    n_estimators: int = 300,
    max_depth: int = 5,
    random_state: int = 42,
) -> Pipeline:
    """Random Forest regressor."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])


def make_gbr_pipeline(
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> Pipeline:
    """Gradient Boosting regressor (point prediction, squared-error loss)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            random_state=random_state,
        )),
    ])


def make_bayesian_ridge_pipeline() -> Pipeline:
    """Bayesian linear regression (automatic relevance determination)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", BayesianRidge(max_iter=500)),
    ])


# ---------------------------------------------------------------------------
# Interval-producing wrappers (for Bayesian and Quantile regression baselines)
# ---------------------------------------------------------------------------


class BayesianRidgeInterval:
    """
    Bayesian credible interval from BayesianRidge posterior predictive distribution.

    BayesianRidge computes the posterior predictive mean and std (integrating
    out the weight uncertainty).  We build the interval as:
        [μ - z_{α/2} · σ,  μ + z_{α/2} · σ]

    This requires a Gaussian noise assumption — unlike conformal prediction,
    coverage is NOT guaranteed when returns are non-Gaussian.
    """

    def __init__(self, alpha: float = 0.05):
        from scipy import stats

        self.alpha = alpha
        self._z = float(stats.norm.ppf(1 - alpha / 2))
        self._pipeline = make_bayesian_ridge_pipeline()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "BayesianRidgeInterval":
        self._pipeline.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._pipeline.predict(X)

    def predict_interval(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        scaler = self._pipeline.named_steps["scaler"]
        model: BayesianRidge = self._pipeline.named_steps["model"]
        X_scaled = scaler.transform(X)
        y_mean, y_std = model.predict(X_scaled, return_std=True)
        lower = y_mean - self._z * y_std
        upper = y_mean + self._z * y_std
        return lower, upper


class QuantileRegressionInterval:
    """
    Direct quantile regression interval using GradientBoosting with quantile loss.

    Two separate models are trained:
        lower model  → predicts the α/2     quantile
        upper model  → predicts the 1-α/2   quantile

    Coverage is NOT guaranteed (the model may underfit the tails), but if well
    calibrated the coverage should be close to 1-α.
    """

    def __init__(self, alpha: float = 0.05, **gbr_kwargs):
        self.alpha = alpha
        self._lo_q = alpha / 2
        self._hi_q = 1 - alpha / 2

        base_kw = dict(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        base_kw.update(gbr_kwargs)

        self._lo_model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                loss="quantile", alpha=self._lo_q, **base_kw
            )),
        ])
        self._hi_model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                loss="quantile", alpha=self._hi_q, **base_kw
            )),
        ])
        self._mid_model = make_gbr_pipeline()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "QuantileRegressionInterval":
        self._lo_model.fit(X_train, y_train)
        self._hi_model.fit(X_train, y_train)
        self._mid_model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._mid_model.predict(X)

    def predict_interval(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._lo_model.predict(X), self._hi_model.predict(X)
