"""Split Conformal Prediction for financial return regression.

Reference: Papadopoulos et al. (2002), Vovk et al. (2005).

Theory
------
Given a base model f and a calibration set {(x_i, y_i)}_{i=1}^n:

1. Compute nonconformity scores:  s_i = |y_i - f(x_i)|
2. Compute the (1-α)-quantile with finite-sample correction:
       q̂ = quantile(s_1,...,s_n ; level = ⌈(n+1)(1-α)⌉/n)
3. For a new point x:  Ĉ(x) = [f(x) - q̂,  f(x) + q̂]

Marginal coverage guarantee (for exchangeable data):
       P(Y ∈ Ĉ(X)) ≥ 1 - α

Note on financial data: returns are NOT i.i.d. — coverage holds approximately
over the test period under mild stationarity, but is not a formal guarantee.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.base import BaseEstimator


class SplitConformalRegressor:
    """
    Split conformal predictor with symmetric absolute-residual nonconformity scores.

    Parameters
    ----------
    model : sklearn-compatible estimator (or Pipeline)
        Base point predictor. Must implement fit(X, y) and predict(X).
    alpha : float, default=0.05
        Target miscoverage rate (1-α = target coverage, e.g. 0.05 → 95%).
    """

    def __init__(self, model: BaseEstimator, alpha: float = 0.05):
        self.model = model
        self.alpha = alpha
        self.q_hat_: float | None = None
        self.scores_: np.ndarray | None = None
        self._n_cal: int = 0

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> "SplitConformalRegressor":
        """
        Fit on the training set, calibrate on the calibration set.

        Parameters
        ----------
        X_train, y_train : training data for the base model.
        X_cal,   y_cal   : held-out calibration data for the conformal step.
        """
        self.model.fit(X_train, y_train)

        y_hat_cal = self.model.predict(X_cal)
        self.scores_ = np.abs(y_cal - y_hat_cal)
        self._n_cal = len(self.scores_)

        # Finite-sample corrected quantile level
        level = np.ceil((self._n_cal + 1) * (1 - self.alpha)) / self._n_cal
        level = float(np.clip(level, 0.0, 1.0))
        self.q_hat_ = float(np.quantile(self.scores_, level))

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Point prediction from the base model."""
        return self.model.predict(X)

    def predict_interval(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) symmetric prediction interval."""
        if self.q_hat_ is None:
            raise RuntimeError("Call fit() before predict_interval().")
        y_hat = self.model.predict(X)
        return y_hat - self.q_hat_, y_hat + self.q_hat_

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def coverage(self, X: np.ndarray, y: np.ndarray) -> float:
        """Empirical marginal coverage on a test set."""
        lo, hi = self.predict_interval(X)
        return float(np.mean((y >= lo) & (y <= hi)))

    def mean_width(self, X: np.ndarray) -> float:
        """Mean interval width (= 2 * q̂, constant for split conformal)."""
        lo, hi = self.predict_interval(X)
        return float(np.mean(hi - lo))

    def winkler_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Winkler interval score — lower is better.

        WS = width + (2/α)(lo - y)·1{y<lo} + (2/α)(y - hi)·1{y>hi}

        Penalises both wide intervals and missed observations.
        """
        lo, hi = self.predict_interval(X)
        width = hi - lo
        penalty = np.where(
            y < lo, 2 * (lo - y) / self.alpha,
            np.where(y > hi, 2 * (y - hi) / self.alpha, 0.0),
        )
        return float(np.mean(width + penalty))

    def var_estimate(self, X: np.ndarray) -> np.ndarray:
        """
        Conformal Value-at-Risk estimate at level α.

        From the coverage guarantee P(r ≥ lower) ≥ 1-α,
        the conformal VaR at level α is:
            VaR_α = -lower = q̂ - ŷ(x)

        This gives a (1-α)-confidence upper bound on the loss.
        """
        lo, _ = self.predict_interval(X)
        return -lo  # positive number = potential loss
