"""Adaptive Conformal Inference (ACI) for financial time series.

References
----------
Gibbs, I. & Candes, E. (2021). "Adaptive Conformal Inference Under Distribution Shift."
  NeurIPS 2021. https://arxiv.org/abs/2106.00170

Zaffran, M. et al. (2022). "Adaptive Conformal Predictions for Time Series."
  ICML 2022. https://arxiv.org/abs/2202.07282

Algorithm
---------
ACI addresses the fundamental issue with split conformal for time series:
financial data is non-stationary (distribution shifts during crises), so
the fixed calibration quantile q̂ may under-cover in high-volatility regimes.

ACI maintains an adaptive miscoverage rate α_t that evolves over time:

  1. At step t, compute q̂_t = quantile(past residuals; level = 1 - α_t)
  2. Issue interval: Î_t = [ŷ_t - q̂_t,  ŷ_t + q̂_t]
  3. Observe y_t
  4. Define  err_t = 1{y_t ∉ Î_t}   (1 if missed, 0 if covered)
  5. Update:  α_{t+1} = clip(α_t + γ(α_target - err_t), ε, 1-ε)

Intuition of the update:
  - If missed (err_t=1): α_t decreases → next interval is wider (more conservative)
  - If covered (err_t=0): α_t increases → next interval is slightly narrower

Long-run guarantee (Theorem 1 in Gibbs & Candes):
  (1/T) Σ_t err_t  →  α_target   as T → ∞
even under arbitrary distribution shift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


@dataclass
class ACIResult:
    """Container for all outputs of an ACI online run."""

    dates: pd.DatetimeIndex
    y_true: np.ndarray
    y_pred: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    alpha_seq: np.ndarray        # α_t at each step
    covered: np.ndarray          # 1{y_t ∈ Î_t}
    rolling_coverage: np.ndarray  # rolling empirical coverage

    @property
    def empirical_coverage(self) -> float:
        """Overall empirical coverage on the test period."""
        return float(self.covered.mean())

    @property
    def mean_width(self) -> float:
        """Mean interval width over the test period."""
        return float((self.upper - self.lower).mean())

    @property
    def winkler_score(self) -> float:
        """Winkler interval score (lower is better)."""
        alpha = float(self.alpha_seq[0])  # target alpha
        width = self.upper - self.lower
        penalty = np.where(
            self.y_true < self.lower,
            2 * (self.lower - self.y_true) / alpha,
            np.where(
                self.y_true > self.upper,
                2 * (self.y_true - self.upper) / alpha,
                0.0,
            ),
        )
        return float(np.mean(width + penalty))


class AdaptiveConformalInference:
    """
    Online Adaptive Conformal Inference for time-series regression.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Base point predictor. Fitted once on the training set.
    alpha : float, default=0.05
        Target long-run miscoverage rate (→ target coverage 1-α).
    gamma : float, default=0.005
        Adaptation step size.  Typical values: 0.001 (slow) to 0.05 (fast).
        Larger γ reacts faster to distribution shifts but oscillates more.
    min_cal_size : int, default=50
        Minimum calibration scores required before adapting α_t.
        Before this threshold, the target α is used directly.
    """

    def __init__(
        self,
        model: BaseEstimator,
        alpha: float = 0.05,
        gamma: float = 0.005,
        min_cal_size: int = 50,
    ):
        self.model = model
        self.alpha = alpha
        self.gamma = gamma
        self.min_cal_size = min_cal_size

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        dates_test: pd.DatetimeIndex,
        rolling_window: int = 252,
    ) -> ACIResult:
        """
        Fit model on train, warm-start score buffer on calibration,
        then run ACI step-by-step on the test set.

        Parameters
        ----------
        X_train, y_train   : training data (model is fitted here).
        X_cal,   y_cal     : calibration residuals seed the score buffer.
        X_test,  y_test    : test set — ACI runs online here.
        dates_test         : datetime index for the test set rows.
        rolling_window     : window (in trading days) for rolling coverage plot.
        """
        # --- 1. Fit base model on training data ---
        self.model.fit(X_train, y_train)

        # --- 2. Warm-start: seed score buffer with calibration residuals ---
        y_hat_cal = self.model.predict(X_cal)
        score_buffer: list[float] = list(np.abs(y_cal - y_hat_cal))

        # --- 3. Online ACI loop over test set ---
        alpha_t = self.alpha
        alpha_seq, lowers, uppers, y_preds, covereds = [], [], [], [], []

        for t in range(len(X_test)):
            x_t = X_test[t: t + 1]
            y_t = float(y_test[t])

            # Record current α_t
            alpha_seq.append(alpha_t)

            # Compute quantile with finite-sample correction
            n = len(score_buffer)
            if n < self.min_cal_size:
                # Not enough history — use target α directly
                effective_alpha = self.alpha
            else:
                effective_alpha = alpha_t

            level = np.ceil((n + 1) * (1 - effective_alpha)) / n
            level = float(np.clip(level, 0.0, 1.0))
            q_t = float(np.quantile(score_buffer, level))

            # Issue prediction interval
            y_hat_t = float(self.model.predict(x_t)[0])
            lo_t = y_hat_t - q_t
            hi_t = y_hat_t + q_t

            covered_t = 1.0 if (lo_t <= y_t <= hi_t) else 0.0

            lowers.append(lo_t)
            uppers.append(hi_t)
            y_preds.append(y_hat_t)
            covereds.append(covered_t)

            # ACI update: α_{t+1} = clip(α_t + γ(α_target - err_t))
            err_t = 1.0 - covered_t  # 1 if missed, 0 if covered
            alpha_t = float(np.clip(
                alpha_t + self.gamma * (self.alpha - err_t),
                1e-6,
                1.0 - 1e-6,
            ))

            # Append new residual to the growing score buffer
            score_buffer.append(abs(y_t - y_hat_t))

        # --- 4. Pack results ---
        lowers_arr = np.array(lowers)
        uppers_arr = np.array(uppers)
        covered_arr = np.array(covereds)
        alpha_arr = np.array(alpha_seq)
        y_preds_arr = np.array(y_preds)

        # Rolling empirical coverage (252 trading days ≈ 1 year)
        roll_cov = (
            pd.Series(covered_arr, index=dates_test)
            .rolling(rolling_window, min_periods=20)
            .mean()
            .to_numpy()
        )

        return ACIResult(
            dates=dates_test,
            y_true=y_test,
            y_pred=y_preds_arr,
            lower=lowers_arr,
            upper=uppers_arr,
            alpha_seq=alpha_arr,
            covered=covered_arr,
            rolling_coverage=roll_cov,
        )
