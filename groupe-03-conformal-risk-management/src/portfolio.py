"""Conformal Predictive Portfolio Selection (CPPS).

Inspired by: Vazquez-Alcocer et al. (2024), "Conformal Predictive Portfolio Selection"
and the ACI-for-VaR framework (Angelopoulos et al., 2024).

Strategy rationale
------------------
Split conformal prediction with coverage guarantee 1-α gives:

    P(r_{t+1} ≥ lower_t) ≥ 1 - α

The lower bound is therefore a (1-α)-confidence floor on tomorrow's return.
Equivalently, the conformal Value-at-Risk at level α is:

    ConfVaR_α,t = -lower_t  (positive number = max expected loss)

Two portfolio strategies are implemented:

1. **Binary CPPS** (VaR-controlled):
   - Invest fully in SPY if ConfVaR_α,t < risk_budget (e.g. 1.5% daily)
   - Otherwise stay in cash (risk-off)
   - Interpretation: only enter the market when the guaranteed floor on
     tomorrow's return satisfies a maximum loss constraint.

2. **Volatility-scaled CPPS**:
   - Position size ∝ 1 / interval_width_t   (inverse uncertainty)
   - Scaled so the expected daily P&L volatility matches a target (e.g. 10% p.a.)
   - Wider intervals (high uncertainty) → smaller position; narrower → larger

Both strategies use only the conformal interval computed at time t (no look-ahead).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class PortfolioResult:
    """All outputs from a CPPS strategy run."""

    dates: pd.DatetimeIndex
    y_true: np.ndarray            # actual daily log-returns
    lower: np.ndarray             # conformal interval lower bound
    upper: np.ndarray             # conformal interval upper bound
    weight: np.ndarray            # portfolio weight in SPY [0, 1]
    strategy_return: np.ndarray   # daily log-returns of the strategy
    buy_hold_return: np.ndarray   # daily log-returns of buy-and-hold SPY

    # ------------------------------------------------------------------
    # Cumulative return series (for plotting)
    # ------------------------------------------------------------------

    @property
    def strategy_cumulative(self) -> np.ndarray:
        """Cumulative arithmetic return of the strategy."""
        return np.exp(np.cumsum(self.strategy_return)) - 1.0

    @property
    def buy_hold_cumulative(self) -> np.ndarray:
        """Cumulative arithmetic return of buy-and-hold SPY."""
        return np.exp(np.cumsum(self.buy_hold_return)) - 1.0

    # ------------------------------------------------------------------
    # Performance metrics
    # ------------------------------------------------------------------

    def metrics(self, ann: int = 252) -> dict:
        """
        Return a dictionary of annualised performance metrics.

        Parameters
        ----------
        ann : int
            Number of trading days per year for annualisation.
        """
        sr = self.strategy_return
        bh = self.buy_hold_return

        def annualised_sharpe(r: np.ndarray) -> float:
            mu, sigma = r.mean(), r.std()
            return float(mu / sigma * np.sqrt(ann)) if sigma > 1e-10 else 0.0

        def max_drawdown(r: np.ndarray) -> float:
            cum = np.exp(np.cumsum(r))
            running_max = np.maximum.accumulate(cum)
            dd = (cum - running_max) / running_max
            return float(dd.min())

        def calmar(r: np.ndarray) -> float:
            ann_ret = float(np.exp(r.sum() * ann / len(r)) - 1)
            mdd = abs(max_drawdown(r))
            return ann_ret / mdd if mdd > 1e-10 else np.nan

        return {
            "strategy_total_return_%": round(float(np.exp(sr.sum()) - 1) * 100, 2),
            "bh_total_return_%": round(float(np.exp(bh.sum()) - 1) * 100, 2),
            "strategy_ann_sharpe": round(annualised_sharpe(sr), 3),
            "bh_ann_sharpe": round(annualised_sharpe(bh), 3),
            "strategy_max_drawdown_%": round(max_drawdown(sr) * 100, 2),
            "bh_max_drawdown_%": round(max_drawdown(bh) * 100, 2),
            "strategy_calmar": round(calmar(sr), 3),
            "bh_calmar": round(calmar(bh), 3),
            "days_invested_%": round(float((self.weight > 0).mean()) * 100, 1),
            "avg_weight_%": round(float(self.weight.mean()) * 100, 1),
        }


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------


def run_binary_cpps(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    risk_budget: float = 0.015,
) -> PortfolioResult:
    """
    Binary CPPS: fully invested when ConfVaR < risk_budget, else cash.

    ConfVaR_t = -lower_t  (max expected daily loss at confidence 1-α)

    Parameters
    ----------
    risk_budget : float
        Maximum tolerated conformal VaR (daily).  Default 0.015 = 1.5% per day.
    """
    conf_var = -lower  # positive = potential loss
    weight = np.where(conf_var < risk_budget, 1.0, 0.0)
    strategy_return = weight * y_true

    return PortfolioResult(
        dates=dates,
        y_true=y_true,
        lower=lower,
        upper=upper,
        weight=weight,
        strategy_return=strategy_return,
        buy_hold_return=y_true.copy(),
    )


def run_vol_scaled_cpps(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_vol: float = 0.10,
    ann: int = 252,
    max_leverage: float = 1.0,
) -> PortfolioResult:
    """
    Volatility-scaled CPPS: position size inversely proportional to interval width.

    The interval width 2*q̂_t is a proxy for uncertainty. We scale the position
    so that the realised daily P&L has expected annualised volatility ≈ target_vol.

        weight_t = (target_vol / √ann) / (interval_width_t / 2)

    Capped at max_leverage (default 1.0 = no leverage).

    Parameters
    ----------
    target_vol : float
        Target annualised portfolio volatility (e.g. 0.10 = 10%).
    max_leverage : float
        Maximum weight (1.0 = long-only, no leverage).
    """
    width = upper - lower
    daily_target = target_vol / np.sqrt(ann)

    # Half-width ≈ expected absolute daily move at 1-α confidence
    half_width = np.maximum(width / 2, 1e-6)
    raw_weight = daily_target / half_width
    weight = np.clip(raw_weight, 0.0, max_leverage)

    # Apply weight to next-day return (weight_t is known before seeing y_{t+1})
    strategy_return = weight * y_true

    return PortfolioResult(
        dates=dates,
        y_true=y_true,
        lower=lower,
        upper=upper,
        weight=weight,
        strategy_return=strategy_return,
        buy_hold_return=y_true.copy(),
    )


# ---------------------------------------------------------------------------
# Utility: period-level attribution
# ---------------------------------------------------------------------------


def portfolio_period_breakdown(
    result: PortfolioResult,
    periods: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Decompose strategy vs buy-and-hold performance by named market period.
    """
    from .evaluation import CRISIS_PERIODS

    periods = periods or CRISIS_PERIODS
    records = []

    for name, (start, end) in periods.items():
        mask = (result.dates >= pd.Timestamp(start)) & (result.dates <= pd.Timestamp(end))
        if mask.sum() < 5:
            continue
        sr = result.strategy_return[mask]
        bh = result.buy_hold_return[mask]
        records.append({
            "period": name,
            "n_days": int(mask.sum()),
            "strategy_return_%": round(float(np.exp(sr.sum()) - 1) * 100, 2),
            "bh_return_%": round(float(np.exp(bh.sum()) - 1) * 100, 2),
            "avg_weight_%": round(float(result.weight[mask].mean()) * 100, 1),
        })

    return pd.DataFrame(records)
