"""Evaluation metrics and comparison utilities for conformal prediction intervals.

Metrics implemented
-------------------
- Marginal coverage        : P(y ∈ Î)  ≥ 1-α is the conformal guarantee
- Mean interval width      : smaller width = more informative intervals
- Winkler interval score   : combined penalised score (lower is better)
- Conditional coverage     : coverage stratified by VIX regime (tests adaptiveness)
- Period-level evaluation  : coverage/width for named market regimes
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------


def marginal_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Fraction of observations falling inside the prediction interval."""
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def mean_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Average width of the prediction intervals."""
    return float(np.mean(upper - lower))


def winkler_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> float:
    """
    Winkler interval score.

    WS = E[width + (2/α)(lo - y)·1{y<lo} + (2/α)(y - hi)·1{y>hi}]

    Lower is better. Penalises wide intervals AND missed observations.
    This is a proper scoring rule for interval forecasts.
    """
    width = upper - lower
    penalty = np.where(
        y_true < lower,
        2 * (lower - y_true) / alpha,
        np.where(y_true > upper, 2 * (y_true - upper) / alpha, 0.0),
    )
    return float(np.mean(width + penalty))


def pinball_loss(y_true: np.ndarray, q_pred: np.ndarray, tau: float) -> float:
    """Pinball (quantile) loss for a single quantile level τ."""
    residuals = y_true - q_pred
    return float(np.mean(np.where(residuals >= 0, tau * residuals, (tau - 1) * residuals)))


# ---------------------------------------------------------------------------
# Named crisis / market regime periods within the test window (2020-2024)
# ---------------------------------------------------------------------------

CRISIS_PERIODS: Dict[str, Tuple[str, str]] = {
    "Covid crash (2020-Q1)": ("2020-02-01", "2020-04-30"),
    "Covid recovery (2020 H2)": ("2020-07-01", "2020-12-31"),
    "Post-vaccine bull (2021)": ("2021-01-01", "2021-12-31"),
    "Rate hike bear (2022)": ("2022-01-01", "2022-12-31"),
    "Recovery (2023-2024)": ("2023-01-01", "2024-12-31"),
}


# ---------------------------------------------------------------------------
# Conditional coverage (by VIX regime)
# ---------------------------------------------------------------------------


def conditional_coverage_by_vix(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    vix: np.ndarray,
    bins: Sequence[float] = (0, 15, 20, 25, 35, 999),
    labels: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Coverage and width stratified by VIX level at prediction time.

    Tests conditional coverage (a stronger property than marginal coverage):
    conformal guarantees only marginal coverage, but good adaptive methods
    should also achieve approximate conditional coverage across VIX regimes.
    """
    if labels is None:
        labels = ["VIX<15", "15-20", "20-25", "25-35", "VIX>=35"]

    covered = (y_true >= lower) & (y_true <= upper)
    width = upper - lower

    vix_cat = pd.cut(vix, bins=list(bins), labels=labels[:len(bins) - 1], right=False)

    df = pd.DataFrame({
        "covered": covered,
        "width": width,
        "vix_bin": vix_cat,
    })

    return (
        df.groupby("vix_bin", observed=True)
        .agg(
            coverage=("covered", "mean"),
            mean_width=("width", "mean"),
            n=("covered", "count"),
        )
        .reset_index()
        .rename(columns={"vix_bin": "VIX regime"})
    )


# ---------------------------------------------------------------------------
# Period-level evaluation
# ---------------------------------------------------------------------------


def evaluate_by_period(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
    periods: Optional[Dict[str, Tuple[str, str]]] = None,
) -> pd.DataFrame:
    """
    Coverage, mean width and Winkler score for each named market period.

    Only periods overlapping with `dates` are reported.
    """
    periods = periods or CRISIS_PERIODS
    records = []

    for name, (start, end) in periods.items():
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        if mask.sum() < 5:
            continue
        yt = y_true[mask]
        lo = lower[mask]
        hi = upper[mask]
        records.append({
            "period": name,
            "n_days": int(mask.sum()),
            "coverage": marginal_coverage(yt, lo, hi),
            "mean_width": mean_interval_width(lo, hi),
            "winkler": winkler_score(yt, lo, hi, alpha),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Multi-method comparison table
# ---------------------------------------------------------------------------


def comparison_table(
    methods: Dict[str, Dict[str, float]],
    target_coverage: float = 0.95,
) -> pd.DataFrame:
    """
    Build a summary DataFrame comparing multiple interval methods.

    Parameters
    ----------
    methods : dict of {method_name: {metric_name: value}}
    target_coverage : float
        Highlighted in the coverage column.

    Returns
    -------
    pd.DataFrame with columns: method, coverage, mean_width, winkler, coverage_ok
    """
    rows = []
    for name, metrics in methods.items():
        row = {"method": name}
        row.update(metrics)
        row["coverage_ok"] = abs(metrics.get("coverage", 0) - target_coverage) <= 0.02
        rows.append(row)

    df = pd.DataFrame(rows).set_index("method")

    # Format coverage as percent
    if "coverage" in df.columns:
        df["coverage_%"] = (df["coverage"] * 100).round(2)

    return df
