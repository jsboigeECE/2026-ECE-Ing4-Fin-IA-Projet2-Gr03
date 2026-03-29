"""Visualization utilities for conformal prediction and portfolio results.

All functions return a matplotlib Figure and accept an optional `ax` or `axes`
parameter so they can be embedded in notebook subplots.

Crisis bands shaded in every time-series plot for context.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from .aci import ACIResult
from .portfolio import PortfolioResult

# ---------------------------------------------------------------------------
# Crisis / regime shading configuration
# ---------------------------------------------------------------------------

REGIME_BANDS = {
    "Covid crash\n(2020-Q1)": ("2020-02-01", "2020-04-30", "#e74c3c"),
    "Rate hike\n(2022)":      ("2022-01-01", "2022-12-31", "#8e44ad"),
}


def _shade_regimes(ax: plt.Axes, dates: pd.DatetimeIndex) -> None:
    """Add semi-transparent regime bands to an axes."""
    for label, (s, e, color) in REGIME_BANDS.items():
        start, end = pd.Timestamp(s), pd.Timestamp(e)
        if start > dates[-1] or end < dates[0]:
            continue
        ax.axvspan(start, end, alpha=0.10, color=color)
        mid = start + (end - start) / 2
        ylims = ax.get_ylim()
        ax.text(
            mid, ylims[1], label,
            ha="center", va="top", fontsize=6.5, color=color,
            rotation=0,
        )


def _fmt_xaxis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)


# ---------------------------------------------------------------------------
# 1. Prediction intervals plot (for split conformal and ACI)
# ---------------------------------------------------------------------------


def plot_prediction_intervals(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    title: str = "Conformal Prediction Intervals",
    alpha: float = 0.05,
    shade_regimes: bool = True,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """
    Plot prediction intervals, actual returns, and misses.

    Red dots = observations outside the interval (misses).
    The shaded blue band is the prediction interval.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Interval band
    ax.fill_between(
        dates, lower, upper,
        alpha=0.25, color="steelblue",
        label=f"{int((1 - alpha) * 100)}% interval",
    )

    # Point prediction
    ax.plot(dates, y_pred, color="steelblue", linewidth=0.7, label="Prediction ŷ")

    # Actual returns
    ax.scatter(dates, y_true, s=2, color="black", alpha=0.5, label="Actual r", zorder=4)

    # Misses
    missed = (y_true < lower) | (y_true > upper)
    if missed.any():
        ax.scatter(
            dates[missed], y_true[missed],
            s=14, color="red", zorder=5, label=f"Miss ({missed.sum()})",
        )

    if shade_regimes:
        _shade_regimes(ax, dates)

    coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
    mean_width = float(np.mean(upper - lower))
    ax.set_title(
        f"{title}\nCoverage: {coverage:.1%}  (target {1 - alpha:.0%})  |  "
        f"Mean width: {mean_width * 100:.3f}%",
        fontsize=10,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Log return")
    ax.legend(loc="upper left", fontsize=8, ncol=5)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    _fmt_xaxis(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. ACI-specific: adaptive α_t and rolling coverage
# ---------------------------------------------------------------------------


def plot_aci_diagnostics(result: ACIResult, figsize: tuple = (14, 7)) -> plt.Figure:
    """
    Two-panel plot:
      - Top:    adaptive α_t sequence over time
      - Bottom: rolling 252-day empirical coverage vs target
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    target_alpha = float(result.alpha_seq[0])  # starting value = target
    target_cov = 1.0 - target_alpha

    # --- Panel 1: α_t ---
    ax1 = axes[0]
    ax1.plot(result.dates, result.alpha_seq, color="darkorange", linewidth=0.8, label="α_t (adaptive)")
    ax1.axhline(target_alpha, color="gray", linestyle="--", linewidth=0.8, label=f"Target α = {target_alpha}")
    ax1.set_ylabel("α_t", fontsize=9)
    ax1.set_title("ACI: Adaptive miscoverage rate α_t", fontsize=10)
    ax1.legend(fontsize=8)
    _shade_regimes(ax1, result.dates)

    # --- Panel 2: rolling coverage ---
    ax2 = axes[1]
    ax2.plot(
        result.dates, result.rolling_coverage,
        color="steelblue", linewidth=0.9, label="Rolling 252d coverage",
    )
    ax2.axhline(
        target_cov, color="gray", linestyle="--", linewidth=0.8,
        label=f"Target {target_cov:.0%}",
    )
    ax2.fill_between(
        result.dates, target_cov - 0.02, target_cov + 0.02,
        alpha=0.10, color="green", label="±2% band",
    )
    ax2.set_ylabel("Coverage", fontsize=9)
    ax2.set_ylim(0.7, 1.0)
    ax2.set_title("ACI: Rolling 252-day empirical coverage", fontsize=10)
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=8)
    _shade_regimes(ax2, result.dates)
    _fmt_xaxis(ax2)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Method comparison: coverage and width bar charts
# ---------------------------------------------------------------------------


def plot_method_comparison(
    method_names: Sequence[str],
    coverages: Sequence[float],
    widths: Sequence[float],
    winklers: Optional[Sequence[float]] = None,
    target_coverage: float = 0.95,
    figsize: tuple = (14, 4),
) -> plt.Figure:
    """
    Horizontal bar chart comparing coverage and width across methods.
    Bars are green when coverage is within ±2% of the target.
    """
    n = len(method_names)
    ncols = 3 if winklers is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=figsize)

    colors = [
        "steelblue" if abs(c - target_coverage) <= 0.02 else "tomato"
        for c in coverages
    ]

    # Coverage
    ax = axes[0]
    bars = ax.barh(method_names, [c * 100 for c in coverages], color=colors)
    ax.axvline(target_coverage * 100, color="black", linestyle="--", linewidth=1.2,
               label=f"Target {target_coverage:.0%}")
    ax.axvline((target_coverage - 0.02) * 100, color="gray", linestyle=":", linewidth=0.8)
    ax.axvline((target_coverage + 0.02) * 100, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Empirical coverage (%)")
    ax.set_title("Coverage (target 95%)")
    ax.legend(fontsize=8)
    ax.set_xlim(80, 102)

    # Width
    ax2 = axes[1]
    ax2.barh(method_names, [w * 100 for w in widths], color="steelblue")
    ax2.set_xlabel("Mean width (×100 log-return units)")
    ax2.set_title("Mean interval width")

    # Winkler score
    if winklers is not None and axes[2] is not None:
        ax3 = axes[2]
        ax3.barh(method_names, list(winklers), color="steelblue")
        ax3.set_xlabel("Winkler score (lower = better)")
        ax3.set_title("Winkler interval score")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Portfolio cumulative return and position
# ---------------------------------------------------------------------------


def plot_portfolio(result: PortfolioResult, figsize: tuple = (14, 8)) -> plt.Figure:
    """
    Two-panel plot:
      - Top:    cumulative returns — CPPS strategy vs buy-and-hold
      - Bottom: portfolio weight over time
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    # --- Panel 1: cumulative returns ---
    ax1 = axes[0]
    ax1.plot(
        result.dates, result.strategy_cumulative * 100,
        color="steelblue", linewidth=1.2, label="CPPS Strategy",
    )
    ax1.plot(
        result.dates, result.buy_hold_cumulative * 100,
        color="gray", linewidth=0.9, linestyle="--", label="Buy & Hold SPY",
    )
    _shade_regimes(ax1, result.dates)
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_ylabel("Cumulative return (%)")
    ax1.set_title("CPPS vs Buy-and-Hold SPY", fontsize=10)
    ax1.legend(fontsize=9)

    # --- Panel 2: weight ---
    ax2 = axes[1]
    ax2.fill_between(result.dates, result.weight, alpha=0.6, color="steelblue", label="Weight in SPY")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("Weight")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=8)
    _fmt_xaxis(ax2)
    _shade_regimes(ax2, result.dates)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Conformal VaR over time
# ---------------------------------------------------------------------------


def plot_conformal_var(
    dates: pd.DatetimeIndex,
    lower: np.ndarray,
    y_true: np.ndarray,
    alpha: float = 0.05,
    figsize: tuple = (14, 4),
) -> plt.Figure:
    """
    Plot the conformal VaR_α = -lower_t alongside actual daily returns.

    The conformal guarantee says: with probability ≥ 1-α, the actual loss
    tomorrow will be ≤ ConfVaR.
    """
    conf_var = -lower  # positive: max expected loss
    exceeded = y_true < lower  # actual return worse than VaR

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(dates, conf_var * 100, color="tomato", linewidth=0.8,
            label=f"Conformal VaR_{int(alpha * 100)}% (%/day)")
    ax.bar(dates, -y_true * 100, width=1, alpha=0.3, color="steelblue",
           label="Actual loss (inverted returns)")
    ax.scatter(
        dates[exceeded], (-y_true[exceeded]) * 100,
        s=16, color="red", zorder=5,
        label=f"VaR exceeded ({exceeded.sum()} days = {exceeded.mean():.1%})",
    )
    ax.axhline(0, color="black", linewidth=0.5)
    _shade_regimes(ax, dates)
    ax.set_ylabel("Daily loss / VaR (%)")
    ax.set_title(
        f"Conformal VaR at level {alpha:.0%}  "
        f"(VaR exceeded {exceeded.mean():.1%} of days, target {alpha:.0%})",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    _fmt_xaxis(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Conditional coverage heatmap
# ---------------------------------------------------------------------------


def plot_conditional_coverage(
    cond_df: pd.DataFrame,
    target_coverage: float = 0.95,
    figsize: tuple = (9, 3),
) -> plt.Figure:
    """
    Bar chart of coverage by VIX regime from evaluation.conditional_coverage_by_vix().
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = [
        "steelblue" if abs(c - target_coverage) <= 0.03 else "tomato"
        for c in cond_df["coverage"]
    ]
    ax.bar(cond_df["VIX regime"].astype(str), cond_df["coverage"] * 100, color=colors)
    ax.axhline(target_coverage * 100, color="gray", linestyle="--", linewidth=1.2,
               label=f"Target {target_coverage:.0%}")
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_title("Conditional coverage by VIX regime")
    ax.legend(fontsize=8)
    ax.set_ylim(75, 102)

    # Annotate n
    for i, row in cond_df.iterrows():
        ax.text(i, row["coverage"] * 100 + 0.3, f"n={int(row['n'])}", ha="center", fontsize=7)

    fig.tight_layout()
    return fig
