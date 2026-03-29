"""
Financial Insights Visualization
=================================

Trading-oriented visualizations:
    - Hypothetical long-short strategy performance
    - Sector allocation based on CATE
    - Counterfactual scenario comparison
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c"]

plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#7f8c8d",
    "text.color": "#ecf0f1",
    "axes.labelcolor": "#ecf0f1",
    "xtick.color": "#bdc3c7",
    "ytick.color": "#bdc3c7",
    "grid.color": "#2c3e50",
    "grid.alpha": 0.3,
})


def plot_trading_strategy_performance(
    strategy_metrics: Dict[str, float],
    title: str = "CATE-Based Trading Strategy Performance",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Visualize trading strategy metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Return comparison
    ax1 = axes[0]
    labels = ["Long Leg", "Short Leg", "Long-Short", "Market"]
    values = [
        strategy_metrics["Long Leg Return (mean)"],
        strategy_metrics["Short Leg Return (mean)"],
        strategy_metrics["Long-Short Spread"],
        strategy_metrics["Market Return (mean)"],
    ]
    colors = ["#2ecc71", "#e74c3c", "#f39c12", "#3498db"]

    bars = ax1.bar(labels, values, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values):
        y_pos = bar.get_height() + 0.0002 if val >= 0 else bar.get_height() - 0.0008
        ax1.text(bar.get_x() + bar.get_width() / 2, y_pos,
                 f"{val:.4f}", ha="center", fontsize=10, fontweight="bold", color="white")

    ax1.set_ylabel("Mean Return", fontsize=12)
    ax1.set_title("Return Decomposition", fontsize=13, fontweight="bold")
    ax1.axhline(y=0, color="#7f8c8d", linewidth=1, alpha=0.5)
    ax1.grid(axis="y", alpha=0.3)

    # Right: Key metrics
    ax2 = axes[1]
    metric_names = ["Alpha", "Sharpe (Ann.)", "L/S Spread"]
    metric_values = [
        strategy_metrics["Alpha (Long - Market)"],
        strategy_metrics["Annualised Sharpe (L/S)"],
        strategy_metrics["Long-Short Spread"],
    ]

    bars2 = ax2.barh(metric_names, metric_values, color=PALETTE[:3], alpha=0.85,
                     edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars2, metric_values):
        ax2.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=10, fontweight="bold", color="white")

    ax2.set_title("Strategy Metrics", fontsize=13, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02, color="white")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())

    return fig


def plot_counterfactual_scenarios(
    scenarios_df: pd.DataFrame,
    title: str = "What-If Scenario Analysis",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Visualize counterfactual scenario results."""
    fig, ax = plt.subplots(figsize=(12, 7))

    scenarios = scenarios_df["Scenario"].values
    impacts = scenarios_df["Mean Return Impact"].values

    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in impacts]

    bars = ax.barh(scenarios, impacts, color=colors, alpha=0.85,
                   edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, impacts):
        x_pos = bar.get_width() + 0.0005 if val >= 0 else bar.get_width() - 0.0005
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha=ha, fontsize=10,
                fontweight="bold", color="white")

    ax.axvline(x=0, color="#7f8c8d", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Mean Return Impact", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())

    return fig


def plot_sector_allocation(
    sector_cate_df: pd.DataFrame,
    title: str = "Optimal Sector Allocation Based on Causal Effects",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Pie/bar chart of sector allocation weights based on CATE magnitude."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sectors = sector_cate_df["Sector"].values
    cates = sector_cate_df["Mean CATE"].values

    # Normalize to allocation weights (positive CATE → long, negative → short)
    positive_mask = cates > 0
    if positive_mask.sum() > 0:
        weights = np.abs(cates)
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(cates)) / len(cates)

    # Left: Bar chart of CATE
    colors = ["#2ecc71" if c > np.median(cates) else "#3498db" if c > 0 else "#e74c3c"
              for c in cates]
    axes[0].barh(sectors, cates, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel("CATE", fontsize=12)
    axes[0].set_title("Causal Effect by Sector", fontsize=13, fontweight="bold")
    axes[0].axvline(x=0, color="#7f8c8d", linewidth=1, alpha=0.5)
    axes[0].grid(axis="x", alpha=0.3)

    # Right: Allocation weights
    top_n = min(6, len(sectors))
    top_idx = np.argsort(weights)[-top_n:]
    top_sectors = sectors[top_idx]
    top_weights = weights[top_idx]
    top_weights = top_weights / top_weights.sum()

    wedges, texts, autotexts = axes[1].pie(
        top_weights, labels=top_sectors, autopct="%1.1f%%",
        colors=PALETTE[:top_n], startangle=90,
        textprops={"color": "white", "fontsize": 9},
    )
    for autotext in autotexts:
        autotext.set_fontweight("bold")

    axes[1].set_title("Suggested Allocation (Top Sectors)", fontsize=13, fontweight="bold")

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02, color="white")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())

    return fig
