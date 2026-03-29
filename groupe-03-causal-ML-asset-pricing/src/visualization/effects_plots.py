"""
Effects Plots
=============

Visualization of treatment effect estimates:
    - ATE comparison across methods
    - CATE distribution
    - Forest plots with confidence intervals
    - Heterogeneous effects heatmaps
    - Sensitivity analysis plots
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns


# Global style
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
    "font.family": "sans-serif",
})

PALETTE = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c"]


def plot_ate_comparison(
    estimates: Dict[str, Dict[str, float]],
    true_ate: Optional[float] = None,
    title: str = "ATE Comparison Across Methods",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart comparing ATE estimates across methods.

    Parameters
    ----------
    estimates : dict
        {method_name: {"ate": float, "ci_lower": float, "ci_upper": float}}
    true_ate : float, optional
        Ground-truth ATE (for synthetic data).
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    methods = list(estimates.keys())
    ates = [estimates[m]["ate"] for m in methods]
    ci_lower = [estimates[m].get("ci_lower", estimates[m]["ate"]) for m in methods]
    ci_upper = [estimates[m].get("ci_upper", estimates[m]["ate"]) for m in methods]

    errors_lower = [a - cl for a, cl in zip(ates, ci_lower)]
    errors_upper = [cu - a for a, cu in zip(ates, ci_upper)]

    y_pos = np.arange(len(methods))

    bars = ax.barh(
        y_pos, ates,
        xerr=[errors_lower, errors_upper],
        color=PALETTE[:len(methods)],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        capsize=5,
        error_kw={"elinewidth": 2, "capthick": 2, "color": "#ecf0f1"},
    )

    if true_ate is not None:
        ax.axvline(x=true_ate, color="#e74c3c", linestyle="--", linewidth=2.5,
                   label=f"True ATE = {true_ate:.4f}", alpha=0.9)
        ax.legend(fontsize=12, facecolor="#16213e", edgecolor="#7f8c8d", labelcolor="white")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=12)
    ax.set_xlabel("Average Treatment Effect (ATE)", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.grid(axis="x", alpha=0.3)

    # Annotate values
    for i, (bar, ate) in enumerate(zip(bars, ates)):
        ax.text(
            bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{ate:.4f}", va="center", fontsize=10, color="white", fontweight="bold",
        )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[VIZ] ATE comparison saved to {save_path}")

    return fig


def plot_cate_distribution(
    cate_values: np.ndarray,
    true_cate: Optional[np.ndarray] = None,
    title: str = "Distribution of Conditional Average Treatment Effects (CATE)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot the distribution of estimated CATE values."""
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.hist(
        cate_values, bins=50, alpha=0.7, color="#3498db",
        edgecolor="white", linewidth=0.5, label="Estimated CATE", density=True,
    )

    if true_cate is not None:
        ax.hist(
            true_cate, bins=50, alpha=0.5, color="#2ecc71",
            edgecolor="white", linewidth=0.5, label="True CATE", density=True,
        )

    ax.axvline(np.mean(cate_values), color="#e74c3c", linestyle="--", linewidth=2,
               label=f"Mean = {np.mean(cate_values):.4f}")

    ax.set_xlabel("CATE (Effect of 1σ Earnings Surprise on Return)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=11, facecolor="#16213e", edgecolor="#7f8c8d", labelcolor="white")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[VIZ] CATE distribution saved to {save_path}")

    return fig


def plot_heterogeneous_effects_by_sector(
    sector_df: pd.DataFrame,
    title: str = "Heterogeneous Treatment Effects by Sector",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Forest plot of CATE by sector with confidence intervals.

    Parameters
    ----------
    sector_df : pd.DataFrame
        Must have columns: Sector, Mean CATE, CI Lower, CI Upper.
        Optionally: True CATE.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    sectors = sector_df["Sector"].values
    means = sector_df["Mean CATE"].values
    ci_low = sector_df["CI Lower"].values
    ci_high = sector_df["CI Upper"].values

    y_pos = np.arange(len(sectors))

    ax.barh(
        y_pos, means,
        xerr=[means - ci_low, ci_high - means],
        color=PALETTE[0], alpha=0.8,
        edgecolor="white", linewidth=0.5,
        capsize=4, error_kw={"elinewidth": 1.5, "capthick": 1.5, "color": "#ecf0f1"},
    )

    if "True CATE" in sector_df.columns:
        true_vals = sector_df["True CATE"].values
        ax.scatter(true_vals, y_pos, color="#e74c3c", s=80, zorder=5,
                   marker="D", label="True CATE", edgecolors="white", linewidths=1)
        ax.legend(fontsize=11, facecolor="#16213e", edgecolor="#7f8c8d", labelcolor="white")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sectors, fontsize=11)
    ax.set_xlabel("CATE (Effect per 1σ Surprise)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.axvline(x=0, color="#7f8c8d", linestyle="-", linewidth=1, alpha=0.5)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[VIZ] Sector effects saved to {save_path}")

    return fig


def plot_cate_by_size(
    size_df: pd.DataFrame,
    title: str = "Treatment Effect by Firm Size",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot CATE across market cap quintiles."""
    fig, ax = plt.subplots(figsize=(10, 6))

    quintiles = size_df.index.astype(str).values
    means = size_df["cate_mean"].values
    stds = size_df["cate_std"].values

    x_pos = np.arange(len(quintiles))
    ax.bar(x_pos, means, yerr=stds / np.sqrt(size_df["count"].values) * 1.96,
           color=PALETTE[:len(quintiles)], alpha=0.85,
           edgecolor="white", linewidth=0.5,
           capsize=5, error_kw={"elinewidth": 2, "color": "#ecf0f1"})

    if "true_cate_mean" in size_df.columns:
        ax.scatter(x_pos, size_df["true_cate_mean"].values,
                   color="#e74c3c", s=80, zorder=5, marker="D",
                   label="True CATE", edgecolors="white", linewidths=1)
        ax.legend(fontsize=11, facecolor="#16213e", edgecolor="#7f8c8d", labelcolor="white")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(quintiles, fontsize=10, rotation=30)
    ax.set_ylabel("CATE", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())

    return fig


def plot_sensitivity(
    sensitivity_df: pd.DataFrame,
    original_ate: float,
    title: str = "Sensitivity to Unobserved Confounding",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot how ATE changes with confounder strength."""
    fig, ax = plt.subplots(figsize=(10, 6))

    gammas = sensitivity_df["Confounder Strength (γ)"].values
    ates = sensitivity_df["ATE Estimate"].values
    ci_low = sensitivity_df["CI Lower"].values
    ci_high = sensitivity_df["CI Upper"].values

    ax.plot(gammas, ates, "o-", color="#3498db", linewidth=2.5, markersize=8,
            label="ATE Estimate", zorder=5)
    ax.fill_between(gammas, ci_low, ci_high, alpha=0.2, color="#3498db")

    ax.axhline(y=original_ate, color="#2ecc71", linestyle="--", linewidth=2,
               label=f"Original ATE = {original_ate:.4f}")
    ax.axhline(y=0, color="#e74c3c", linestyle=":", linewidth=1.5, alpha=0.7,
               label="Zero Effect")

    ax.set_xlabel("Confounder Strength (γ)", fontsize=12)
    ax.set_ylabel("ATE Estimate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=11, facecolor="#16213e", edgecolor="#7f8c8d", labelcolor="white")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())

    return fig


def plot_refutation_summary(
    refutation_df: pd.DataFrame,
    title: str = "DoWhy Refutation Tests",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Visualize refutation test results."""
    fig, ax = plt.subplots(figsize=(10, 6))

    tests = refutation_df["Refutation Test"].values
    original = refutation_df["Original Effect"].values
    new_effect = refutation_df["New Effect"].values
    passed = refutation_df["Passed"].values

    x_pos = np.arange(len(tests))
    width = 0.35

    ax.bar(x_pos - width / 2, original, width, label="Original Effect",
           color="#3498db", alpha=0.85, edgecolor="white")
    colors = ["#2ecc71" if p else "#e74c3c" for p in passed]
    ax.bar(x_pos + width / 2, new_effect, width, label="After Refutation",
           color=colors, alpha=0.85, edgecolor="white")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(tests, fontsize=10, rotation=20, ha="right")
    ax.set_ylabel("Effect Estimate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=11, facecolor="#16213e", edgecolor="#7f8c8d", labelcolor="white")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())

    return fig


def plot_ols_vs_dml_bias(
    ols_coef: float,
    dml_ate: float,
    true_ate: float,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Visual comparison highlighting OLS bias vs DML debiasing."""
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = ["True ATE", "OLS (Biased)", "DML (Debiased)"]
    values = [true_ate, ols_coef, dml_ate]
    colors = ["#2ecc71", "#e74c3c", "#3498db"]

    bars = ax.bar(methods, values, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=1)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                f"{val:.4f}", ha="center", fontsize=12, fontweight="bold", color="white")

    # Bias annotation
    bias = ols_coef - true_ate
    ax.annotate(
        f"Bias = {bias:.4f}",
        xy=(1, ols_coef), xytext=(1.3, ols_coef + 0.005),
        fontsize=11, color="#e74c3c",
        arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=2),
    )

    ax.set_ylabel("Effect Estimate", fontsize=12)
    ax.set_title("OLS Bias vs DML Debiasing", fontsize=14, fontweight="bold", pad=15)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())

    return fig
