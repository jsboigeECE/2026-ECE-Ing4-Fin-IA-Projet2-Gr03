"""
Counterfactual / What-If Analysis
=================================

Implements intervention analysis:
    - "What would stock returns be if earnings surprise were +2σ vs -1σ?"
    - Individual Treatment Effects (ITE) estimation
    - Policy evaluation: optimal trading rule based on predicted CATE

This is a key component of the "Excellent" level, moving from
estimation to actionable financial insights.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def compute_counterfactual_outcomes(
    estimator: Any,
    X: np.ndarray,
    treatment_values: list[float] | None = None,
) -> pd.DataFrame:
    """
    Compute predicted outcomes under different treatment interventions.

    Parameters
    ----------
    estimator : fitted EconML estimator
        Must have an `.effect()` method.
    X : array (n, d)
        Effect modifier features.
    treatment_values : list of float
        Hypothetical treatment values to evaluate.
        Measured in standard deviations of earnings surprise.

    Returns
    -------
    pd.DataFrame
        Counterfactual outcomes for each treatment level.
    """
    if treatment_values is None:
        treatment_values = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

    results = []
    baseline_effect = estimator.effect(X, T0=0, T1=0)  # zero effect

    for t_val in treatment_values:
        effects = estimator.effect(X, T0=0, T1=t_val)
        results.append({
            "Treatment Level (σ)": t_val,
            "Mean Effect on Return": float(np.mean(effects)),
            "Median Effect": float(np.median(effects)),
            "Std Effect": float(np.std(effects)),
            "Min Effect": float(np.min(effects)),
            "Max Effect": float(np.max(effects)),
            "% Positive": float(np.mean(effects > 0) * 100),
        })

    return pd.DataFrame(results)


def compute_individual_treatment_effects(
    estimator: Any,
    X: np.ndarray,
    df: pd.DataFrame,
    treatment_shift: float = 1.0,
) -> pd.DataFrame:
    """
    Compute Individual Treatment Effects (ITE) for each observation.

    Parameters
    ----------
    estimator : fitted EconML estimator
    X : array (n, d)
        Effect modifiers.
    df : pd.DataFrame
        Original dataframe (for merging metadata).
    treatment_shift : float
        Counterfactual treatment shift (in σ units).

    Returns
    -------
    pd.DataFrame
        Original data with ITE estimates appended.
    """
    ite = estimator.effect(X, T0=0, T1=treatment_shift).flatten()

    df_result = df.copy()
    df_result["ite_estimate"] = ite
    df_result["ite_rank"] = df_result["ite_estimate"].rank(ascending=False)

    return df_result


def evaluate_trading_strategy(
    df: pd.DataFrame,
    ite_col: str = "ite_estimate",
    return_col: str = "stock_return",
    top_pct: float = 0.2,
    bottom_pct: float = 0.2,
) -> Dict[str, float]:
    """
    Evaluate a hypothetical long-short trading strategy based on CATE.

    Strategy:
        - LONG stocks with highest predicted CATE (most responsive to positive surprise)
        - SHORT stocks with lowest predicted CATE (least responsive)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `ite_col` and `return_col`.
    ite_col : str
        Column with ITE/CATE estimates.
    return_col : str
        Column with actual stock returns.
    top_pct, bottom_pct : float
        Fraction of stocks for long/short legs.

    Returns
    -------
    dict
        Strategy performance metrics.
    """
    n = len(df)
    n_top = max(1, int(n * top_pct))
    n_bottom = max(1, int(n * bottom_pct))

    df_sorted = df.sort_values(ite_col, ascending=False)

    long_returns = df_sorted.head(n_top)[return_col].values
    short_returns = df_sorted.tail(n_bottom)[return_col].values
    all_returns = df[return_col].values

    long_short = np.mean(long_returns) - np.mean(short_returns)
    long_only = np.mean(long_returns)
    market = np.mean(all_returns)

    # Annualised Sharpe (assuming quarterly data → 4 periods per year)
    ls_sharpe = (long_short / np.std(long_returns - short_returns)) * np.sqrt(4) if np.std(long_returns - short_returns) > 0 else 0

    return {
        "Long Leg Return (mean)": float(long_only),
        "Short Leg Return (mean)": float(np.mean(short_returns)),
        "Long-Short Spread": float(long_short),
        "Market Return (mean)": float(market),
        "Alpha (Long - Market)": float(long_only - market),
        "Annualised Sharpe (L/S)": float(ls_sharpe),
        "N Long": n_top,
        "N Short": n_bottom,
    }


def what_if_scenario(
    estimator: Any,
    X: np.ndarray,
    df: pd.DataFrame,
    scenario_name: str = "Bull Market Earnings Beat",
    treatment_from: float = 0.0,
    treatment_to: float = 2.0,
) -> Dict[str, Any]:
    """
    Run a named what-if scenario.

    Example scenarios:
        - "Bull Market Earnings Beat": surprise goes from 0 to +2σ
        - "Earnings Miss": surprise goes from 0 to -1.5σ
        - "Neutral → Slight Beat": surprise goes from 0 to +0.5σ
    """
    effects = estimator.effect(X, T0=treatment_from, T1=treatment_to).flatten()

    return {
        "Scenario": scenario_name,
        "Treatment Shift": f"{treatment_from}σ → {treatment_to}σ",
        "Mean Return Impact": float(np.mean(effects)),
        "Median Return Impact": float(np.median(effects)),
        "Std Return Impact": float(np.std(effects)),
        "Max Positive Impact": float(np.max(effects)),
        "Max Negative Impact": float(np.min(effects)),
        "% Stocks Positively Affected": float(np.mean(effects > 0) * 100),
    }


def run_scenario_analysis(
    estimator: Any,
    X: np.ndarray,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Run a battery of predefined what-if scenarios."""
    scenarios = [
        ("Strong Earnings Beat", 0.0, 2.0),
        ("Moderate Beat", 0.0, 1.0),
        ("Slight Beat", 0.0, 0.5),
        ("Earnings Miss", 0.0, -1.0),
        ("Severe Miss", 0.0, -2.0),
        ("Beat Recovery", -1.0, 1.0),
        ("Deterioration", 1.0, -1.0),
    ]

    results = []
    for name, t0, t1 in scenarios:
        results.append(what_if_scenario(estimator, X, df, name, t0, t1))

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.data.generator import generate_synthetic_dataset
    from src.data.preprocessor import prepare_causal_matrices
    from src.models.causal_forest import run_causal_forest

    df = generate_synthetic_dataset(n_obs=3000, seed=42)
    matrices = prepare_causal_matrices(df)

    # Fit causal forest first
    from src.models.causal_forest import run_causal_forest
    cf_results = run_causal_forest(
        matrices["Y"], matrices["T"], matrices["W"], matrices["X"],
        X_names=matrices["X_names"],
    )

    print("=" * 70)
    print("COUNTERFACTUAL / WHAT-IF ANALYSIS")
    print("=" * 70)

    # Counterfactual outcomes
    cf_outcomes = compute_counterfactual_outcomes(cf_results.estimator, matrices["X"])
    print("\n--- Counterfactual Outcomes ---")
    print(cf_outcomes.to_string(index=False))

    # Trading strategy
    df_ite = compute_individual_treatment_effects(
        cf_results.estimator, matrices["X"], df
    )
    strategy = evaluate_trading_strategy(df_ite)
    print("\n--- Trading Strategy Evaluation ---")
    for k, v in strategy.items():
        print(f"  {k:30s}: {v:.5f}" if isinstance(v, float) else f"  {k:30s}: {v}")

    # Scenario analysis
    scenarios = run_scenario_analysis(cf_results.estimator, matrices["X"], df)
    print("\n--- Scenario Analysis ---")
    print(scenarios.to_string(index=False))
