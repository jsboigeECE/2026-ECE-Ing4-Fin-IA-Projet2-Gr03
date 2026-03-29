"""
Causal Forest Estimator
=======================

Implements Causal Forests (Wager & Athey, 2018) via EconML's CausalForestDML.

Key idea:
    - Combines the DML residualization with a forest-based final stage
    - Each tree splits to maximize treatment effect heterogeneity
    - Provides point estimates AND confidence intervals for individual CATEs
    - Enables discovery of which subgroups respond most/least to treatment

Use cases in finance:
    - Which sectors react most to earnings surprises?
    - Do small caps react differently than large caps?
    - Can we build a profitable strategy from heterogeneous effects?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from econml.dml import CausalForestDML


@dataclass
class CausalForestResults:
    """Container for Causal Forest estimation results."""

    ate: float
    ate_ci_lower: float
    ate_ci_upper: float
    cate_mean: float
    cate_std: float
    cate_values: np.ndarray = field(repr=False)
    cate_ci_lower: np.ndarray = field(repr=False)
    cate_ci_upper: np.ndarray = field(repr=False)
    feature_importances: Dict[str, float] = field(repr=False)
    n_obs: int = 0
    estimator: Any = field(default=None, repr=False)


def run_causal_forest(
    Y: np.ndarray,
    T: np.ndarray,
    W: np.ndarray,
    X: np.ndarray,
    X_names: Optional[list[str]] = None,
    n_estimators: int = 500,
    max_depth: int = 10,
    min_samples_leaf: int = 20,
    first_stage: str = "gradient_boosting",
    alpha: float = 0.05,
) -> CausalForestResults:
    """
    Fit a Causal Forest using EconML's CausalForestDML.

    Parameters
    ----------
    Y : array (n,)
        Outcome.
    T : array (n,)
        Treatment.
    W : array (n, d_w)
        Confounders.
    X : array (n, d_x)
        Effect modifiers (used for heterogeneity).
    X_names : list of str, optional
        Names of X columns for feature importance.
    n_estimators : int
        Number of trees in the forest.
    max_depth : int
        Maximum tree depth.
    min_samples_leaf : int
        Minimum samples per leaf.
    first_stage : str
        First-stage ML model name.
    alpha : float
        Significance level.

    Returns
    -------
    CausalForestResults
    """
    # First-stage models for residualization
    if first_stage == "gradient_boosting":
        model_y = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            min_samples_leaf=20, random_state=42
        )
        model_t = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            min_samples_leaf=20, random_state=42
        )
    else:
        model_y = RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=20,
            n_jobs=-1, random_state=42
        )
        model_t = RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=20,
            n_jobs=-1, random_state=42
        )

    est = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        cv=5,
    )

    est.fit(Y, T, X=X, W=W)

    # --- CATE estimation ---
    cate = est.effect(X)
    cate_inf = est.effect_inference(X)
    cate_ci = cate_inf.conf_int(alpha=alpha)

    # --- ATE ---
    ate_inf = est.ate_inference(X=X)
    ate = float(ate_inf.mean_point)
    ate_ci = ate_inf.conf_int_mean(alpha=alpha)

    # --- Feature importances ---
    if X_names is None:
        X_names = [f"X_{i}" for i in range(X.shape[1])]

    try:
        fi = est.feature_importances_
        importance_dict = dict(zip(X_names, fi))
    except Exception:
        importance_dict = {name: 0.0 for name in X_names}

    return CausalForestResults(
        ate=ate,
        ate_ci_lower=float(ate_ci[0]),
        ate_ci_upper=float(ate_ci[1]),
        cate_mean=float(np.mean(cate)),
        cate_std=float(np.std(cate)),
        cate_values=cate.flatten(),
        cate_ci_lower=cate_ci[0].flatten(),
        cate_ci_upper=cate_ci[1].flatten(),
        feature_importances=importance_dict,
        n_obs=len(Y),
        estimator=est,
    )


def analyze_heterogeneity_by_group(
    df: pd.DataFrame,
    cate_values: np.ndarray,
    group_col: str = "sector",
) -> pd.DataFrame:
    """
    Analyze heterogeneous treatment effects by group.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe with group column.
    cate_values : array
        Estimated CATE for each observation.
    group_col : str
        Column to group by.

    Returns
    -------
    pd.DataFrame
        Group-level CATE statistics.
    """
    df_analysis = df.copy()
    df_analysis["estimated_cate"] = cate_values

    # Also include ground truth if available
    has_truth = "true_cate" in df_analysis.columns

    agg_dict = {
        "estimated_cate": ["mean", "std", "min", "max", "count"],
    }
    if has_truth:
        agg_dict["true_cate"] = ["mean"]

    grouped = df_analysis.groupby(group_col).agg(agg_dict).round(5)
    grouped.columns = ["_".join(col).strip("_") for col in grouped.columns]

    if has_truth:
        grouped["bias"] = grouped["estimated_cate_mean"] - grouped["true_cate_mean"]

    return grouped.sort_values("estimated_cate_mean", ascending=False)


def cate_by_quantile(
    df: pd.DataFrame,
    cate_values: np.ndarray,
    continuous_col: str = "log_market_cap",
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Analyze CATE variation across quantiles of a continuous variable."""
    df_q = df.copy()
    df_q["estimated_cate"] = cate_values
    df_q["quantile"] = pd.qcut(df_q[continuous_col], q=n_quantiles, duplicates="drop")

    result = df_q.groupby("quantile").agg(
        cate_mean=("estimated_cate", "mean"),
        cate_std=("estimated_cate", "std"),
        count=("estimated_cate", "count"),
    ).round(5)

    if "true_cate" in df.columns:
        true_by_q = df_q.groupby("quantile")["true_cate"].mean().round(5)
        result["true_cate_mean"] = true_by_q
        result["bias"] = result["cate_mean"] - result["true_cate_mean"]

    return result


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.data.generator import generate_synthetic_dataset, get_true_ate
    from src.data.preprocessor import prepare_causal_matrices

    df = generate_synthetic_dataset(n_obs=3000, seed=42)
    matrices = prepare_causal_matrices(df)
    true_ate = get_true_ate(df)

    print("=" * 70)
    print("CAUSAL FOREST ESTIMATION")
    print("=" * 70)

    res = run_causal_forest(
        matrices["Y"], matrices["T"], matrices["W"], matrices["X"],
        X_names=matrices["X_names"],
    )

    print(f"\nTrue ATE:         {true_ate:.5f}")
    print(f"CF ATE Estimate:  {res.ate:.5f}")
    print(f"95% CI:           [{res.ate_ci_lower:.5f}, {res.ate_ci_upper:.5f}]")
    print(f"CATE Std Dev:     {res.cate_std:.5f}")
    print(f"\nFeature importances:")
    for name, imp in sorted(res.feature_importances.items(), key=lambda x: -x[1]):
        print(f"  {name:40s}: {imp:.4f}")

    print("\n--- Heterogeneity by sector ---")
    het_sector = analyze_heterogeneity_by_group(df, res.cate_values, "sector")
    print(het_sector.to_string())
