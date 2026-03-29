"""
Double Machine Learning (DML) Estimator
=======================================

Implements the DML framework of Chernozhukov et al. (2018) using EconML.

Key idea:
    1. Use ML (Random Forest, GBM) to *residualize* both T and Y on confounders W
    2. Regress Y-residuals on T-residuals → debiased causal estimate
    3. Cross-fitting prevents overfitting bias

This yields a √n-consistent, asymptotically normal estimate of the
Average Treatment Effect (ATE) even when the nuisance functions are
estimated with flexible ML methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV

from econml.dml import LinearDML, NonParamDML, CausalForestDML


@dataclass
class DMLResults:
    """Container for DML estimation results."""

    ate: float
    ate_ci_lower: float
    ate_ci_upper: float
    ate_std_error: float
    model_name: str
    first_stage_y: str
    first_stage_t: str
    n_obs: int
    estimator: Any = field(repr=False)
    cate_values: Optional[np.ndarray] = field(default=None, repr=False)


def _get_first_stage_model(name: str) -> Any:
    """Return a first-stage ML model by name."""
    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=20,
            n_jobs=-1, random_state=42
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            min_samples_leaf=20, random_state=42
        ),
        "lasso": LassoCV(cv=5, random_state=42, max_iter=5000),
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Choose from {list(models.keys())}")
    return models[name]


def run_linear_dml(
    Y: np.ndarray,
    T: np.ndarray,
    W: np.ndarray,
    X: Optional[np.ndarray] = None,
    first_stage: str = "random_forest",
    n_folds: int = 5,
    alpha: float = 0.05,
) -> DMLResults:
    """
    Run Linear DML estimator.

    The treatment effect is assumed constant (homogeneous) or linear in X.

    Parameters
    ----------
    Y : array, shape (n,)
        Outcome variable (stock returns).
    T : array, shape (n,)
        Treatment variable (earnings surprise).
    W : array, shape (n, d)
        Confounders.
    X : array, shape (n, p), optional
        Effect modifiers for heterogeneous effects.
    first_stage : str
        Name of first-stage ML model.
    n_folds : int
        Number of cross-fitting folds.
    alpha : float
        Significance level.

    Returns
    -------
    DMLResults
    """
    model_y = _get_first_stage_model(first_stage)
    model_t = _get_first_stage_model(first_stage)

    est = LinearDML(
        model_y=model_y,
        model_t=model_t,
        cv=n_folds,
        random_state=42,
    )

    est.fit(Y, T, X=X, W=W)

    # ATE and inference
    ate_inf = est.ate_inference(X=X)
    ate = ate_inf.mean_point
    ate_ci = ate_inf.conf_int_mean(alpha=alpha)
    ate_se = ate_inf.stderr_mean

    # CATE if X provided
    cate = est.effect(X) if X is not None else None

    return DMLResults(
        ate=float(ate),
        ate_ci_lower=float(ate_ci[0]),
        ate_ci_upper=float(ate_ci[1]),
        ate_std_error=float(ate_se),
        model_name="LinearDML",
        first_stage_y=first_stage,
        first_stage_t=first_stage,
        n_obs=len(Y),
        estimator=est,
        cate_values=cate,
    )


def run_nonparam_dml(
    Y: np.ndarray,
    T: np.ndarray,
    W: np.ndarray,
    X: Optional[np.ndarray] = None,
    first_stage: str = "random_forest",
    n_folds: int = 5,
    alpha: float = 0.05,
) -> DMLResults:
    """
    Run Non-Parametric DML estimator.

    Uses a flexible final-stage model for arbitrary treatment effect functions.
    """
    model_y = _get_first_stage_model(first_stage)
    model_t = _get_first_stage_model(first_stage)
    model_final = _get_first_stage_model("gradient_boosting")

    est = NonParamDML(
        model_y=model_y,
        model_t=model_t,
        model_final=model_final,
        cv=n_folds,
        random_state=42,
    )

    est.fit(Y, T, X=X if X is not None else W, W=W)

    X_for_effect = X if X is not None else W
    ate = float(np.mean(est.effect(X_for_effect)))
    cate = est.effect(X_for_effect)

    # Bootstrap CI for NonParamDML
    ate_se = float(np.std(cate) / np.sqrt(len(cate)))
    z = 1.96
    ate_ci_lower = ate - z * ate_se
    ate_ci_upper = ate + z * ate_se

    return DMLResults(
        ate=ate,
        ate_ci_lower=ate_ci_lower,
        ate_ci_upper=ate_ci_upper,
        ate_std_error=ate_se,
        model_name="NonParamDML",
        first_stage_y=first_stage,
        first_stage_t=first_stage,
        n_obs=len(Y),
        estimator=est,
        cate_values=cate,
    )


def compare_first_stages(
    Y: np.ndarray,
    T: np.ndarray,
    W: np.ndarray,
    X: Optional[np.ndarray] = None,
    first_stages: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compare DML estimates across different first-stage models.

    Returns a DataFrame with ATE estimates and CIs for each first-stage.
    """
    if first_stages is None:
        first_stages = ["random_forest", "gradient_boosting", "lasso"]

    results = []
    for fs in first_stages:
        try:
            r = run_linear_dml(Y, T, W, X, first_stage=fs)
            results.append({
                "First Stage": fs,
                "ATE": r.ate,
                "Std Error": r.ate_std_error,
                "CI Lower": r.ate_ci_lower,
                "CI Upper": r.ate_ci_upper,
            })
        except Exception as e:
            results.append({
                "First Stage": fs,
                "ATE": np.nan,
                "Std Error": np.nan,
                "CI Lower": np.nan,
                "CI Upper": np.nan,
            })
            print(f"  [WARN] {fs} failed: {e}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.data.generator import generate_synthetic_dataset, get_true_ate
    from src.data.preprocessor import prepare_causal_matrices

    df = generate_synthetic_dataset(n_obs=5000, seed=42)
    matrices = prepare_causal_matrices(df)
    true_ate = get_true_ate(df)

    print("=" * 70)
    print("DOUBLE MACHINE LEARNING ESTIMATION")
    print("=" * 70)

    res = run_linear_dml(
        matrices["Y"], matrices["T"], matrices["W"], matrices["X"]
    )
    print(f"\nTrue ATE:       {true_ate:.5f}")
    print(f"DML Estimate:   {res.ate:.5f} ± {res.ate_std_error:.5f}")
    print(f"95% CI:         [{res.ate_ci_lower:.5f}, {res.ate_ci_upper:.5f}]")

    print("\n--- First-stage comparison ---")
    comp = compare_first_stages(matrices["Y"], matrices["T"], matrices["W"], matrices["X"])
    print(comp.to_string(index=False))
