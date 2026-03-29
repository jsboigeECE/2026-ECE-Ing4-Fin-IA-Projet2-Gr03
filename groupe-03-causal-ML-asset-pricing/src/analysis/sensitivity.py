"""
Sensitivity Analysis
====================

Analyzes robustness of causal estimates to potential violations
of the unconfoundedness assumption.

Key questions:
    - How strong would an unobserved confounder need to be to nullify results?
    - Are estimates stable across subsamples?
    - Does the estimate change when we add random noise confounders?
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from econml.dml import LinearDML


def sensitivity_to_unobserved_confounder(
    Y: np.ndarray,
    T: np.ndarray,
    W: np.ndarray,
    X: Optional[np.ndarray] = None,
    effect_strengths: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Test how the ATE changes when adding synthetic unobserved confounders
    of different strengths.

    For each strength γ:
        U ~ Normal(0, 1)
        Y_new = Y + γ·U
        T_new = T + γ·U

    Then re-estimate the ATE. If the estimate stays stable, the original
    result is robust.

    Parameters
    ----------
    Y, T, W : arrays
        Outcome, treatment, confounders.
    X : array, optional
        Effect modifiers.
    effect_strengths : list of float
        Strength of the synthetic confounder.

    Returns
    -------
    pd.DataFrame
        ATE estimates for each confounder strength.
    """
    if effect_strengths is None:
        effect_strengths = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

    results = []
    rng = np.random.default_rng(42)

    for gamma in effect_strengths:
        U = rng.normal(0, 1, len(Y))

        Y_aug = Y + gamma * U
        T_aug = T + gamma * U

        # Add U as an observed confounder
        W_aug = np.column_stack([W, U])

        try:
            est = LinearDML(
                model_y=GradientBoostingRegressor(
                    n_estimators=100, max_depth=4, random_state=42
                ),
                model_t=GradientBoostingRegressor(
                    n_estimators=100, max_depth=4, random_state=42
                ),
                cv=3,
                random_state=42,
            )
            est.fit(Y_aug, T_aug, X=X, W=W_aug)
            ate_inf = est.ate_inference(X=X)
            ate = float(ate_inf.mean_point)
            ci = ate_inf.conf_int_mean()
            se = float(ate_inf.stderr_mean)
        except Exception:
            ate, ci, se = np.nan, (np.nan, np.nan), np.nan

        results.append({
            "Confounder Strength (γ)": gamma,
            "ATE Estimate": ate,
            "Std Error": se,
            "CI Lower": ci[0] if isinstance(ci, tuple) else float(ci[0]),
            "CI Upper": ci[1] if isinstance(ci, tuple) else float(ci[1]),
        })

    return pd.DataFrame(results)


def subsample_stability(
    Y: np.ndarray,
    T: np.ndarray,
    W: np.ndarray,
    X: Optional[np.ndarray] = None,
    fractions: Optional[List[float]] = None,
    n_reps: int = 10,
) -> pd.DataFrame:
    """
    Test stability of ATE estimates across random subsamples.

    Parameters
    ----------
    fractions : list of float
        Subsample fractions to test.
    n_reps : int
        Number of repetitions per fraction.
    """
    if fractions is None:
        fractions = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    results = []
    rng = np.random.default_rng(42)
    n = len(Y)

    for frac in fractions:
        ates = []
        for rep in range(n_reps):
            idx = rng.choice(n, size=int(n * frac), replace=False)

            try:
                est = LinearDML(
                    model_y=GradientBoostingRegressor(
                        n_estimators=100, max_depth=4, random_state=rep
                    ),
                    model_t=GradientBoostingRegressor(
                        n_estimators=100, max_depth=4, random_state=rep
                    ),
                    cv=3,
                    random_state=rep,
                )
                est.fit(Y[idx], T[idx], X=X[idx] if X is not None else None, W=W[idx])
                ate_inf = est.ate_inference(X=X[idx] if X is not None else None)
                ates.append(float(ate_inf.mean_point))
            except Exception:
                continue

        if ates:
            results.append({
                "Sample Fraction": frac,
                "Mean ATE": np.mean(ates),
                "Std ATE": np.std(ates),
                "Min ATE": np.min(ates),
                "Max ATE": np.max(ates),
                "N Successful": len(ates),
            })

    return pd.DataFrame(results).round(5)


def random_cause_test(
    Y: np.ndarray,
    T: np.ndarray,
    W: np.ndarray,
    X: Optional[np.ndarray] = None,
    n_random_vars: int = 5,
) -> pd.DataFrame:
    """
    Add random (irrelevant) variables as confounders and check if ATE changes.

    If the model is well-specified, adding random noise variables
    should not meaningfully change the ATE estimate.
    """
    rng = np.random.default_rng(42)
    results = []

    # Baseline
    est = LinearDML(
        model_y=GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
        model_t=GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
        cv=3, random_state=42,
    )
    est.fit(Y, T, X=X, W=W)
    base_ate = float(est.ate_inference(X=X).mean_point)
    results.append({"N Random Vars": 0, "ATE": base_ate, "Change": 0.0})

    # Add random variables one by one
    for k in range(1, n_random_vars + 1):
        random_cols = rng.normal(0, 1, (len(Y), k))
        W_aug = np.column_stack([W, random_cols])

        try:
            est_k = LinearDML(
                model_y=GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
                model_t=GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
                cv=3, random_state=42,
            )
            est_k.fit(Y, T, X=X, W=W_aug)
            ate_k = float(est_k.ate_inference(X=X).mean_point)
        except Exception:
            ate_k = np.nan

        results.append({
            "N Random Vars": k,
            "ATE": ate_k,
            "Change": ate_k - base_ate if not np.isnan(ate_k) else np.nan,
        })

    return pd.DataFrame(results).round(6)
