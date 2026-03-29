"""
Baseline OLS Regression
=======================

Implements a naive OLS regression to estimate the association between
earnings surprises and stock returns. This serves as the **biased baseline**
that causal methods aim to improve upon.

The OLS estimate is biased because confounders (size, momentum, etc.)
affect both the treatment (earnings surprise) and the outcome (return),
creating spurious correlation that inflates or deflates the coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class OLSResults:
    """Container for OLS regression results."""

    coefficient: float
    std_error: float
    ci_lower: float
    ci_upper: float
    p_value: float
    r_squared: float
    n_obs: int
    summary_table: str
    full_model: sm.OLS


def run_ols_baseline(
    df: pd.DataFrame,
    treatment_col: str = "earnings_surprise",
    outcome_col: str = "stock_return",
    confounder_cols: Optional[list[str]] = None,
    alpha: float = 0.05,
) -> OLSResults:
    """
    Run OLS regression: Y ~ T + W.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with treatment, outcome, and confounders.
    treatment_col : str
        Name of the treatment variable.
    outcome_col : str
        Name of the outcome variable.
    confounder_cols : list of str, optional
        Confounder columns. If None, uses all numeric columns except T and Y.
    alpha : float
        Significance level for confidence intervals.

    Returns
    -------
    OLSResults
        Named results container.
    """
    if confounder_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        confounder_cols = [
            c for c in numeric_cols
            if c not in [treatment_col, outcome_col, "true_cate"]
        ]

    # Build design matrix
    X_cols = [treatment_col] + confounder_cols
    X = df[X_cols].values.astype(np.float64)
    X = sm.add_constant(X)
    Y = df[outcome_col].values.astype(np.float64)

    # Fit OLS
    model = sm.OLS(Y, X).fit()

    # Extract treatment coefficient (index 1 = first regressor after constant)
    coef = model.params[1]
    se = model.bse[1]
    ci = model.conf_int(alpha=alpha)
    ci_lower, ci_upper = ci[1, 0], ci[1, 1]
    p_val = model.pvalues[1]

    # Build summary table
    col_names = ["const"] + X_cols
    summary_rows = []
    for i, name in enumerate(col_names):
        summary_rows.append({
            "Variable": name,
            "Coefficient": model.params[i],
            "Std Error": model.bse[i],
            "t-stat": model.tvalues[i],
            "P>|t|": model.pvalues[i],
            "CI Lower": ci[i, 0],
            "CI Upper": ci[i, 1],
        })
    summary_df = pd.DataFrame(summary_rows).round(6)

    return OLSResults(
        coefficient=coef,
        std_error=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_val,
        r_squared=model.rsquared,
        n_obs=model.nobs,
        summary_table=summary_df.to_string(index=False),
        full_model=model,
    )


def run_simple_ols(
    df: pd.DataFrame,
    treatment_col: str = "earnings_surprise",
    outcome_col: str = "stock_return",
) -> OLSResults:
    """
    Run simple OLS without confounders: Y ~ T.

    This shows the raw bivariate association, maximally confounded.
    """
    return run_ols_baseline(df, treatment_col, outcome_col, confounder_cols=[])


def compare_ols_specifications(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare multiple OLS specifications to show sensitivity
    to covariate inclusion.
    """
    specs = {
        "Simple (Y ~ T)": [],
        "With Size": ["log_market_cap"],
        "With Size + Momentum": ["log_market_cap", "momentum"],
        "Full Controls": None,  # uses all
    }

    results = []
    for name, confounders in specs.items():
        r = run_ols_baseline(df, confounder_cols=confounders if confounders is not None else None)
        results.append({
            "Specification": name,
            "Coefficient": r.coefficient,
            "Std Error": r.std_error,
            "95% CI": f"[{r.ci_lower:.5f}, {r.ci_upper:.5f}]",
            "R²": r.r_squared,
            "N": int(r.n_obs),
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.data.generator import generate_synthetic_dataset, get_true_ate

    df = generate_synthetic_dataset(n_obs=5000, seed=42)
    true_ate = get_true_ate(df)

    print("=" * 70)
    print("BASELINE OLS REGRESSION")
    print("=" * 70)

    # Full OLS
    res = run_ols_baseline(df)
    print(f"\nTrue ATE:       {true_ate:.5f}")
    print(f"OLS Estimate:   {res.coefficient:.5f} ± {res.std_error:.5f}")
    print(f"95% CI:         [{res.ci_lower:.5f}, {res.ci_upper:.5f}]")
    print(f"R²:             {res.r_squared:.4f}")
    print(f"Bias:           {res.coefficient - true_ate:.5f}")

    print("\n" + "=" * 70)
    print("OLS SPECIFICATION COMPARISON")
    print("=" * 70)
    comp = compare_ols_specifications(df)
    print(comp.to_string(index=False))
