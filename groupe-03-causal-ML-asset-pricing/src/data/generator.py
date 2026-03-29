"""
Synthetic Data Generator for Causal Asset Pricing
==================================================

Generates a realistic synthetic dataset with **known causal structure** so that
causal estimation methods can be validated against ground truth.

Data Generating Process (DGP)
-----------------------------
The DGP mirrors the stylized facts of earnings-announcement returns:

    Confounders W  →  Treatment T  (earnings surprise)
    Confounders W  →  Outcome   Y  (abnormal return)
    Treatment   T  →  Outcome   Y  (causal effect, heterogeneous by sector/size)

Ground-truth parameters are exposed so we can benchmark OLS vs DML vs Causal Forest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Sector definitions (GICS-inspired)
# ---------------------------------------------------------------------------
SECTORS: list[str] = [
    "Technology", "Healthcare", "Financials", "Consumer Discretionary",
    "Industrials", "Energy", "Utilities", "Materials",
    "Communication Services", "Consumer Staples",
]

# True heterogeneous treatment effect by sector (τ_sector)
# Technology reacts most to earnings surprises; Utilities least
TRUE_SECTOR_CATE: Dict[str, float] = {
    "Technology": 0.035,
    "Healthcare": 0.028,
    "Consumer Discretionary": 0.025,
    "Communication Services": 0.022,
    "Industrials": 0.020,
    "Financials": 0.018,
    "Materials": 0.016,
    "Energy": 0.014,
    "Consumer Staples": 0.010,
    "Utilities": 0.008,
}


def _generate_confounders(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate confounder variables (W)."""
    data = pd.DataFrame()

    # Market cap (log-normal, mean ~$10B)
    data["log_market_cap"] = rng.normal(loc=23.0, scale=1.5, size=n)  # ln($)

    # Book-to-market ratio (positive, right-skewed)
    data["book_to_market"] = np.exp(rng.normal(loc=-0.5, scale=0.6, size=n))

    # Momentum: past 12-month return
    data["momentum"] = rng.normal(loc=0.08, scale=0.25, size=n)

    # Volatility (annualised, positive)
    data["volatility"] = np.abs(rng.normal(loc=0.30, scale=0.12, size=n))

    # Analyst coverage (count, correlated with size)
    base_coverage = 2 + 1.5 * (data["log_market_cap"] - 20)
    data["analyst_coverage"] = np.clip(
        rng.poisson(lam=np.clip(base_coverage, 1, 40)), 1, 60
    )

    # Institutional ownership (0-1, correlated with size)
    raw_io = 0.3 + 0.02 * (data["log_market_cap"] - 20) + rng.normal(0, 0.1, n)
    data["institutional_ownership"] = np.clip(raw_io, 0.05, 0.98)

    # Sector assignment (categorical, equal probability)
    data["sector"] = rng.choice(SECTORS, size=n)

    return data


def _generate_instrument(
    confounders: pd.DataFrame, rng: np.random.Generator
) -> np.ndarray:
    """Generate analyst revision instrument (Z), correlated with confounders."""
    n = len(confounders)
    z = (
        0.3 * (confounders["log_market_cap"].values - 23) / 1.5
        + 0.2 * confounders["momentum"].values
        + rng.normal(0, 0.8, n)
    )
    return z


def _treatment_model(
    confounders: pd.DataFrame, instrument: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """
    T = f(W, Z) + noise.

    Earnings surprise depends on fundamentals and analyst expectations.
    """
    n = len(confounders)
    t = (
        -0.15 * (confounders["log_market_cap"].values - 23) / 1.5
        + 0.20 * confounders["momentum"].values
        + 0.10 * confounders["book_to_market"].values
        - 0.05 * confounders["volatility"].values
        + 0.12 * confounders["analyst_coverage"].values / 20
        + 0.25 * instrument
        + rng.normal(0, 0.6, n)
    )
    return t


def _outcome_model(
    treatment: np.ndarray,
    confounders: pd.DataFrame,
    rng: np.random.Generator,
    size_interaction: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Y = τ(X) · T + g(W) + noise.

    Returns (outcome, true_cate) so ground truth is available.
    """
    n = len(confounders)

    # ---- Heterogeneous treatment effect τ(X) ----
    sector_effect = confounders["sector"].map(TRUE_SECTOR_CATE).values

    # Size interaction: small caps react more
    if size_interaction:
        size_mod = 1.0 + 0.3 * (23.0 - confounders["log_market_cap"].values) / 1.5
        size_mod = np.clip(size_mod, 0.5, 2.0)
    else:
        size_mod = np.ones(n)

    true_cate = sector_effect * size_mod  # ground-truth CATE

    # ---- Confounding function g(W) ----
    g_w = (
        0.02 * confounders["momentum"].values
        - 0.01 * (confounders["log_market_cap"].values - 23) / 1.5
        + 0.005 * confounders["book_to_market"].values
        - 0.008 * confounders["volatility"].values
        + 0.003 * confounders["institutional_ownership"].values
    )

    # ---- Outcome ----
    noise_y = rng.normal(0, 0.03, n)
    outcome = true_cate * treatment + g_w + noise_y

    return outcome, true_cate


def generate_synthetic_dataset(
    n_obs: int = 5000,
    seed: int = 42,
    size_interaction: bool = True,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate the full synthetic dataset.

    Parameters
    ----------
    n_obs : int
        Number of firm-quarter observations.
    seed : int
        Random seed for reproducibility.
    size_interaction : bool
        Whether treatment effect varies with firm size.
    save_path : str, optional
        If provided, save the DataFrame to this CSV path.

    Returns
    -------
    pd.DataFrame
        Dataset with columns for confounders, treatment, outcome,
        instrument, and ground-truth CATE.
    """
    rng = np.random.default_rng(seed)

    # 1. Confounders
    df = _generate_confounders(n_obs, rng)

    # 2. Instrument
    df["analyst_revision"] = _generate_instrument(df, rng)

    # 3. Treatment
    df["earnings_surprise"] = _treatment_model(df, df["analyst_revision"].values, rng)

    # 4. Outcome + ground truth
    df["stock_return"], df["true_cate"] = _outcome_model(
        df["earnings_surprise"].values, df, rng, size_interaction
    )

    # 5. Derived features
    df["market_cap_quintile"] = pd.qcut(
        df["log_market_cap"], q=5, labels=["Q1_Small", "Q2", "Q3", "Q4", "Q5_Large"]
    )

    # Reorder columns for clarity
    col_order = [
        "sector", "log_market_cap", "market_cap_quintile",
        "book_to_market", "momentum", "volatility",
        "analyst_coverage", "institutional_ownership",
        "analyst_revision",
        "earnings_surprise",
        "stock_return",
        "true_cate",
    ]
    df = df[col_order]

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"[DATA] Synthetic dataset saved to {save_path}  ({n_obs} rows)")

    return df


def get_variable_roles() -> Dict[str, list[str]]:
    """Return a dictionary describing the causal role of each variable."""
    return {
        "treatment": ["earnings_surprise"],
        "outcome": ["stock_return"],
        "confounders": [
            "log_market_cap", "book_to_market", "momentum",
            "volatility", "analyst_coverage", "institutional_ownership",
        ],
        "effect_modifiers": ["sector", "market_cap_quintile"],
        "instrument": ["analyst_revision"],
        "ground_truth": ["true_cate"],
    }


def get_true_ate(df: pd.DataFrame) -> float:
    """Return the true Average Treatment Effect from the ground-truth column."""
    return float(df["true_cate"].mean())


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = generate_synthetic_dataset(n_obs=5000, seed=42, save_path="data/synthetic_data.csv")
    roles = get_variable_roles()
    print(f"\n[DATA] Variable roles:\n{roles}")
    print(f"[DATA] True ATE = {get_true_ate(df):.5f}")
    print(f"\n[DATA] Summary statistics:\n{df.describe().round(4)}")
