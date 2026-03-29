"""
Data Preprocessor
=================

Handles feature engineering, encoding, train/test splitting,
and prepares data matrices for causal estimators.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def prepare_causal_matrices(
    df: pd.DataFrame,
    treatment_col: str = "earnings_surprise",
    outcome_col: str = "stock_return",
    confounder_cols: list[str] | None = None,
    effect_modifier_cols: list[str] | None = None,
) -> Dict[str, np.ndarray]:
    """
    Prepare the data matrices required by EconML estimators.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    treatment_col : str
        Name of treatment column (T).
    outcome_col : str
        Name of outcome column (Y).
    confounder_cols : list of str, optional
        Confounder column names (W). If None, uses defaults.
    effect_modifier_cols : list of str, optional
        Effect modifier column names (X). If None, uses defaults.

    Returns
    -------
    dict
        Keys: 'Y', 'T', 'W', 'X', 'X_names', 'W_names', 'feature_names'
    """
    if confounder_cols is None:
        confounder_cols = [
            "log_market_cap", "book_to_market", "momentum",
            "volatility", "analyst_coverage", "institutional_ownership",
        ]
    # Filter to columns that actually exist
    confounder_cols = [c for c in confounder_cols if c in df.columns]

    if effect_modifier_cols is None:
        effect_modifier_cols = ["sector"]
    effect_modifier_cols = [c for c in effect_modifier_cols if c in df.columns]

    # --- Encode categoricals ---
    df_encoded = df.copy()

    # One-hot encode sector
    if "sector" in df_encoded.columns:
        sector_dummies = pd.get_dummies(df_encoded["sector"], prefix="sector", drop_first=True)
        df_encoded = pd.concat([df_encoded, sector_dummies], axis=1)
        sector_dummy_cols = list(sector_dummies.columns)
    else:
        sector_dummy_cols = []

    # Build confounder matrix W (includes dummy-encoded categoricals)
    w_cols = confounder_cols + sector_dummy_cols
    W = df_encoded[w_cols].values.astype(np.float64)

    # Build effect modifier matrix X
    x_cols = sector_dummy_cols + [c for c in confounder_cols if c in ["log_market_cap"]]
    X = df_encoded[x_cols].values.astype(np.float64) if x_cols else W

    # Treatment and outcome
    T = df_encoded[treatment_col].values.astype(np.float64)
    Y = df_encoded[outcome_col].values.astype(np.float64)

    return {
        "Y": Y,
        "T": T,
        "W": W,
        "X": X,
        "W_names": w_cols,
        "X_names": x_cols if x_cols else w_cols,
        "feature_names": w_cols,
        "df": df_encoded,
    }


def train_test_split_temporal(
    df: pd.DataFrame,
    test_frac: float = 0.2,
    time_col: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train/test.

    If `time_col` exists, uses temporal split; otherwise random split.
    """
    if time_col and time_col in df.columns:
        df_sorted = df.sort_values(time_col)
        split_idx = int(len(df_sorted) * (1 - test_frac))
        return df_sorted.iloc[:split_idx].copy(), df_sorted.iloc[split_idx:].copy()
    else:
        from sklearn.model_selection import train_test_split
        return train_test_split(df, test_size=test_frac, random_state=42)


def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and return summary statistics for numerical columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = df[numeric_cols].describe().T
    stats["skewness"] = df[numeric_cols].skew()
    stats["kurtosis"] = df[numeric_cols].kurtosis()
    return stats.round(4)


def check_overlap(
    df: pd.DataFrame,
    treatment_col: str = "earnings_surprise",
    confounder_cols: list[str] | None = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Check the overlap assumption by binning treatment and looking at
    confounder distributions across treatment quantiles.
    """
    if confounder_cols is None:
        confounder_cols = ["log_market_cap", "momentum", "volatility"]
    confounder_cols = [c for c in confounder_cols if c in df.columns]

    df_check = df.copy()
    df_check["treatment_bin"] = pd.qcut(df_check[treatment_col], q=n_bins, duplicates="drop")

    overlap_stats = df_check.groupby("treatment_bin")[confounder_cols].mean()
    return overlap_stats.round(4)
