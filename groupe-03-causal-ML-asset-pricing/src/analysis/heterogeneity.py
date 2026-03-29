"""
Heterogeneous Effects Analysis
==============================

Deep analysis of how treatment effects vary across subgroups:
    - By sector (GICS classification)
    - By firm size (market cap quintiles)
    - Interaction effects
    - Statistical tests for heterogeneity
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


def analyze_effects_by_sector(
    df: pd.DataFrame,
    cate_values: np.ndarray,
    sector_col: str = "sector",
) -> pd.DataFrame:
    """Detailed CATE analysis by sector with confidence intervals."""
    df_a = df.copy()
    df_a["cate"] = cate_values

    results = []
    for sector in sorted(df_a[sector_col].unique()):
        mask = df_a[sector_col] == sector
        sector_cate = df_a.loc[mask, "cate"].values

        ci = stats.t.interval(
            0.95,
            df=len(sector_cate) - 1,
            loc=np.mean(sector_cate),
            scale=stats.sem(sector_cate),
        )

        row = {
            "Sector": sector,
            "Mean CATE": np.mean(sector_cate),
            "Std CATE": np.std(sector_cate),
            "CI Lower": ci[0],
            "CI Upper": ci[1],
            "N Obs": len(sector_cate),
        }

        if "true_cate" in df.columns:
            true_mean = df.loc[mask, "true_cate"].mean()
            row["True CATE"] = true_mean
            row["Bias"] = row["Mean CATE"] - true_mean

        results.append(row)

    return pd.DataFrame(results).sort_values("Mean CATE", ascending=False).round(5)


def analyze_effects_by_size(
    df: pd.DataFrame,
    cate_values: np.ndarray,
    size_col: str = "market_cap_quintile",
) -> pd.DataFrame:
    """CATE analysis by market cap quintile."""
    df_a = df.copy()
    df_a["cate"] = cate_values

    results = []
    for q in df_a[size_col].unique():
        mask = df_a[size_col] == q
        q_cate = df_a.loc[mask, "cate"].values

        if len(q_cate) < 2:
            continue

        ci = stats.t.interval(
            0.95, df=len(q_cate) - 1,
            loc=np.mean(q_cate), scale=stats.sem(q_cate),
        )

        row = {
            "Size Quintile": str(q),
            "Mean CATE": np.mean(q_cate),
            "Std CATE": np.std(q_cate),
            "CI Lower": ci[0],
            "CI Upper": ci[1],
            "N Obs": len(q_cate),
        }
        if "true_cate" in df.columns:
            row["True CATE"] = df.loc[mask, "true_cate"].mean()
            row["Bias"] = row["Mean CATE"] - row["True CATE"]

        results.append(row)

    return pd.DataFrame(results).round(5)


def test_heterogeneity(
    df: pd.DataFrame,
    cate_values: np.ndarray,
    group_col: str = "sector",
) -> Dict[str, float]:
    """
    Statistical test for heterogeneity of treatment effects.

    Uses ANOVA (F-test) to test whether mean CATE differs across groups.
    """
    df_a = df.copy()
    df_a["cate"] = cate_values

    groups = [
        df_a.loc[df_a[group_col] == g, "cate"].values
        for g in df_a[group_col].unique()
    ]
    groups = [g for g in groups if len(g) > 1]

    f_stat, p_value = stats.f_oneway(*groups)

    return {
        "F-statistic": float(f_stat),
        "p-value": float(p_value),
        "Significant (α=0.05)": p_value < 0.05,
        "Number of Groups": len(groups),
    }


def sector_size_interaction(
    df: pd.DataFrame,
    cate_values: np.ndarray,
    sector_col: str = "sector",
    size_col: str = "market_cap_quintile",
) -> pd.DataFrame:
    """
    Two-way analysis: CATE by sector × size quintile.

    Returns a pivot table of mean CATE values.
    """
    df_a = df.copy()
    df_a["cate"] = cate_values

    pivot = df_a.pivot_table(
        values="cate",
        index=sector_col,
        columns=size_col,
        aggfunc="mean",
    ).round(5)

    return pivot


def top_bottom_responders(
    df: pd.DataFrame,
    cate_values: np.ndarray,
    n: int = 20,
) -> Dict[str, pd.DataFrame]:
    """
    Identify the top-N and bottom-N responders by predicted CATE.
    """
    df_a = df.copy()
    df_a["cate"] = cate_values

    cols = ["sector", "log_market_cap", "momentum", "volatility", "cate"]
    if "true_cate" in df.columns:
        cols.append("true_cate")
    cols = [c for c in cols if c in df_a.columns]

    top = df_a.nlargest(n, "cate")[cols]
    bottom = df_a.nsmallest(n, "cate")[cols]

    return {"top_responders": top, "bottom_responders": bottom}
