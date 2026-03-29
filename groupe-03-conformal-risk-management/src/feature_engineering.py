"""Feature engineering for the SPY/VIX risk management dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SPLITS: Mapping[str, Tuple[str, str]] = {
    "train": ("2004-01-02", "2014-12-31"),
    "calibration": ("2015-01-02", "2017-12-31"),
    "validation": ("2018-01-02", "2019-12-31"),
    "test": ("2020-01-02", "2024-12-31"),
}

FEATURE_COLUMNS = [
    "lag_return_1",
    "lag_return_2",
    "lag_return_5",
    "rolling_20_volatility",
    "lag_vix_1",
    "rolling_5_mean_return",
    "lag_squared_return",
]

TARGET_COLUMN = "target_next_return"


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute simple daily log returns from a price series."""

    prices = prices.astype(float)
    returns = np.log(prices / prices.shift(1))
    returns.name = "log_return"
    return returns


def build_feature_frame(aligned_market_data: pd.DataFrame) -> pd.DataFrame:
    """Build the final, leakage-safe modeling frame."""

    if "spy_adj_close" in aligned_market_data.columns:
        spy_price = aligned_market_data["spy_adj_close"]
    elif "spy_close" in aligned_market_data.columns:
        spy_price = aligned_market_data["spy_close"]
    else:
        raise KeyError("Expected SPY price column not found.")

    if "vix_close" in aligned_market_data.columns:
        vix_price = aligned_market_data["vix_close"]
    elif "vix_adj_close" in aligned_market_data.columns:
        vix_price = aligned_market_data["vix_adj_close"]
    else:
        raise KeyError("Expected VIX price column not found.")

    frame = pd.DataFrame(index=aligned_market_data.index.copy())
    frame.index.name = "date"

    frame["spy_adj_close"] = spy_price.astype(float)
    frame["vix_close"] = vix_price.astype(float)

    frame["spy_log_return"] = compute_log_returns(frame["spy_adj_close"])

    # ------------------------------------------------------------------
    # Feature naming convention (important for Phase 2 developers):
    # lag_k = the k-th most recent return available at prediction time t+1.
    # lag_1 = r_t   (most recent — no shift; r_t is known at EOD of day t)
    # lag_2 = r_{t-1} (shift 1)
    # lag_5 = r_{t-4} (shift 4)
    # None of these are future-leaking: all use only EOD-of-day-t prices.
    # Do NOT apply additional .shift() to these columns in downstream code.
    # ------------------------------------------------------------------
    frame["lag_return_1"] = frame["spy_log_return"]           # r_t, no shift — intentional
    frame["lag_return_2"] = frame["spy_log_return"].shift(1)  # r_{t-1}
    frame["lag_return_5"] = frame["spy_log_return"].shift(4)  # r_{t-4}
    frame["rolling_20_volatility"] = frame["spy_log_return"].rolling(window=20, min_periods=20).std()
    frame["lag_vix_1"] = frame["vix_close"].shift(1)          # VIX_{t-1}, correct lag
    frame["rolling_5_mean_return"] = frame["spy_log_return"].rolling(window=5, min_periods=5).mean()
    # lag_squared_return = r_t^2: current-day squared return (known at EOD). No shift — intentional.
    frame["lag_squared_return"] = frame["spy_log_return"] ** 2

    # Next-day target.
    frame[TARGET_COLUMN] = frame["spy_log_return"].shift(-1)

    final_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    usable = frame[final_cols].dropna().copy()

    # Clip to project start. Two warmup rows from Dec 2003 survive dropna but
    # fall outside all defined splits. Exclude them defensively here so that
    # the returned frame always begins at 2004-01-02 and downstream code
    # does not need to remember to call split_by_date_range().
    project_start = pd.Timestamp("2004-01-02")
    usable = usable[usable.index >= project_start].copy()

    usable.index.name = "date"
    return usable


def save_processed_dataset(feature_frame: pd.DataFrame, output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Save features, targets, and split metadata."""

    out_dir = output_dir or PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    features = feature_frame[FEATURE_COLUMNS].copy()
    targets = feature_frame[[TARGET_COLUMN]].copy()

    features_path = out_dir / "features.csv"
    targets_path = out_dir / "targets.csv"
    splits_path = out_dir / "splits.json"

    features.to_csv(features_path, index=True, index_label="date")
    targets.to_csv(targets_path, index=True, index_label="date")

    splits_payload = {
        name: {"start": start, "end": end}
        for name, (start, end) in SPLITS.items()
    }
    splits_path.write_text(json.dumps(splits_payload, indent=2), encoding="utf-8")

    return {
        "features": features_path,
        "targets": targets_path,
        "splits": splits_path,
    }


def split_by_date_range(frame: pd.DataFrame, splits: Mapping[str, Tuple[str, str]] = SPLITS) -> Dict[str, pd.DataFrame]:
    """Return chronological subsets defined by the configured date ranges."""

    subsets: Dict[str, pd.DataFrame] = {}
    for name, (start, end) in splits.items():
        subsets[name] = frame.loc[pd.Timestamp(start) : pd.Timestamp(end)].copy()
    return subsets


def _source_windows_for_date(index: pd.DatetimeIndex, date: pd.Timestamp) -> Dict[str, object]:
    """Describe the source rows used for each engineered feature."""

    pos = index.get_loc(date)
    if isinstance(pos, slice):
        raise ValueError(f"Date {date} resolves to a slice, expected a unique position.")

    source_info: Dict[str, object] = {}
    source_info["lag_return_1"] = index[pos]
    source_info["lag_return_2"] = index[pos - 1] if pos - 1 >= 0 else None
    source_info["lag_return_5"] = index[pos - 4] if pos - 4 >= 0 else None
    source_info["rolling_20_volatility"] = (
        index[pos - 19],
        index[pos],
    ) if pos - 19 >= 0 else None
    source_info["lag_vix_1"] = index[pos - 1] if pos - 1 >= 0 else None
    source_info["rolling_5_mean_return"] = (
        index[pos - 4],
        index[pos],
    ) if pos - 4 >= 0 else None
    source_info["lag_squared_return"] = index[pos]
    source_info["target_next_return"] = index[pos + 1] if pos + 1 < len(index) else None
    return source_info


def leakage_audit(feature_frame: pd.DataFrame, raw_market_data: pd.DataFrame, audit_dates: Optional[Sequence[str]] = None) -> Tuple[pd.DataFrame, bool]:
    """Run an explicit leakage audit on three manually selected dates."""

    if audit_dates is None:
        audit_dates = ("2016-06-15", "2019-06-14", "2023-06-15")

    if "spy_adj_close" in raw_market_data.columns:
        spy_price = raw_market_data["spy_adj_close"]
    elif "spy_close" in raw_market_data.columns:
        spy_price = raw_market_data["spy_close"]
    else:
        raise KeyError("Expected SPY price column not found in raw market data.")

    if "vix_close" in raw_market_data.columns:
        vix_price = raw_market_data["vix_close"]
    elif "vix_adj_close" in raw_market_data.columns:
        vix_price = raw_market_data["vix_adj_close"]
    else:
        raise KeyError("Expected VIX price column not found in raw market data.")

    spy_returns = compute_log_returns(spy_price)
    vix_series = vix_price.astype(float)

    audit_index = feature_frame.index
    reports: List[Dict[str, object]] = []

    for requested in audit_dates:
        requested_ts = pd.Timestamp(requested)
        pos = audit_index.get_indexer([requested_ts], method="pad")[0]
        if pos == -1:
            raise ValueError(f"Requested audit date {requested} is earlier than the first usable row.")

        row_date = audit_index[pos]
        raw_pos = raw_market_data.index.get_loc(row_date)
        if isinstance(raw_pos, slice):
            raise ValueError(f"Raw market date {row_date} is ambiguous.")

        # Recompute the relevant values directly from raw data.
        expected = {
            "lag_return_1": spy_returns.iloc[raw_pos],
            "lag_return_2": spy_returns.iloc[raw_pos - 1] if raw_pos - 1 >= 0 else np.nan,
            "lag_return_5": spy_returns.iloc[raw_pos - 4] if raw_pos - 4 >= 0 else np.nan,
            "rolling_20_volatility": spy_returns.iloc[raw_pos - 19 : raw_pos + 1].std() if raw_pos - 19 >= 0 else np.nan,
            "lag_vix_1": vix_series.iloc[raw_pos - 1] if raw_pos - 1 >= 0 else np.nan,
            "rolling_5_mean_return": spy_returns.iloc[raw_pos - 4 : raw_pos + 1].mean() if raw_pos - 4 >= 0 else np.nan,
            "lag_squared_return": spy_returns.iloc[raw_pos] ** 2,
            "target_next_return": spy_returns.iloc[raw_pos + 1] if raw_pos + 1 < len(spy_returns) else np.nan,
        }

        actual = feature_frame.loc[row_date, FEATURE_COLUMNS + [TARGET_COLUMN]]
        comparisons = {}
        for key, value in expected.items():
            actual_value = actual[key]
            if pd.isna(value) and pd.isna(actual_value):
                comparisons[key] = True
            else:
                comparisons[key] = bool(np.isclose(float(actual_value), float(value), rtol=1e-10, atol=1e-12))

        source_windows = _source_windows_for_date(raw_market_data.index, row_date)

        feature_sources_ok = True
        for name, source in source_windows.items():
            if name == "target_next_return" or source is None:
                continue
            if isinstance(source, tuple):
                feature_sources_ok = feature_sources_ok and all(part <= row_date for part in source)
            else:
                feature_sources_ok = feature_sources_ok and (source <= row_date)

        target_source_ok = source_windows["target_next_return"] is not None and source_windows["target_next_return"] > row_date

        passed = feature_sources_ok and target_source_ok and all(comparisons.values())

        reports.append(
            {
                "requested_date": requested,
                "audit_row_date": row_date.strftime("%Y-%m-%d"),
                "lag_return_1_source": source_windows["lag_return_1"].strftime("%Y-%m-%d") if source_windows["lag_return_1"] is not None else None,
                "lag_return_2_source": source_windows["lag_return_2"].strftime("%Y-%m-%d") if source_windows["lag_return_2"] is not None else None,
                "rolling_20_window_start": source_windows["rolling_20_volatility"][0].strftime("%Y-%m-%d") if source_windows["rolling_20_volatility"] is not None else None,
                "rolling_20_window_end": source_windows["rolling_20_volatility"][1].strftime("%Y-%m-%d") if source_windows["rolling_20_volatility"] is not None else None,
                "lag_vix_1_source": source_windows["lag_vix_1"].strftime("%Y-%m-%d") if source_windows["lag_vix_1"] is not None else None,
                "target_source": source_windows["target_next_return"].strftime("%Y-%m-%d") if source_windows["target_next_return"] is not None else None,
                "all_value_checks_passed": all(comparisons.values()),
                "feature_source_checks_passed": feature_sources_ok,
                "target_source_check_passed": target_source_ok,
                "passed": passed,
            }
        )

    audit_df = pd.DataFrame(reports)
    overall_passed = bool(audit_df["passed"].all())
    return audit_df, overall_passed

