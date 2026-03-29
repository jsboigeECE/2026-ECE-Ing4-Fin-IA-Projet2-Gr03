"""Data loading and caching utilities for SPY and VIX daily data."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_START = "2003-12-01"
DEFAULT_END = "2025-01-01"

SPY_CACHE = RAW_DIR / "spy_daily.csv"
VIX_CACHE = RAW_DIR / "vix_daily.csv"


def _normalize_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize the index to timezone-naive dates and sort it."""

    if frame.empty:
        raise ValueError("Downloaded dataframe is empty.")

    out = frame.copy()
    idx = pd.to_datetime(out.index, utc=True).tz_convert(None).normalize()
    out.index = idx
    out = out.sort_index()
    out = out.loc[~out.index.duplicated(keep="first")]
    return out


def _standardize_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Standardize yfinance output to a compact, CSV-friendly schema."""

    frame = _normalize_index(frame)

    # yfinance returns a MultiIndex for single-ticker downloads in many versions.
    # Keep the price level and discard the ticker level so the schema is stable.
    if isinstance(frame.columns, pd.MultiIndex):
        if "Ticker" in frame.columns.names and ticker in frame.columns.get_level_values("Ticker"):
            frame = frame.xs(ticker, level="Ticker", axis=1)
        else:
            frame.columns = frame.columns.get_level_values(0)

    # Normalize all column names to strings before any selection.
    frame.columns = [str(c).lower().replace(" ", "_") for c in frame.columns]

    # Ensure a single usable price column exists.
    if "adj_close" not in frame.columns and "close" in frame.columns:
        frame["adj_close"] = frame["close"]

    wanted = [c for c in ["open", "high", "low", "close", "adj_close", "volume"] if c in frame.columns]
    frame = frame[wanted].copy()
    frame.index.name = "date"
    frame["ticker"] = ticker
    return frame


def _load_cached_csv(path: Path, expected_cols: tuple[str, ...]) -> pd.DataFrame | None:
    """Load a cached CSV only if it contains the expected columns."""

    if not path.exists():
        return None

    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df.index = pd.to_datetime(df.index).normalize()
    if not set(expected_cols).issubset(df.columns):
        return None
    return df


def _download_ticker(ticker: str, start: str = DEFAULT_START, end: str = DEFAULT_END) -> pd.DataFrame:
    """Download a daily ticker series from yfinance."""

    data = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=True,
    )

    if data.empty:
        raise RuntimeError(f"No data returned for ticker {ticker!r}.")

    return _standardize_frame(data, ticker)


def load_or_download_spy(force: bool = False, start: str = DEFAULT_START, end: str = DEFAULT_END) -> pd.DataFrame:
    """Load cached SPY data or download it if missing."""

    if not force:
        cached = _load_cached_csv(SPY_CACHE, ("adj_close", "close"))
        if cached is not None:
            return cached

    df = _download_ticker("SPY", start=start, end=end)
    df.to_csv(SPY_CACHE)
    return df


def load_or_download_vix(force: bool = False, start: str = DEFAULT_START, end: str = DEFAULT_END) -> pd.DataFrame:
    """Load cached VIX data or download it if missing."""

    if not force:
        cached = _load_cached_csv(VIX_CACHE, ("close", "adj_close"))
        if cached is not None:
            return cached

    df = _download_ticker("^VIX", start=start, end=end)
    df.to_csv(VIX_CACHE)
    return df


def load_aligned_market_data(force: bool = False, start: str = DEFAULT_START, end: str = DEFAULT_END) -> pd.DataFrame:
    """Load SPY and VIX, align on common trading dates, and return a merged frame."""

    spy = load_or_download_spy(force=force, start=start, end=end)
    vix = load_or_download_vix(force=force, start=start, end=end)

    aligned = pd.DataFrame(index=spy.index.intersection(vix.index)).sort_index()

    if "adj_close" in spy.columns:
        aligned["spy_adj_close"] = spy.reindex(aligned.index)["adj_close"]
    elif "close" in spy.columns:
        aligned["spy_adj_close"] = spy.reindex(aligned.index)["close"]
    else:
        raise RuntimeError("SPY cache missing both adj_close and close columns.")

    if "close" in vix.columns:
        aligned["vix_close"] = vix.reindex(aligned.index)["close"]
    elif "adj_close" in vix.columns:
        aligned["vix_close"] = vix.reindex(aligned.index)["adj_close"]
    else:
        raise RuntimeError("VIX cache missing both close and adj_close columns.")

    aligned = aligned.sort_index()
    if aligned[["spy_adj_close", "vix_close"]].isna().any().any():
        raise RuntimeError("Aligned market data contains missing core prices after intersection.")

    return aligned[["spy_adj_close", "vix_close"]].copy()


def load_cached_paths() -> Tuple[Path, Path]:
    """Return the expected raw cache paths."""

    return SPY_CACHE, VIX_CACHE

