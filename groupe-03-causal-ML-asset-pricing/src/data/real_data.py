"""
Real Market Data Fetcher
========================

Downloads real financial data using yfinance and constructs
a dataset suitable for causal analysis of earnings surprises
on stock returns.

Since real earnings-surprise timestamps require premium data sources
(e.g., WRDS, Compustat), we construct a proxy measure from price-based
signals around earnings dates.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None  # graceful fallback


# ---------------------------------------------------------------------------
# Default universe — S&P 500 representative sample (30 stocks, diverse sectors)
# ---------------------------------------------------------------------------
DEFAULT_TICKERS: Dict[str, List[str]] = {
    "Technology":              ["AAPL", "MSFT", "NVDA"],
    "Healthcare":              ["JNJ", "UNH", "PFE"],
    "Financials":              ["JPM", "BAC", "GS"],
    "Consumer Discretionary":  ["AMZN", "TSLA", "HD"],
    "Industrials":             ["CAT", "UPS", "HON"],
    "Energy":                  ["XOM", "CVX", "COP"],
    "Utilities":               ["NEE", "DUK", "SO"],
    "Materials":               ["LIN", "APD", "ECL"],
    "Communication Services":  ["GOOGL", "META", "DIS"],
    "Consumer Staples":        ["PG", "KO", "PEP"],
}


def _flatten_tickers(tickers_by_sector: Dict[str, List[str]]) -> List[str]:
    return [t for tickers in tickers_by_sector.values() for t in tickers]


def _ticker_to_sector(tickers_by_sector: Dict[str, List[str]]) -> Dict[str, str]:
    return {t: sector for sector, tickers in tickers_by_sector.items() for t in tickers}


def fetch_market_data(
    tickers_by_sector: Optional[Dict[str, List[str]]] = None,
    start: str = "2019-01-01",
    end: str = "2024-12-31",
    quarterly_windows: int = 63,  # ~1 quarter in trading days
) -> pd.DataFrame:
    """
    Fetch real market data and construct a quarterly panel.

    For each ticker, we:
    1. Download daily prices
    2. Slice into non-overlapping quarterly windows
    3. Compute features (momentum, volatility, returns) per window

    Parameters
    ----------
    tickers_by_sector : dict, optional
        Mapping sector → list of tickers. Defaults to DEFAULT_TICKERS.
    start, end : str
        Date range for download.
    quarterly_windows : int
        Trading days per window (~63 = 1 quarter).

    Returns
    -------
    pd.DataFrame
        Panel with one row per ticker-quarter.
    """
    if yf is None:
        raise ImportError("yfinance is required. Install with: pip install yfinance")

    if tickers_by_sector is None:
        tickers_by_sector = DEFAULT_TICKERS

    all_tickers = _flatten_tickers(tickers_by_sector)
    ticker_sector = _ticker_to_sector(tickers_by_sector)

    print(f"[DATA] Downloading data for {len(all_tickers)} tickers from {start} to {end}...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(all_tickers, start=start, end=end, group_by="ticker", progress=False)

    rows = []
    for ticker in all_tickers:
        try:
            if len(all_tickers) > 1:
                df_t = raw[ticker].dropna()
            else:
                df_t = raw.dropna()

            if len(df_t) < quarterly_windows * 2:
                continue

            closes = df_t["Close"].values
            volumes = df_t["Volume"].values
            n_windows = len(closes) // quarterly_windows

            for w in range(1, n_windows):  # start at 1 so we have a lookback
                end_idx = (w + 1) * quarterly_windows
                start_idx = w * quarterly_windows
                prev_start = (w - 1) * quarterly_windows

                if end_idx > len(closes):
                    break

                window_closes = closes[start_idx:end_idx]
                prev_closes = closes[prev_start:start_idx]
                window_volumes = volumes[start_idx:end_idx]

                # Quarterly return (outcome proxy)
                quarterly_return = (window_closes[-1] / window_closes[0]) - 1

                # Momentum: previous quarter return
                momentum = (prev_closes[-1] / prev_closes[0]) - 1

                # Volatility: annualised std of daily returns
                daily_rets = np.diff(window_closes) / window_closes[:-1]
                volatility = np.std(daily_rets) * np.sqrt(252)

                # Earnings surprise proxy: abnormal return in first week
                # (captures post-announcement drift)
                week_return = (window_closes[min(5, len(window_closes)-1)] / window_closes[0]) - 1
                # Compare to market expectation (momentum-adjusted)
                earnings_surprise_proxy = week_return - momentum / 4  # quarterly adj

                # Log market cap proxy (price × avg volume as liquidity proxy)
                avg_price = np.mean(window_closes)
                avg_volume = np.mean(window_volumes)
                log_market_cap_proxy = np.log(avg_price * avg_volume + 1)

                # Book-to-market proxy (inverse of price momentum, rough)
                btm_proxy = 1.0 / (1.0 + max(momentum, -0.5))

                rows.append({
                    "ticker": ticker,
                    "sector": ticker_sector[ticker],
                    "quarter_idx": w,
                    "log_market_cap": log_market_cap_proxy,
                    "book_to_market": btm_proxy,
                    "momentum": momentum,
                    "volatility": volatility,
                    "avg_volume": avg_volume,
                    "earnings_surprise": earnings_surprise_proxy,
                    "stock_return": quarterly_return,
                })
        except Exception as e:
            print(f"  [WARN] Skipping {ticker}: {e}")
            continue

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise ValueError("No data could be constructed. Check tickers and date range.")

    # Add derived columns
    df["market_cap_quintile"] = pd.qcut(
        df["log_market_cap"], q=5, labels=["Q1_Small", "Q2", "Q3", "Q4", "Q5_Large"],
        duplicates="drop",
    )

    # Standardize earnings_surprise
    df["earnings_surprise"] = (
        (df["earnings_surprise"] - df["earnings_surprise"].mean())
        / df["earnings_surprise"].std()
    )

    print(f"[DATA] Real dataset constructed: {len(df)} observations, {df['ticker'].nunique()} tickers")
    return df


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = fetch_market_data()
    print(df.describe().round(4))
    df.to_csv("data/real_market_data.csv", index=False)
    print("[DATA] Saved to data/real_market_data.csv")
