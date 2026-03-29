"""
data.py — Récupération des données et calcul des rendements.
"""

import numpy as np
import pandas as pd
import yfinance as yf


def download_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    """
    Télécharge les prix de clôture ajustés depuis Yahoo Finance.

    Args:
        tickers : liste de symboles boursiers, ex. ['AAPL', 'MSFT', 'GOOGL']
        start   : date de début au format 'YYYY-MM-DD'
        end     : date de fin   au format 'YYYY-MM-DD'

    Returns:
        DataFrame de prix de clôture ajustés, colonnes = tickers, index = dates.
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers
    prices.dropna(how="all", inplace=True)
    return prices


def compute_returns(prices: pd.DataFrame, freq: str = "daily") -> pd.DataFrame:
    """
    Calcule les rendements logarithmiques à partir des prix.

    Args:
        prices : DataFrame de prix
        freq   : 'daily' (défaut), 'weekly' ou 'monthly'

    Returns:
        DataFrame de rendements log.
    """
    if freq == "weekly":
        prices = prices.resample("W").last()
    elif freq == "monthly":
        prices = prices.resample("ME").last()
    return np.log(prices / prices.shift(1)).dropna()


if __name__ == "__main__":
    prices = download_prices(["AAPL", "MSFT", "GOOGL"], "2022-01-01", "2024-01-01")
    returns = compute_returns(prices)
    print(f"Prix    : {prices.shape[0]} jours x {prices.shape[1]} actifs")
    print(f"Rendements : {returns.shape[0]} jours")
    print(prices.tail(3))
