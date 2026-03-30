import yfinance as yf
import pandas as pd


def load_data(ticker="SPY", start="2018-01-01", end="2025-01-01"):
    """
    Télécharge les données boursières avec yfinance.
    On garde seulement les colonnes utiles.
    """

    df = yf.download(ticker, start=start, end=end, auto_adjust=True)

    if df.empty:
        raise ValueError("Aucune donnée téléchargée. Vérifie internet ou le ticker.")

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)

    return df