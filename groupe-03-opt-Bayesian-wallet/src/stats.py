"""
stats.py — Calculs statistiques : rendements moyens, covariance, performance.
"""

import numpy as np
import pandas as pd


def compute_mean_returns(returns: pd.DataFrame, annualize: bool = True) -> pd.Series:
    """
    Calcule les rendements moyens (annualisés si demandé, base 252 jours).
    """
    mu = returns.mean()
    if annualize:
        mu = mu * 252
    return mu


def compute_cov_matrix(returns: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
    """
    Calcule la matrice de covariance (annualisée si demandé, base 252 jours).
    """
    cov = returns.cov()
    if annualize:
        cov = cov * 252
    return cov


def portfolio_performance(
    weights: pd.Series,
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float = 0.02,
) -> dict:
    """
    Calcule rendement, volatilité et Sharpe d'un portefeuille donné.
    """
    ret = float(weights @ mu)
    vol = float(np.sqrt(weights @ cov.values @ weights))
    sharpe = (ret - risk_free_rate) / vol
    return {"return": ret, "volatility": vol, "sharpe": sharpe}


if __name__ == "__main__":
    from data import download_prices, compute_returns

    prices = download_prices(["AAPL", "MSFT", "GOOGL"], "2022-01-01", "2024-01-01")
    returns = compute_returns(prices)
    mu = compute_mean_returns(returns)
    cov = compute_cov_matrix(returns)

    print("Rendements annualisés :")
    for ticker, val in mu.items():
        print(f"  {ticker:6s} : {val:+.2%}")

    print("\nVolatilités annualisées :")
    for ticker in cov.columns:
        print(f"  {ticker:6s} : {cov.loc[ticker, ticker]**0.5:.2%}")
