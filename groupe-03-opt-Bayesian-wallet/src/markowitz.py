"""
markowitz.py — Optimisation Markowitz classique (Mean-Variance) et frontière efficiente.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

from stats import compute_mean_returns, compute_cov_matrix, portfolio_performance


def market_cap_weights(tickers: list, prices: pd.DataFrame) -> pd.Series:
    """
    Approxime les poids de capitalisation boursière via Yahoo Finance.
    Fallback sur poids égaux si la market cap n'est pas disponible.
    """
    market_caps = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).fast_info
            market_caps[ticker] = getattr(info, "market_cap", None) or 1.0
        except Exception:
            market_caps[ticker] = 1.0

    caps = pd.Series(market_caps)
    weights = caps / caps.sum()
    return weights.reindex(tickers).fillna(1.0 / len(tickers))


def markowitz_weights(
    mu: pd.Series,
    cov: pd.DataFrame,
    target_return: float = None,
    risk_free_rate: float = 0.02,
    max_weight: float = 1.0,
) -> dict:
    """
    Optimisation Markowitz : maximise le ratio de Sharpe (ou minimise la variance
    pour un rendement cible donné).

    Args:
        mu            : rendements moyens annualisés
        cov           : matrice de covariance annualisée
        target_return : si fourni, minimise la variance pour ce rendement cible
        risk_free_rate: taux sans risque annualisé (défaut 2%)

    Returns:
        dict avec clés 'weights', 'return', 'volatility', 'sharpe'
    """
    n = len(mu)
    w0 = np.ones(n) / n
    bounds = [(0, max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    if target_return is not None:
        constraints.append(
            {"type": "eq", "fun": lambda w: w @ mu.values - target_return}
        )
        def objective(w):
            return w @ cov.values @ w
    else:
        def objective(w):
            ret = w @ mu.values
            vol = np.sqrt(w @ cov.values @ w)
            return -(ret - risk_free_rate) / vol

    result = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = pd.Series(result.x, index=mu.index)
    ret = float(weights @ mu)
    vol = float(np.sqrt(weights @ cov.values @ weights))
    sharpe = (ret - risk_free_rate) / vol

    return {"weights": weights, "return": ret, "volatility": vol, "sharpe": sharpe}


def efficient_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    n_points: int = 50,
    risk_free_rate: float = 0.02,
) -> pd.DataFrame:
    """
    Calcule la frontière efficiente en faisant varier le rendement cible.

    Returns:
        DataFrame avec colonnes ['return', 'volatility', 'sharpe']
    """
    target_returns = np.linspace(float(mu.min()), float(mu.max()), n_points)
    frontier = []
    for target in target_returns:
        try:
            res = markowitz_weights(mu, cov, target_return=target, risk_free_rate=risk_free_rate)
            frontier.append({
                "return": res["return"],
                "volatility": res["volatility"],
                "sharpe": res["sharpe"],
            })
        except Exception:
            continue
    return pd.DataFrame(frontier)


if __name__ == "__main__":
    from data import download_prices, compute_returns

    TICKERS = ["AAPL", "MSFT", "GOOGL"]
    prices = download_prices(TICKERS, "2022-01-01", "2024-01-01")
    returns = compute_returns(prices)
    mu = compute_mean_returns(returns)
    cov = compute_cov_matrix(returns)

    print("=== Poids de marché ===")
    mw = market_cap_weights(TICKERS, prices)
    for t, w in mw.items():
        print(f"  {t:6s} : {w:.2%}")

    print("\n=== Markowitz (max Sharpe) ===")
    result = markowitz_weights(mu, cov)
    for t, w in result["weights"].items():
        print(f"  {t:6s} : {w:.2%}")
    print(f"  Sharpe : {result['sharpe']:.2f}")
