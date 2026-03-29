"""
backtest.py — Backtesting de la stratégie BL et analyse de sensibilité aux views.
"""

import numpy as np
import pandas as pd

from data import compute_returns
from stats import compute_mean_returns, compute_cov_matrix
from black_litterman import (
    compute_equilibrium_returns,
    build_views,
    compute_omega,
    black_litterman_posterior,
    optimize_bl_portfolio,
)


def backtest_bl(
    prices: pd.DataFrame,
    market_weights: pd.Series,
    views_fn,
    train_window: int = 252,
    rebalance_freq: int = 21,
    risk_free_rate: float = 0.02,
    tau: float = 0.05,
) -> pd.DataFrame:
    """
    Backtest de la stratégie Black-Litterman sur données historiques.

    Fonctionnement (rolling window) :
        - On avance dans le temps tous les 'rebalance_freq' jours
        - À chaque pas : on estime le modèle sur les 'train_window' jours précédents
        - On calcule les nouveaux poids BL
        - On mesure la performance sur la période suivante

    Args:
        prices         : DataFrame de prix historiques
        market_weights : poids du portefeuille de marché
        views_fn       : fonction qui génère les views à partir des prix
        train_window   : jours d'historique pour l'estimation (défaut 252 = 1 an)
        rebalance_freq : fréquence de rééquilibrage en jours (défaut 21 = ~1 mois)
        risk_free_rate : taux sans risque annualisé
        tau            : paramètre tau du modèle BL

    Returns:
        DataFrame avec ['date', 'portfolio_return', 'cumulative_return']
    """
    results = []
    n = len(prices)

    for start in range(train_window, n - rebalance_freq, rebalance_freq):
        train_prices = prices.iloc[start - train_window: start]
        future_prices = prices.iloc[start: start + rebalance_freq]

        try:
            returns = compute_returns(train_prices)
            mu = compute_mean_returns(returns)
            cov = compute_cov_matrix(returns)
            mw = market_weights.reindex(prices.columns).fillna(0)
            mw = mw / mw.sum()
            pi = compute_equilibrium_returns(cov, mw)

            views = views_fn(train_prices)
            if not views:
                weights = mw
            else:
                P, Q = build_views(list(prices.columns), views)
                omega = compute_omega(P, cov, tau=tau, confidences=[0.6] * len(views))
                mu_bl, cov_bl = black_litterman_posterior(pi, cov, P, Q, omega, tau=tau)
                result = optimize_bl_portfolio(mu_bl, cov_bl, risk_free_rate=risk_free_rate)
                weights = result["weights"]

            future_returns = compute_returns(future_prices)
            period_return = float((future_returns * weights).sum(axis=1).sum())
            results.append({
                "date": future_prices.index[-1],
                "portfolio_return": period_return,
                "weights": weights.to_dict(),
            })
        except Exception:
            continue

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df["cumulative_return"] = (1 + df["portfolio_return"]).cumprod() - 1
    return df


def sensitivity_analysis(
    pi: pd.Series,
    cov: pd.DataFrame,
    P: np.ndarray,
    Q_base: np.ndarray,
    market_weights: pd.Series,
    perturbations: list = None,
    tau: float = 0.05,
) -> pd.DataFrame:
    """
    Analyse comment les poids optimaux changent quand on modifie les views.

    On fait varier Q autour de sa valeur de base et on observe l'impact sur les poids.

    Args:
        perturbations : liste de multiplicateurs à tester (ex: [0.5, 1.0, 1.5, 2.0])

    Returns:
        DataFrame avec les poids et le Sharpe pour chaque niveau de perturbation.
    """
    if perturbations is None:
        perturbations = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    rows = []
    for scale in perturbations:
        Q_perturbed = Q_base * scale
        omega = compute_omega(P, cov, tau=tau)
        mu_bl, cov_bl = black_litterman_posterior(pi, cov, P, Q_perturbed, omega, tau=tau)
        result = optimize_bl_portfolio(mu_bl, cov_bl)
        row = {"view_scale": scale}
        row.update(result["weights"].to_dict())
        row["sharpe"] = result["sharpe"]
        rows.append(row)

    return pd.DataFrame(rows).set_index("view_scale")


if __name__ == "__main__":
    from data import download_prices
    from markowitz import market_cap_weights
    from ml_views import generate_momentum_views

    TICKERS = ["AAPL", "MSFT", "GOOGL"]
    prices = download_prices(TICKERS, "2020-01-01", "2024-01-01")
    mw = market_cap_weights(TICKERS, prices)

    print("=== Backtesting BL (momentum views) ===")
    bt = backtest_bl(prices, mw, generate_momentum_views)
    if not bt.empty:
        print(f"  Périodes testées    : {len(bt)}")
        print(f"  Rendement cumulé    : {bt['cumulative_return'].iloc[-1]:+.2%}")
        print(f"  Rendement moyen/mois: {bt['portfolio_return'].mean():+.2%}")
    else:
        print("  Pas assez de données pour le backtest.")
