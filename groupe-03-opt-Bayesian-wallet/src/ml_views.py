"""
ml_views.py — Génération automatique de views par Machine Learning (Momentum).
"""

import numpy as np
import pandas as pd


def generate_momentum_views(
    prices: pd.DataFrame,
    lookback: int = 63,
    threshold: float = 0.0,
    view_scale: float = 0.10,
) -> list:
    """
    Génère automatiquement des views à partir du momentum des prix.

    Logique :
        - On calcule le rendement sur les 'lookback' derniers jours (≈ 3 mois)
        - Si le rendement > threshold → view positive
        - Si le rendement < -threshold → view négative

    C'est le principe du "trend following" : les actifs qui ont bien performé
    récemment tendent à continuer sur leur lancée.

    Args:
        prices     : DataFrame de prix
        lookback   : nombre de jours pour le momentum (défaut 63 = ~3 mois)
        threshold  : seuil minimum pour générer une view (défaut 0)
        view_scale : magnitude de la view générée (défaut 10%)

    Returns:
        liste de views au format attendu par build_views()
    """
    recent = prices.iloc[-lookback:]
    momentum = (recent.iloc[-1] / recent.iloc[0]) - 1

    views = []
    for ticker in prices.columns:
        m = float(momentum[ticker])
        if m > threshold:
            views.append({
                "type": "absolute",
                "asset": ticker,
                "return": view_scale * np.sign(m) * min(abs(m), 1.0),
            })
        elif m < -threshold:
            views.append({
                "type": "absolute",
                "asset": ticker,
                "return": -view_scale * min(abs(m), 1.0),
            })
    return views


if __name__ == "__main__":
    from data import download_prices

    TICKERS = ["AAPL", "MSFT", "GOOGL"]
    prices = download_prices(TICKERS, "2022-01-01", "2024-01-01")
    views = generate_momentum_views(prices)

    print("=== Views générées par Momentum ===")
    for v in views:
        print(f"  {v['asset']:6s} : view {v['return']:+.2%}")
