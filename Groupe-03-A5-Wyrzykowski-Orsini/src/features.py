import numpy as np
import pandas as pd


def create_features(df):
    """
    Crée les variables utiles pour prédire le rendement du lendemain.
    """

    data = df.copy()

    # Rendement journalier
    data["return"] = data["Close"].pct_change()

    # Features simples basées sur le passé
    data["return_lag_1"] = data["return"].shift(1)
    data["return_lag_2"] = data["return"].shift(2)
    data["return_lag_3"] = data["return"].shift(3)

    # Moyennes mobiles
    data["ma_5"] = data["Close"].rolling(5).mean()
    data["ma_10"] = data["Close"].rolling(10).mean()

    # Volatilité glissante
    data["volatility_5"] = data["return"].rolling(5).std()
    data["volatility_10"] = data["return"].rolling(10).std()

    # Momentum simple
    data["momentum_5"] = data["Close"] / data["Close"].shift(5) - 1

    # Cible = rendement du lendemain
    data["target"] = data["return"].shift(-1)

    data.dropna(inplace=True)

    feature_cols = [
        "return_lag_1",
        "return_lag_2",
        "return_lag_3",
        "ma_5",
        "ma_10",
        "volatility_5",
        "volatility_10",
        "momentum_5",
        "Volume",
    ]

    X = data[feature_cols].copy()
    y = data["target"].copy()

    return data, X, y, feature_cols
        