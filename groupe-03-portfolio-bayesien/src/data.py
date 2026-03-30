"""
Téléchargement et préparation des données — Yahoo Finance
"""
import os
import numpy as np
import pandas as pd

from src.config import TICKERS, TRAIN_START, TEST_END, RISK_FREE_RATE, DATA_CACHE, MARKET_CAPS


def download_prices(start: str = TRAIN_START, end: str = TEST_END,
                    cache: str = DATA_CACHE) -> pd.DataFrame:
    """Télécharge (ou charge depuis le cache) les prix ajustés de clôture."""
    if os.path.exists(cache):
        print(f"[data] Chargement depuis le cache : {cache}")
        prices = pd.read_csv(cache, index_col=0, parse_dates=True)
        # Vérifier que tous les tickers sont présents
        missing = [t for t in TICKERS if t not in prices.columns]
        if not missing:
            return prices[TICKERS]

    print("[data] Téléchargement Yahoo Finance…")
    try:
        import yfinance as yf
        raw = yf.download(TICKERS, start=start, end=end,
                          auto_adjust=True, progress=False)
        prices = raw["Close"][TICKERS].dropna()
    except Exception as e:
        print(f"[data] ⚠ Téléchargement échoué ({e}). Génération de données synthétiques.")
        prices = _synthetic_prices(start, end)

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    prices.to_csv(cache)
    print(f"[data] {len(prices)} jours téléchargés pour {len(TICKERS)} actifs.")
    return prices


def _synthetic_prices(start: str, end: str) -> pd.DataFrame:
    """Génère des prix synthétiques réalistes si Yahoo Finance est indisponible."""
    np.random.seed(42)
    dates = pd.bdate_range(start, end)
    n = len(dates)

    # Paramètres annuels approximatifs (mu, sigma) par actif
    params = {
        "AAPL":  (0.22, 0.28), "MSFT":  (0.20, 0.25),
        "GOOGL": (0.16, 0.26), "NVDA":  (0.35, 0.50),
        "JPM":   (0.12, 0.22), "BRK-B": (0.11, 0.16),
        "JNJ":   (0.07, 0.14), "UNH":   (0.18, 0.20),
        "AMZN":  (0.18, 0.30), "XOM":   (0.10, 0.25),
    }

    # Correlation structure légère entre secteurs
    prices_dict = {}
    dt = 1 / 252
    for ticker, (mu, sigma) in params.items():
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt)
        shocks = np.random.normal(drift, diffusion, n)
        prices_dict[ticker] = 100 * np.exp(np.cumsum(shocks))

    return pd.DataFrame(prices_dict, index=dates)


# ─── Statistiques ─────────────────────────────────────────────────────────────

def compute_returns(prices: pd.DataFrame, freq: str = "daily") -> pd.DataFrame:
    """Calcule les rendements logarithmiques."""
    log_rets = np.log(prices / prices.shift(1)).dropna()
    return log_rets


def annualize(log_rets: pd.DataFrame) -> tuple:
    """
    Retourne (mu_annual, sigma_annual, cov_annual) à partir des log-rendements quotidiens.
    Convention : 252 jours de trading par an.
    """
    mu_daily  = log_rets.mean().values           # (n,)
    cov_daily = log_rets.cov().values             # (n,n)
    mu_ann    = mu_daily  * 252                   # annualiser
    cov_ann   = cov_daily * 252
    return mu_ann, cov_ann


def market_weights(tickers=TICKERS) -> np.ndarray:
    """Poids de marché normalisés par capitalisation boursière."""
    caps = np.array([MARKET_CAPS[t] for t in tickers], dtype=float)
    return caps / caps.sum()


def split_train_test(prices: pd.DataFrame,
                     train_end: str = "2022-12-31",
                     test_start: str = "2023-01-01") -> tuple:
    """Sépare train / test."""
    train = prices.loc[:train_end]
    test  = prices.loc[test_start:]
    return train, test


def prepare_all(train_prices: pd.DataFrame) -> dict:
    """Calcule toutes les statistiques nécessaires à partir des prix d'entraînement."""
    log_rets = compute_returns(train_prices)
    mu, cov  = annualize(log_rets)
    w_mkt    = market_weights(list(train_prices.columns))
    return {
        "log_rets": log_rets,
        "mu":       mu,
        "cov":      cov,
        "w_mkt":    w_mkt,
        "tickers":  list(train_prices.columns),
        "rf":       RISK_FREE_RATE,
    }
