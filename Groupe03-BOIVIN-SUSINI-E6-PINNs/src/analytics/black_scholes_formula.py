"""
Analytical Black-Scholes formulas for European calls and puts.
Used as the ground-truth benchmark against the PINN solution.
"""

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma):
    """Compute d1 and d2 parameters. T is time-to-maturity (T-t)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_call_price(S, K, T, r, sigma):
    """
    Black-Scholes price for a European call option.

    Parameters
    ----------
    S     : spot price (scalar or array)
    K     : strike price
    T     : time to maturity in years  (T - t, so 0 = at expiry)
    r     : continuously-compounded risk-free rate
    sigma : annualised volatility

    Returns
    -------
    Call price, same shape as S.
    """
    S, T = np.asarray(S, dtype=float), np.asarray(T, dtype=float)
    price = np.where(
        T <= 0.0,
        np.maximum(S - K, 0.0),
        _call_positive_T(S, K, T, r, sigma),
    )
    return price


def _call_positive_T(S, K, T, r, sigma):
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, T, r, sigma):
    """
    Black-Scholes price for a European put option (via put-call parity).
    """
    return bs_call_price(S, K, T, r, sigma) - S + K * np.exp(-r * np.asarray(T, dtype=float))


def bs_delta(S, K, T, r, sigma, option="call"):
    """First derivative dV/dS (delta) for a European option."""
    S, T = np.asarray(S, dtype=float), np.asarray(T, dtype=float)
    d1, _ = _d1_d2(S, K, T, r, sigma)
    if option == "call":
        return np.where(T <= 0.0, (S > K).astype(float), norm.cdf(d1))
    return np.where(T <= 0.0, -(S < K).astype(float), norm.cdf(d1) - 1.0)


def bs_gamma(S, K, T, r, sigma):
    """Second derivative d²V/dS² (gamma) — identical for calls and puts."""
    S, T = np.asarray(S, dtype=float), np.asarray(T, dtype=float)
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return np.where(T <= 0.0, 0.0, norm.pdf(d1) / (S * sigma * np.sqrt(T)))


def bs_vega(S, K, T, r, sigma):
    """Sensitivity to volatility (vega), per unit volatility."""
    S, T = np.asarray(S, dtype=float), np.asarray(T, dtype=float)
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return np.where(T <= 0.0, 0.0, S * norm.pdf(d1) * np.sqrt(T))
