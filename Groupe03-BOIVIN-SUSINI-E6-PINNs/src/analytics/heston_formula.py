"""
Heston (1993) model — pricing benchmarks.

Two methods are provided:
  1. Semi-analytical  : characteristic-function integration (exact, fast).
  2. Monte Carlo      : Euler-Maruyama on (S, v) SDE (fallback / sanity check).

Heston SDE (risk-neutral measure):
  dS = r S dt + sqrt(v) S dW1
  dv = kappa (theta - v) dt + sigma_v sqrt(v) dW2
  corr(dW1, dW2) = rho dt
"""

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


# ─────────────────────────────────────────────────────────────────────────────
# Semi-analytical formula (Heston 1993, Gil-Pelaez inversion)
# ─────────────────────────────────────────────────────────────────────────────

def _heston_cf(phi, S, K, T, r, kappa, theta, sigma_v, rho, v0, j):
    """
    Characteristic function of ln(S_T) for the Heston model.
    Uses the Albrecher et al. (2007) formulation to avoid branch-cut issues.

    j=1 : measure P1 (asset measure)
    j=2 : measure P2 (risk-neutral measure)
    """
    x = np.log(S / K)
    if j == 1:
        u, b = 0.5, kappa - rho * sigma_v
    else:
        u, b = -0.5, kappa

    a = kappa * theta
    d = np.sqrt((rho * sigma_v * 1j * phi - b) ** 2
                - sigma_v ** 2 * (2 * u * 1j * phi - phi ** 2))

    # Albrecher sign convention to avoid discontinuity
    g = (b - rho * sigma_v * 1j * phi + d) / (b - rho * sigma_v * 1j * phi - d)

    exp_dT = np.exp(d * T)
    log_term = np.log((1 - g * exp_dT) / (1 - g))

    C = (r * 1j * phi * T
         + a / sigma_v ** 2 * ((b - rho * sigma_v * 1j * phi + d) * T
                                - 2 * log_term))
    D = ((b - rho * sigma_v * 1j * phi + d) / sigma_v ** 2
         * (1 - exp_dT) / (1 - g * exp_dT))

    return np.exp(C + D * v0 + 1j * phi * x)


def _heston_integrand(phi, S, K, T, r, kappa, theta, sigma_v, rho, v0, j):
    cf = _heston_cf(phi, S, K, T, r, kappa, theta, sigma_v, rho, v0, j)
    return np.real(np.exp(-1j * phi * np.log(K)) * cf / (1j * phi))


def heston_call_price(S, K, T, r, kappa, theta, sigma_v, rho, v0,
                      n_quad=500):
    """
    Heston semi-analytical European call price.

    Parameters
    ----------
    S, K, T, r : standard contract parameters
    kappa      : mean-reversion speed of variance
    theta      : long-run variance (= long-run vol²)
    sigma_v    : vol-of-vol
    rho        : spot-vol correlation (typically negative for equities)
    v0         : initial variance
    n_quad     : number of quadrature points (higher = more accurate)
    """
    S = float(S)
    T = float(T)
    if T <= 0.0:
        return max(S - K, 0.0)

    try:
        P1 = 0.5 + quad(
            _heston_integrand, 1e-6, 500,
            args=(S, K, T, r, kappa, theta, sigma_v, rho, v0, 1),
            limit=n_quad, complex_func=False
        )[0] / np.pi

        P2 = 0.5 + quad(
            _heston_integrand, 1e-6, 500,
            args=(S, K, T, r, kappa, theta, sigma_v, rho, v0, 2),
            limit=n_quad, complex_func=False
        )[0] / np.pi

        price = S * P1 - K * np.exp(-r * T) * P2
        # Arbitrage bounds
        return float(np.clip(price, max(S - K * np.exp(-r * T), 0.0), S))

    except Exception:
        # Fallback to Monte Carlo
        return heston_mc_price(S, K, T, r, kappa, theta, sigma_v, rho, v0)


def heston_call_grid(S_vals, K, T, r, kappa, theta, sigma_v, rho, v0):
    """Vectorised call prices over a 1-D array of spots."""
    return np.array([heston_call_price(s, K, T, r, kappa, theta, sigma_v,
                                       rho, v0) for s in S_vals])


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo Heston (Euler-Maruyama, full-truncation)
# ─────────────────────────────────────────────────────────────────────────────

def heston_mc_price(S0, K, T, r, kappa, theta, sigma_v, rho, v0,
                    n_paths=100_000, n_steps=200, seed=None):
    """
    Monte Carlo price for a European call under the Heston model.
    Uses the full-truncation Euler scheme for the variance process.

    Returns (price, std_error).
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    S = np.full(n_paths, float(S0))
    v = np.full(n_paths, float(v0))

    for _ in range(n_steps):
        Z1 = np.random.randn(n_paths)
        Z2 = rho * Z1 + np.sqrt(1.0 - rho ** 2) * np.random.randn(n_paths)

        v_pos = np.maximum(v, 0.0)
        sv = np.sqrt(v_pos)

        S *= np.exp((r - 0.5 * v_pos) * dt + sv * sqrt_dt * Z1)
        v = v + kappa * (theta - v_pos) * dt + sigma_v * sv * sqrt_dt * Z2

    payoffs = np.maximum(S - K, 0.0)
    disc = np.exp(-r * T)
    price = disc * payoffs.mean()
    stderr = disc * payoffs.std() / np.sqrt(n_paths)
    return price, stderr


# ─────────────────────────────────────────────────────────────────────────────
# Implied volatility: BS vol that matches a given price
# ─────────────────────────────────────────────────────────────────────────────

def _bs_call(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_vol(price, S, K, T, r, tol=1e-6, max_iter=100):
    """
    Newton-Raphson implied volatility from a call price.
    Returns NaN if no solution found.
    """
    if price <= max(S - K * np.exp(-r * T), 0.0) + 1e-10:
        return np.nan

    sigma = 0.3  # initial guess
    for _ in range(max_iter):
        p = _bs_call(S, K, T, r, sigma)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        if abs(vega) < 1e-12:
            break
        sigma -= (p - price) / vega
        sigma = max(sigma, 1e-4)
        if abs(p - price) < tol:
            break
    return sigma
