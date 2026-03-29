"""
Tests unitaires pour black_litterman.py
"""

import numpy as np
import pandas as pd
import pytest

from utils import compute_returns, compute_mean_returns, compute_cov_matrix
from black_litterman import (
    compute_equilibrium_returns,
    build_views,
    compute_omega,
    black_litterman_posterior,
    optimize_bl_portfolio,
    generate_momentum_views,
    sensitivity_analysis,
)


# ---------------------------------------------------------------------------
# Données fictives partagées
# ---------------------------------------------------------------------------

@pytest.fixture
def setup():
    """Prépare toutes les données nécessaires pour les tests."""
    np.random.seed(0)
    dates = pd.date_range("2022-01-01", periods=200, freq="B")
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(np.random.randn(200, 3) * 0.01, axis=0)),
        index=dates,
        columns=["AAPL", "MSFT", "GOOGL"],
    )
    returns = compute_returns(prices)
    mu = compute_mean_returns(returns)
    cov = compute_cov_matrix(returns)
    market_weights = pd.Series({"AAPL": 0.4, "MSFT": 0.35, "GOOGL": 0.25})
    pi = compute_equilibrium_returns(cov, market_weights)
    return {"prices": prices, "mu": mu, "cov": cov, "mw": market_weights, "pi": pi}


# ---------------------------------------------------------------------------
# Tests : compute_equilibrium_returns
# ---------------------------------------------------------------------------

def test_equilibrium_returns_shape(setup):
    """Pi doit avoir autant d'éléments que d'actifs."""
    assert len(setup["pi"]) == 3


def test_equilibrium_returns_positive(setup):
    """Avec une aversion au risque positive, Pi doit être positif."""
    assert (setup["pi"] > 0).all()


def test_equilibrium_returns_scales_with_risk_aversion(setup):
    """Doubler l'aversion au risque doit doubler Pi."""
    cov, mw = setup["cov"], setup["mw"]
    pi1 = compute_equilibrium_returns(cov, mw, risk_aversion=1.0)
    pi2 = compute_equilibrium_returns(cov, mw, risk_aversion=2.0)
    np.testing.assert_allclose(pi2.values, 2 * pi1.values, rtol=1e-6)


# ---------------------------------------------------------------------------
# Tests : build_views
# ---------------------------------------------------------------------------

def test_build_views_absolute(setup):
    """Une view absolue doit mettre 1 sur l'actif concerné."""
    views = [{"type": "absolute", "asset": "AAPL", "return": 0.10}]
    P, Q = build_views(["AAPL", "MSFT", "GOOGL"], views)
    assert P.shape == (1, 3)
    assert P[0, 0] == 1.0  # AAPL est en position 0
    assert Q[0] == 0.10


def test_build_views_relative(setup):
    """Une view relative doit mettre +1 et -1 sur les actifs concernés."""
    views = [{"type": "relative", "outperformer": "MSFT", "underperformer": "GOOGL", "return": 0.05}]
    P, Q = build_views(["AAPL", "MSFT", "GOOGL"], views)
    assert P[0, 1] == 1.0   # MSFT
    assert P[0, 2] == -1.0  # GOOGL
    assert Q[0] == 0.05


def test_build_views_multiple(setup):
    """Plusieurs views doivent produire une matrice P avec autant de lignes."""
    views = [
        {"type": "absolute", "asset": "AAPL", "return": 0.10},
        {"type": "absolute", "asset": "MSFT", "return": 0.08},
    ]
    P, Q = build_views(["AAPL", "MSFT", "GOOGL"], views)
    assert P.shape[0] == 2
    assert len(Q) == 2


# ---------------------------------------------------------------------------
# Tests : compute_omega
# ---------------------------------------------------------------------------

def test_omega_is_diagonal(setup):
    """Omega doit être une matrice diagonale."""
    views = [{"type": "absolute", "asset": "AAPL", "return": 0.10}]
    P, Q = build_views(["AAPL", "MSFT", "GOOGL"], views)
    omega = compute_omega(P, setup["cov"])
    off_diag = omega - np.diag(np.diag(omega))
    np.testing.assert_allclose(off_diag, 0, atol=1e-10)


def test_omega_confidence_reduces_uncertainty(setup):
    """Une confiance plus haute doit produire un Omega plus petit."""
    views = [{"type": "absolute", "asset": "AAPL", "return": 0.10}]
    P, Q = build_views(["AAPL", "MSFT", "GOOGL"], views)
    omega_low = compute_omega(P, setup["cov"], confidences=[0.3])
    omega_high = compute_omega(P, setup["cov"], confidences=[0.9])
    assert omega_high[0, 0] < omega_low[0, 0]


# ---------------------------------------------------------------------------
# Tests : black_litterman_posterior
# ---------------------------------------------------------------------------

def test_posterior_shape(setup):
    """mu_bl et cov_bl doivent avoir les bonnes dimensions."""
    views = [{"type": "absolute", "asset": "AAPL", "return": 0.10}]
    P, Q = build_views(["AAPL", "MSFT", "GOOGL"], views)
    omega = compute_omega(P, setup["cov"])
    mu_bl, cov_bl = black_litterman_posterior(setup["pi"], setup["cov"], P, Q, omega)
    assert len(mu_bl) == 3
    assert cov_bl.shape == (3, 3)


def test_posterior_moves_toward_view(setup):
    """Si on dit qu'AAPL fait +30%, mu_bl[AAPL] doit être > pi[AAPL]."""
    views = [{"type": "absolute", "asset": "AAPL", "return": 0.30}]
    P, Q = build_views(["AAPL", "MSFT", "GOOGL"], views)
    omega = compute_omega(P, setup["cov"], confidences=[0.99])
    mu_bl, _ = black_litterman_posterior(setup["pi"], setup["cov"], P, Q, omega)
    assert mu_bl["AAPL"] > setup["pi"]["AAPL"]


def test_posterior_no_views_stays_near_prior(setup):
    """Avec une très faible confiance (Omega grand), mu_bl doit rester proche de Pi."""
    views = [{"type": "absolute", "asset": "AAPL", "return": 0.50}]
    P, Q = build_views(["AAPL", "MSFT", "GOOGL"], views)
    omega = compute_omega(P, setup["cov"], confidences=[0.01])  # très peu confiant
    mu_bl, _ = black_litterman_posterior(setup["pi"], setup["cov"], P, Q, omega)
    np.testing.assert_allclose(mu_bl.values, setup["pi"].values, atol=0.05)


# ---------------------------------------------------------------------------
# Tests : optimize_bl_portfolio
# ---------------------------------------------------------------------------

def test_bl_weights_sum_to_one(setup):
    """Les poids BL doivent sommer à 1."""
    views = [{"type": "absolute", "asset": "AAPL", "return": 0.10}]
    P, Q = build_views(["AAPL", "MSFT", "GOOGL"], views)
    omega = compute_omega(P, setup["cov"])
    mu_bl, cov_bl = black_litterman_posterior(setup["pi"], setup["cov"], P, Q, omega)
    result = optimize_bl_portfolio(mu_bl, cov_bl)
    assert abs(result["weights"].sum() - 1.0) < 1e-5


def test_bl_weights_non_negative(setup):
    """Pas de vente à découvert dans les poids BL."""
    views = [{"type": "absolute", "asset": "AAPL", "return": 0.10}]
    P, Q = build_views(["AAPL", "MSFT", "GOOGL"], views)
    omega = compute_omega(P, setup["cov"])
    mu_bl, cov_bl = black_litterman_posterior(setup["pi"], setup["cov"], P, Q, omega)
    result = optimize_bl_portfolio(mu_bl, cov_bl)
    assert (result["weights"] >= -1e-6).all()


# ---------------------------------------------------------------------------
# Tests : generate_momentum_views
# ---------------------------------------------------------------------------

def test_momentum_views_format(setup):
    """Les views générées doivent avoir le bon format."""
    views = generate_momentum_views(setup["prices"])
    for v in views:
        assert "type" in v
        assert "asset" in v
        assert "return" in v
        assert v["type"] == "absolute"


def test_momentum_views_assets_in_tickers(setup):
    """Les actifs dans les views doivent faire partie des tickers connus."""
    views = generate_momentum_views(setup["prices"])
    for v in views:
        assert v["asset"] in ["AAPL", "MSFT", "GOOGL"]


# ---------------------------------------------------------------------------
# Tests : sensitivity_analysis
# ---------------------------------------------------------------------------

def test_sensitivity_shape(setup):
    """La table de sensibilité doit avoir autant de lignes que de perturbations."""
    views = [{"type": "absolute", "asset": "AAPL", "return": 0.10}]
    P, Q = build_views(["AAPL", "MSFT", "GOOGL"], views)
    perturbations = [0.5, 1.0, 1.5]
    result = sensitivity_analysis(setup["pi"], setup["cov"], P, Q, setup["mw"], perturbations=perturbations)
    assert len(result) == 3


def test_sensitivity_sharpe_increases_with_views(setup):
    """Un Sharpe plus élevé quand les views sont plus fortes (actif sous-évalué)."""
    views = [{"type": "absolute", "asset": "AAPL", "return": 0.10}]
    P, Q = build_views(["AAPL", "MSFT", "GOOGL"], views)
    perturbations = [0.5, 1.0, 2.0]
    result = sensitivity_analysis(setup["pi"], setup["cov"], P, Q, setup["mw"], perturbations=perturbations)
    sharpes = result["sharpe"].values
    assert sharpes[0] != sharpes[-1]  # le Sharpe change avec la force des views
