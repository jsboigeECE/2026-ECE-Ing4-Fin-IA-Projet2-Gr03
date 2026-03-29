"""
Tests unitaires pour utils.py
"""

import numpy as np
import pandas as pd
import pytest

from utils import (
    compute_returns,
    compute_mean_returns,
    compute_cov_matrix,
    markowitz_weights,
    efficient_frontier,
    portfolio_performance,
)


# ---------------------------------------------------------------------------
# Données fictives partagées entre les tests (pas besoin d'internet)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_prices():
    """3 actifs, 100 jours de prix simulés."""
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=100, freq="B")
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(np.random.randn(100, 3) * 0.01, axis=0)),
        index=dates,
        columns=["AAPL", "MSFT", "GOOGL"],
    )
    return prices


@pytest.fixture
def sample_returns(sample_prices):
    return compute_returns(sample_prices)


@pytest.fixture
def sample_mu_cov(sample_returns):
    mu = compute_mean_returns(sample_returns)
    cov = compute_cov_matrix(sample_returns)
    return mu, cov


# ---------------------------------------------------------------------------
# Tests : compute_returns
# ---------------------------------------------------------------------------

def test_returns_shape(sample_prices):
    """Les rendements doivent avoir une ligne de moins que les prix (diff)."""
    returns = compute_returns(sample_prices)
    assert returns.shape == (len(sample_prices) - 1, 3)


def test_returns_no_nan(sample_prices):
    """Aucune valeur manquante dans les rendements."""
    returns = compute_returns(sample_prices)
    assert not returns.isnull().any().any()


def test_returns_weekly(sample_prices):
    """Le resampling weekly doit réduire le nombre de lignes."""
    returns_daily = compute_returns(sample_prices, freq="daily")
    returns_weekly = compute_returns(sample_prices, freq="weekly")
    assert len(returns_weekly) < len(returns_daily)


# ---------------------------------------------------------------------------
# Tests : compute_mean_returns / compute_cov_matrix
# ---------------------------------------------------------------------------

def test_mean_returns_annualized(sample_returns):
    """Les rendements annualisés doivent être ~252x les rendements journaliers."""
    mu_daily = compute_mean_returns(sample_returns, annualize=False)
    mu_annual = compute_mean_returns(sample_returns, annualize=True)
    np.testing.assert_allclose(mu_annual.values, mu_daily.values * 252, rtol=1e-6)


def test_cov_matrix_symmetric(sample_returns):
    """La matrice de covariance doit être symétrique."""
    cov = compute_cov_matrix(sample_returns)
    np.testing.assert_allclose(cov.values, cov.values.T, atol=1e-10)


def test_cov_matrix_positive_definite(sample_returns):
    """Toutes les valeurs propres doivent être positives (matrice définie positive)."""
    cov = compute_cov_matrix(sample_returns)
    eigenvalues = np.linalg.eigvalsh(cov.values)
    assert np.all(eigenvalues > 0)


# ---------------------------------------------------------------------------
# Tests : markowitz_weights
# ---------------------------------------------------------------------------

def test_markowitz_weights_sum_to_one(sample_mu_cov):
    """Les poids doivent sommer à 1."""
    mu, cov = sample_mu_cov
    result = markowitz_weights(mu, cov)
    assert abs(result["weights"].sum() - 1.0) < 1e-6


def test_markowitz_weights_non_negative(sample_mu_cov):
    """Pas de vente à découvert : tous les poids >= 0."""
    mu, cov = sample_mu_cov
    result = markowitz_weights(mu, cov)
    assert (result["weights"] >= -1e-6).all()


def test_markowitz_returns_dict_keys(sample_mu_cov):
    """Le résultat doit contenir les clés attendues."""
    mu, cov = sample_mu_cov
    result = markowitz_weights(mu, cov)
    assert set(result.keys()) == {"weights", "return", "volatility", "sharpe"}


def test_markowitz_sharpe_positive(sample_mu_cov):
    """Le Sharpe optimal doit être supérieur au portefeuille équipondéré."""
    mu, cov = sample_mu_cov
    result_opt = markowitz_weights(mu, cov)
    n = len(mu)
    equal_weights = pd.Series(np.ones(n) / n, index=mu.index)
    perf_equal = portfolio_performance(equal_weights, mu, cov)
    assert result_opt["sharpe"] >= perf_equal["sharpe"] - 1e-6


# ---------------------------------------------------------------------------
# Tests : efficient_frontier
# ---------------------------------------------------------------------------

def test_efficient_frontier_returns_dataframe(sample_mu_cov):
    """La frontière efficiente doit retourner un DataFrame avec les bonnes colonnes."""
    mu, cov = sample_mu_cov
    frontier = efficient_frontier(mu, cov, n_points=10)
    assert isinstance(frontier, pd.DataFrame)
    assert set(["return", "volatility", "sharpe"]).issubset(frontier.columns)


def test_efficient_frontier_volatility_positive(sample_mu_cov):
    """Toutes les volatilités doivent être positives."""
    mu, cov = sample_mu_cov
    frontier = efficient_frontier(mu, cov, n_points=10)
    assert (frontier["volatility"] > 0).all()


# ---------------------------------------------------------------------------
# Tests : portfolio_performance
# ---------------------------------------------------------------------------

def test_portfolio_performance_equal_weights(sample_mu_cov):
    """Test de cohérence sur un portefeuille équipondéré."""
    mu, cov = sample_mu_cov
    n = len(mu)
    weights = pd.Series(np.ones(n) / n, index=mu.index)
    perf = portfolio_performance(weights, mu, cov)
    assert "return" in perf
    assert "volatility" in perf
    assert perf["volatility"] > 0
