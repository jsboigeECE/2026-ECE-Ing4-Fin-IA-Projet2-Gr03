"""
Tests unitaires pour la solution analytique de Black-Scholes.

Exécuter avec :  pytest tests/test_bs_analytical.py -v
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from src.analytics.black_scholes_formula import (
    bs_call_price, bs_put_price, bs_delta, bs_gamma,
)


# -----------------------------------------------------------------------
# Fixtures / valeurs de référence
# -----------------------------------------------------------------------
# Valeurs vérifiées contre scipy.stats et des tables financières standard

PARAMS = dict(K=100.0, T=1.0, r=0.05, sigma=0.20)
ATOL = 1e-4   # tolérance absolue en dollars


class TestCallPrice:
    def test_atm_approx(self):
        """ATM call ≈ 0.4 × σ × S × sqrt(T) pour r ≈ 0."""
        price = bs_call_price(S=100, **PARAMS)
        assert 8.0 < float(price) < 12.0   # plage raisonnable

    def test_deep_itm(self):
        """Call très en-the-money ≈ S - K·e^{-rT}."""
        S = 200.0
        intrinsic = S - PARAMS["K"] * np.exp(-PARAMS["r"] * PARAMS["T"])
        price = bs_call_price(S=S, **PARAMS)
        assert abs(float(price) - intrinsic) < 1.0

    def test_deep_otm(self):
        """Call très hors-the-money ≈ 0."""
        price = bs_call_price(S=20.0, **PARAMS)
        assert float(price) < 0.01

    def test_at_expiry(self):
        """À maturité (T=0), prix = payoff."""
        price_itm = bs_call_price(S=120, K=100, T=0, r=0.05, sigma=0.2)
        assert abs(float(price_itm) - 20.0) < ATOL
        price_otm = bs_call_price(S=80, K=100, T=0, r=0.05, sigma=0.2)
        assert abs(float(price_otm) - 0.0) < ATOL

    def test_known_value(self):
        """Référence : S=100, K=100, T=1, r=5%, σ=20% → ~10.451$."""
        price = bs_call_price(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        assert abs(float(price) - 10.4506) < 1e-2

    def test_array_input(self):
        """Fonctionne sur tableaux numpy."""
        S = np.array([80, 100, 120])
        prices = bs_call_price(S=S, **PARAMS)
        assert prices.shape == (3,)
        assert np.all(prices >= 0)


class TestPutCallParity:
    """Put-call parity: C - P = S - K·e^{-rT}."""

    @pytest.mark.parametrize("S", [60, 80, 100, 120, 150])
    def test_parity(self, S):
        call = bs_call_price(S=S, **PARAMS)
        put  = bs_put_price (S=S, **PARAMS)
        lhs = float(call - put)
        rhs = S - PARAMS["K"] * np.exp(-PARAMS["r"] * PARAMS["T"])
        assert abs(lhs - rhs) < ATOL


class TestGreeks:
    def test_delta_call_between_0_and_1(self):
        S = np.linspace(50, 200, 50)
        d = bs_delta(S=S, **PARAMS, option="call")
        assert np.all(d >= -1e-6) and np.all(d <= 1 + 1e-6)

    def test_delta_put_between_minus1_and_0(self):
        S = np.linspace(50, 200, 50)
        d = bs_delta(S=S, **PARAMS, option="put")
        assert np.all(d >= -1 - 1e-6) and np.all(d <= 1e-6)

    def test_gamma_positive(self):
        S = np.linspace(50, 200, 50)
        g = bs_gamma(S=S, **PARAMS)
        assert np.all(g >= 0)

    def test_gamma_at_expiry_zero(self):
        g = bs_gamma(S=100, K=100, T=0, r=0.05, sigma=0.2)
        assert float(g) == 0.0


class TestEdgeCases:
    def test_spot_zero(self):
        price = bs_call_price(S=1e-9, K=100, T=1.0, r=0.05, sigma=0.2)
        assert float(price) >= 0

    def test_very_high_vol(self):
        price = bs_call_price(S=100, K=100, T=1.0, r=0.05, sigma=2.0)
        assert float(price) > 0

    def test_zero_rate(self):
        price = bs_call_price(S=100, K=100, T=1.0, r=0.0, sigma=0.20)
        assert float(price) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
