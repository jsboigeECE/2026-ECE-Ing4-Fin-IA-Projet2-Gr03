# 06 — Target Definition

---

## Primary Target: SPY 1-Day Ahead Log-Return

**Definition:**

```
r_{t+1} = log(Close_{t+1} / Close_t)
```

Where `Close_t` is the adjusted closing price of SPY on trading day t.

**What we are predicting:** the return of SPY from the close of day t to the close of day t+1.

**At prediction time t:** we know all data up to and including day t. We predict `r_{t+1}` using features available at time t and before.

---

## Why This Target — Academic Justification

1. **Log-returns are the standard in quantitative finance.** They are additively decomposable across time horizons and approximately normally distributed over short windows (though the tails are fat — which is exactly what we are testing the interval methods on).

2. **1-day horizon = regulatory VaR horizon.** Basel III, Article 325ba(2): banks must compute VaR over a 10-day horizon, but this is standardized from 1-day VaR via the square-root-of-time rule. Our 1-day target directly maps to the foundational input of all risk capital models.

3. **Conformal prediction is easiest to evaluate at 1-step-ahead.** Multi-step ahead prediction intervals require propagation of uncertainty across steps, complicating the coverage guarantee derivation. At 1-step-ahead, the coverage guarantee is clean and directly testable.

4. **Residuals from a basic forecasting model on log-returns have near-zero autocorrelation at long lags** — this means conformal scores (residuals) are approximately exchangeable on average, which validates the exchangeability assumption for static conformal. The violation of this assumption under volatility clustering is precisely what motivates adaptive methods.

---

## Why NOT Alternative Targets

| Alternative Target | Rejection Reason |
|---|---|
| Price level (absolute Close_t) | Non-stationary; conformal prediction on integrated processes is theoretically invalid |
| 5-day compounded return | Overlapping returns introduce autocorrelation that inflates conformal scores; requires Hansen-Hodrick correction |
| Realized volatility | Changes the problem entirely: not VaR but vol forecasting. Different literature, different finance application, dilutes the narrative. |
| Option implied volatility (VIX prediction) | VIX is not tradeable directly; would require derivatives data for the decision layer |
| Binary direction (up/down) | Classification conformal prediction exists but is far weaker on coverage guarantees for financial applications; loses all magnitude information |
| VaR directly as a regression target | Circular: you cannot train a model to predict yesterday's VaR and call it a VaR model for tomorrow without double-dipping on the calibration data |

---

## Target Construction: Exact Recipe

```
Step 1: Download SPY adjusted close via yfinance, ticker "SPY", start 2003-12-01, end 2024-12-31
        (extra one month before 2004-01-02 to allow lag features to warm up)

Step 2: Compute log-return series:
        r_t = log(adj_close_t / adj_close_{t-1})

Step 3: Shift target one period forward:
        y_t (our prediction target for predictions made at time t) = r_{t+1}

Step 4: Drop first row (NaN due to log-return computation)

Step 5: The resulting dataframe has:
        - Index: trading date t
        - y_t: return from close_t to close_{t+1}  ← TARGET
        - All features computed from data up to and including date t  ← FEATURES
```

This guarantees zero temporal leakage by construction.

---

## What Makes This Target Suitable for Conformal Evaluation

**Key property:** The actual return `r_{t+1}` is observed exactly at time t+1. There is no ambiguity, estimation error, or model dependency in the realization of the target. This makes coverage calculation exact:

```
Coverage at day t = 1 if r_{t+1} ∈ [L_t, U_t], else 0
Empirical coverage over window [t_a, t_b] = mean( coverage_{t_a:t_b} )
```

This is cleaner than option pricing (where "true" price is debated) or volatility (where realized vol is itself an estimator).

**On exchangeability:** The theoretical exchangeability assumption for split conformal requires that calibration and test scores come from the same distribution. Log-returns approximately satisfy this during stable regimes. The adaptive methods (ACI, EnbPI) are designed precisely for the case where this fails — the volatility clustering and regime shifts that characterize financial crises. Testing both types of methods on this target directly confronts the exchangeability assumption.

---

## Numerical Characteristics of the Target (Expected)

| Statistic | Approximate Value for SPY Daily Log-Returns |
|---|---|
| Mean | ~+0.03% (daily) |
| Standard deviation | ~0.95% (daily, full sample) |
| Skewness | ~−0.5 (left-skewed: crashes worse than rallies) |
| Excess kurtosis | ~5–8 (heavy-tailed: fat tails are real) |
| Autocorrelation (lag 1) | ~−0.02 (near-zero, weak mean reversion) |
| Min (crisis day) | ~−12% (March 16, 2020) |
| Max (crisis bounce) | ~+10% (March 24, 2020) |
| VaR 95% (empirical, 1-day) | ~−1.7% |
| VaR 99% (empirical, 1-day) | ~−3.2% |

These characteristics are well-known in the financial econometrics literature and provide immediate sanity checks when our pipeline produces results. Any coverage estimate wildly inconsistent with the empirical quantiles above is a bug, not a finding.
