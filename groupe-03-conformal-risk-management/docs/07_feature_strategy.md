# 07 — Feature Strategy

---

## Design Philosophy

Features exist to improve the **quality of residuals from the base learner**, not to maximize predictive accuracy. In the conformal framework, the base learner's job is to remove first-order signal so that the residuals (conformal scores) are well-behaved and approximately exchangeable.

This is a fundamentally different objective from a typical ML feature engineering exercise. We do not want overfitted features that produce vanishingly small residuals on training data but large residuals on new data. We want features that produce **stable, predictable residuals across regimes**.

**Guiding principle:** small, interpretable, regime-robust feature set. Every feature must earn its inclusion by improving residual behavior, not by boosting training R².

---

## Essential Features (Mandatory)

These seven features form the core design matrix. All are computed at time t for predicting r_{t+1}.

| Feature | Formula | Economic Rationale |
|---|---|---|
| Lagged return (lag 1) | `r_{t}` | Mean reversion / momentum signal at 1-day lag |
| Lagged return (lag 2) | `r_{t-1}` | Eliminates autocorrelation structure in residuals |
| Lagged return (lag 5) | `r_{t-4}` | Weekly seasonal echo in equity returns |
| Rolling 20-day realized vol | `rv_t = std(r_{t-19:t})` | Volatility clustering proxy; most important regime feature |
| VIX (lagged 1 day) | `VIX_{t-1}` | Market-implied fear gauge; covers tail-risk regime shifts |
| Rolling return (5-day) | `mean(r_{t-4:t})` | Short-term momentum capture |
| Return squared (lag 1) | `r_t^2` | ARCH-like heteroscedasticity signal |

**Note on rv_t:** This is the central feature. If the base learner conditions on realized volatility, then intervals will be heteroscedastic by construction. This is exactly what conformalized quantile regression and normalized conformal methods need to adjust interval width properly.

---

## Optional Features (Include Only If Time Permits)

These features add marginal value and should only be added if the essential set is already validated and working:

| Feature | Formula | Rationale | Risk |
|---|---|---|---|
| Rolling 60-day realized vol | `std(r_{t-59:t})` | Longer memory volatility regime signal | High correlation with rv_t (may be redundant) |
| Up/down volume ratio | `(vol on up days) / (vol on down days)` over 10-day window | Market breadth indicator | Requires intraday volume split — may not be cleanly available in yfinance daily |
| Lag 10 return | `r_{t-9}` | Bi-weekly reversal pattern | Very weak signal, adds noise |
| VIX 5-day change | `VIX_{t-1} - VIX_{t-6}` | VIX term structure dynamics | Correlated with existing VIX feature |

**Verdict:** Unless the essential model shows systematic residual autocorrelation patterns at 10-day lag, these optional features should be excluded. Adding them consumes implementation time and raises questions about overfitting during the oral defense.

---

## Explicitly Rejected Features

These features are rejected permanently. Including them would be a grading risk.

| Feature | Why Rejected |
|---|---|
| RSI, MACD, Bollinger Bands | Lagging technical indicators with no incremental information beyond lagged returns + vol; add complexity without theoretical basis |
| Earnings surprises | Cannot be cleanly downloaded from free sources; creates data sourcing risk |
| Sentiment scores (news NLP) | Black box; hard to reproduce; not within conformal methods scope |
| Macro indicators (CPI, PMI, GDP) | Mixed-frequency data alignment is complex; macro at monthly frequency adds very little to daily return prediction |
| Correlation with other assets | No second asset in scope |
| Options implied vol surface (term structure) | Requires options data pipeline; out of scope |
| Order flow imbalance | Intraday; not available at daily granularity |
| Analyst price targets | Requires external data sourcing; not reproducible via public APIs |
| Sector rotation indicators | Multiple tickers required; out of scope |

---

## Feature Engineering Rules

1. **All rolling windows: backward-looking only.** `rolling(20).std().shift(1)` — the shift(1) is non-negotiable.
2. **No standardization of features across the full sample.** Standardize only within the training set, then apply the training-set mean and std to validation and test sets.
3. **No winsorization of extreme returns.** Tail events must be preserved — they are the subject of the study.
4. **No PCA or dimensionality reduction.** 7 features is already minimal; PCA would destroy interpretability.
5. **No feature interactions (products of features).** The Ridge Regression base learner is a linear model — added polynomial terms are not warranted and create overfitting risk.

---

## Expected Feature Importance (Ridge Coefficients)

Based on financial econometrics literature, the expected ordering of Ridge coefficient magnitudes is approximately:

1. Rolling 20-day realized volatility (`rv_t`) — highest magnitude
2. VIX_{t-1} — strongly correlated with rv_t, but provides forward-looking component
3. Lag-1 return `r_t` — weak but persistent momentum/reversal signal
4. Return squared `r_t^2` — ARCH effect proxy
5. Rolling 5-day return mean — momentum
6. Lag-2 return, lag-5 return — diminishing returns

If the fitted coefficients deviate dramatically from this ordering, investigate for data leakage.

---

## Feature–Target Relationship Summary

The return `r_{t+1}` has very low conditional predictability from lagged returns alone (R² ≈ 0.5–2% is typical). This is expected and is NOT a failure. The goal is not high R²; it is well-calibrated residuals. A Ridge model with R² = 1.5% but stable, near-white-noise residuals is superior to a Random Forest with R² = 8% but residuals that are heteroscedastic and regime-dependent.

The conditional heteroscedasticity (time-varying volatility) visible in `r_t^2` and `rv_t` is intentionally kept in the features rather than modeled away with GARCH. This ensures the Ridge residuals retain heteroscedasticity, which gives the conformal methods meaningful signal to work with — especially conformalized quantile regression and normalized split conformal.
