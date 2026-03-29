# 08 — Modeling Strategy

---

## Strategic Objective

The modeling layer serves one purpose: produce a point forecast `ŷ_{t+1}` and a set of residuals (conformal scores) `s_t = |r_t - ŷ_t|` that are as stable and interpretable as possible. High R² is irrelevant. Low overfitting and stable out-of-sample residual behavior are everything.

**Model complexity tradeoff:**
- Too simple (constant mean model) → large residuals, wide intervals, uninformative
- Too complex (deep learning) → overfits training residuals, collapses in test, breaks conformal calibration
- Just right (Ridge Regression) → linearly adjusts for known signals, leaves clean heteroscedastic residuals for conformal to handle

---

## Primary Base Learner: Ridge Regression

**Why Ridge:**
- Linear model: coefficients are interpretable under hostile questioning
- L2 regularization controls variance without feature selection instability
- Handles correlated features (rv_t and VIX are correlated) without blowing up coefficients
- Fast: trains in milliseconds, re-trains at every walk-forward step without pipeline bottleneck
- Residuals are Gaussian-like in the bulk distribution (by construction of OLS), with fat tails coming from true tail events — exactly the right conformal score structure
- Scikit-learn's `Ridge` implementation is battle-tested

**Regularization parameter λ:**  
Select via time-series cross-validation within the training set (rolling expanding window, 5 folds). Use `RidgeCV` with `alphas=[0.001, 0.01, 0.1, 1.0, 10.0]`. Fix the selected λ at fit time. Re-tune only at walk-forward re-training steps if the pipeline is designed to allow it — but this is optional.

**Training mode:**  
- Initial fit: on full training set (2004–2014)
- Walk-forward update: re-fit monthly on expanding window. Do NOT use a rolling window for the base learner — losing the 2008 crisis data from training is a mistake. Expanding window preserves long-run distributional knowledge.

---

## Point Forecast Baseline: Constant Mean (Historical Mean Return)

**What it is:** `ŷ_{t+1} = mean(r_{training})` — a trivially simple benchmark.

**Why include it:**  
The conformal interval on top of the mean model is equivalent to a historical simulation VaR with conformal calibration. This creates a natural connection to the industry-standard Historical Simulation approach and shows that conformal adds coverage guarantees over naive simulation.

**Expected performance:** Weak RMSE. This model serves as the interpretable floor: if no conformal method demonstrates better coverage than intervals built on this baseline, the conformal calibration logic must be audited.

---

## Quantile Regression Baseline

**Model:** Linear Quantile Regression (Koenker-Bassett, implemented via `statsmodels.regression.quantile_regression.QuantReg`)

**Why linear QR and not gradient boosted QR:**  
- Linear QR is theoretically motivated (it IS the optimal solution under asymmetric Laplace loss when the DGP has linear structure)
- Gradient Boosted QR (LightGBM or XGBoost quantile loss) risks overfitting on training data and then violating coverage guarantees on test data — this would be embarrassing if the QR baseline "wins" by overfitting
- Linear QR is reproducible and fast
- It provides a clean "model-based interval" baseline that is distribution-free by construction (pinball loss), creating an interesting comparison with conformal methods

**Quantiles to fit:** τ = 0.025, 0.05, 0.10 for lower tail; τ = 0.975, 0.95, 0.90 for upper tail. This gives 90%, 95%, and 99% prediction intervals without requiring separate models.

**Note on quantile crossing:** Linear QR may exhibit quantile crossing in extreme regions. Document this when it occurs — it is not a bug but a known limitation of linear QR that motivates richer approaches.

---

## Parametric Interval Baseline 1: Historical Simulation VaR

**Definition:**  
`VaR_α(t) = −quantile_α(r_{t-252:t-1})` — the α-quantile of the last 252 trading days of returns.

**Why 252 days:** Standard industry convention = 1 calendar year of trading days.

**This is NOT a forecast model.** It is a pure distributional baseline that ignores all predictive features. It answers: "What happens if you ignore predictive modeling entirely and just use empirical historical quantiles?"

**Hypothesized failure mode:** During rapid volatility expansion (e.g., COVID Feb–Mar 2020), this method draws from a calm 252-day window and is expected to underestimate tail risk. This hypothesis is tested empirically by comparing empirical coverage to nominal levels during identified stress sub-periods.

---

## Parametric Interval Baseline 2: GARCH(1,1) with Gaussian Innovations

**Why GARCH(1,1):**  
- Industry standard for daily equity return volatility modeling (as of RiskMetrics / JP Morgan methodology)
- Re-estimated on expanding window at every test step
- Implemented via `arch` library (`arch_model(returns, vol='Garch', p=1, q=1, dist='Normal')`)
- Gaussian assumption is intentionally retained (NOT Student-t) because the Gaussian-GARCH is the exact model that Basel I originally mandated — we are testing the classic approach, not the already-improved Student-t version

**Interval construction:**  
`[μ̂_t - z_α * σ̂_t, μ̂_t + z_α * σ̂_t]` where `σ̂_t` is the one-step-ahead conditional volatility from GARCH and `z_α` is the normal quantile at level α.

**Hypothesized failure mode:** Under leptokurtic return distributions, Gaussian GARCH is expected to underestimate tail risk at high confidence levels, producing intervals that are too tight. This is a standard critique in the financial econometrics literature and serves as the primary motivation for distribution-free alternatives. The hypothesis is tested empirically during stress sub-periods.

---

## Model Simplicity vs Robustness Tradeoff

| Model | Complexity | Interpretability | Expected RMSE | Coverage Stability | Recommended |
|---|---|---|---|---|---|
| Constant mean | Trivial | Perfect | Worst | N/A | Yes (baseline floor) |
| Ridge Regression | Low | High | Moderate | High | YES — primary base learner |
| Linear Quantile Regression | Low | High | N/A (not RMSE task) | Moderate | Yes (interval baseline) |
| Historical Simulation | None | Perfect | N/A | Poor under regime shift | Yes (parametric baseline) |
| GARCH Gaussian | Low-Medium | Moderate | N/A | Poor at tails | Yes (parametric baseline) |
| Random Forest | Medium | Low | Better | Lower (overfitting risk) | NO |
| XGBoost | Medium-High | Low | Better | Lower (overfitting risk) | NO |
| LSTM | High | Very Low | Potentially better | Very low (training instability) | ABSOLUTELY NOT |

---

## Explicit Rejections

**Random Forest as base learner:** Rejected because (a) non-linear predictions complicate residual interpretation, (b) inconsistent residual variance across feature space breaks split conformal calibration assumptions, (c) adds hyperparameter tuning cost with no theoretical benefit to the conformal layer.

**XGBoost / LightGBM quantile:** Rejected for quantile baseline because we cannot guarantee its quantile coverage out-of-sample without explicit conformal correction — which would make it no longer a standalone baseline.

**Student-t GARCH:** Rejected for GARCH baseline. If we used Student-t innovations, GARCH would actually perform reasonably well — which would reduce the contrast with conformal methods. We use Gaussian GARCH because that is the exact historical industry standard we are testing against. This is not cherry-picking; it is methodological honesty about what practitioners actually used under Basel I/II.

---

## Implementation Note

All models are fitted within a `ModelWrapper` class structure that exposes:
- `.fit(X_train, y_train)` → fits the model
- `.predict(X)` → returns point forecasts
- `.predict_interval(X, alpha)` → returns (lower, upper) tuples (for parametric methods only)
- `.get_residuals(X, y)` → returns `|y - ŷ|` or signed residuals for conformal score computation

This uniform interface allows the conformal methods to wrap any base learner without changing their core logic.
