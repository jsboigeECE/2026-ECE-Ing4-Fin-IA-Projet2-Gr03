# 11 — Experimental Protocol

---

## Governing Principle

Every result in this project must be the output of a pre-specified, frozen experimental protocol. No post-hoc cherry-picking. No test-set tuning. No moving of boundaries after seeing results.

**This protocol is written before code is run. It must be followed exactly.**

---

## Data Split Architecture

```
 2004-01-02          2014-12-31   2015-01-01       2017-12-31   2018-01-01     2019-12-31   2020-01-01       2024-12-31
     │────────────────────│────────────────────────│───────────────────────│────────────────────────│
     ◄─────────────────── TRAINING ──────────────────► ◄── CALIBRATION ────► ◄── VALIDATION ────────► ◄──────── TEST ──────────►
          ~2,770 days                                    ~755 days                ~504 days                ~1,260 days
       (model fitting)                          (conformal calibration)       (hyperparameter        (final evaluation only)
                                                                              selection only)
```

### Critical Rules
1. **Test set is never touched until the protocol is complete.** All method choices, hyperparameters (γ for ACI, β for EnbPI, λ for Ridge, window sizes) are frozen using only training + calibration + validation sets.
2. **Calibration set is used only for conformal score computation** — never for Ridge training.
3. **Validation set is used only for γ selection (ACI) and β selection (EnbPI)** — never for coverage reporting in publications.
4. **Test set results are computed exactly once and never repeated with different parameters.**

---

## Walk-Forward Evaluation Design

### Why Walk-Forward

Financial time series exhibit non-stationarity. A single train/test split is not sufficient to detect regime-dependent performance. Walk-forward evaluation simulates the actual deployment conditions of a risk model.

### Walk-Forward Protocol for Parametric Methods (GARCH, HistSim)

```
For each test day t in [2020-01-02, 2024-12-31]:
  1. Fit GARCH on data from [2004-01-02, t-1]   (expanding window)
  2. Compute σ̂_t from GARCH one-step-ahead forecast
  3. Record interval [μ̂ - z_α σ̂_t, μ̂ + z_α σ̂_t]
  4. Observe r_t, record coverage indicator 1{r_t ∈ interval}
```

**Re-fitting frequency:** Monthly (first trading day of each month). Daily re-fitting of GARCH costs 25× more compute for negligible gain. Monthly re-fitting is the industry standard.

### Walk-Forward Protocol for Ridge Regression (Base Learner)

```
For each test day t:
  1. Fit Ridge on data from [2004-01-02, t-1]   (expanding window)
  2. Predict ŷ_t = f(X_t)
  3. Record residual e_t = r_t - ŷ_t
```

**Re-fitting frequency:** Monthly. Expanding window (never rolling — 2008 crisis data stays in).

### Protocol for Split Conformal Prediction

```
One-time calibration:
  1. Fit Ridge on [2004-01-02, 2014-12-31]
  2. Compute calibration scores: {|r_i - ŷ_i| : i ∈ [2015-01-02, 2017-12-31]}
  3. q̂_α = quantile_{⌈(1-α)(n_cal+1)/n_cal⌉} of calibration scores

For each test day t in [2020-01-02, 2024-12-31]:
  1. Predict ŷ_t using current Ridge model (re-fitted monthly on expanding window)
  2. Interval: [ŷ_t - q̂_α, ŷ_t + q̂_α]  (static quantile, never updated)
  3. Record coverage indicator
```

**Note:** The calibration quantile q̂_α is fixed at January 2020 and NEVER updated throughout the test period. This is deliberately strict to test the static assumption.

### Protocol for CQR (Conformalized Quantile Regression)

```
One-time calibration:
  1. Fit linear QR (τ_low, τ_high) on [2004-01-02, 2014-12-31]
  2. Compute CQR scores: {max(q̂_low(X_i) - r_i, r_i - q̂_high(X_i)) : i ∈ calibration}
  3. q̂_CQR_α = quantile_{⌈(1-α)(n+1)/n⌉} of CQR scores

For each test day t:
  1. Compute q̂_low(X_t), q̂_high(X_t) from fitted QR models
  2. Interval: [q̂_low(X_t) - q̂_CQR_α, q̂_high(X_t) + q̂_CQR_α]
  3. Record coverage indicator
```

### Protocol for EnbPI

```
Setup:
  1. Train K=20 bootstrap Ridge models on subsets of [2004-01-02, 2014-12-31]
  2. Initialize score buffer with calibration residuals [2015-01-02, 2017-12-31]

For each test day t:
  1. Predict ŷ_t = ensemble mean of K models
  2. Update score buffer: remove oldest score, add |r_{t-1} - ŷ_{t-1}|  (previous step)
  3. q̂_α(t) = quantile_{α} of current score buffer (rolling ~126 observations)
  4. Interval: [ŷ_t - q̂_α(t), ŷ_t + q̂_α(t)]
  5. Record coverage indicator
```

### Protocol for ACI

```
Initialize: α_0 = α (nominal miscoverage level, e.g., 0.10 for 90% coverage)
Initialize conformal score distribution from calibration set (same as Split CP)

For each test day t:
  1. Current effective level: α_t
  2. Prediction interval: Split CP interval at level α_t (using calibration quantiles)
  3. Observe r_t, compute err_t = 1{r_t ∉ interval_t}
  4. Update: α_{t+1} = α_t + γ(α - err_t)
  5. Clip α_{t+1} to [0.001, 0.499]  (prevent degenerate levels)
  6. Record interval, coverage, and α_t time series
```

**γ selection:** Run ACI on validation set [2018-01-01, 2019-12-31] for γ ∈ {0.005, 0.01, 0.02, 0.05}. Select γ that minimizes |rolling_coverage_60d - α| on the validation set. Fix γ before test.

---

## Confidence Levels and Two-Track Evaluation Framework

This project uses **two-sided prediction intervals** for the uncertainty quantification comparison. Because conformal intervals are symmetric by construction (absolute residual scores), the VaR implication of the lower bound must be computed correctly.

**Track 1 — Two-sided interval coverage (primary conformal evaluation):**

| Nominal two-sided coverage | α parameter | α parameter in code |
|---|---|---|
| 80% | 0.20 | Supplementary only |
| **90%** | **0.10** | **Primary** |
| 95% | 0.05 | Secondary |

**Track 2 — VaR backtesting (Kupiec test on lower bound):**

For a symmetric two-sided (1−α) conformal interval, the lower bound satisfies P(r < L) ≤ α/2.

| Two-sided interval level | Lower bound = VaR at level | Kupiec exceedance rate | Kupiec test label |
|---|---|---|---|
| 90% | 95% VaR | 5% | **Primary Kupiec test** |
| 95% | 97.5% VaR | 2.5% | Secondary |

**Consequence:** When reporting Kupiec test results, label all tables clearly as "Kupiec test at 5% exceedance (lower bound of 90% two-sided interval = 95% VaR estimate)." Do not conflate the two-sided coverage level with the VaR confidence level.

**For CQR and Linear QR:** These methods fit quantiles directly. The lower bound of the QR interval at τ_low = 0.05 directly targets the 95% VaR (5% exceedance). This is the correct comparison baseline for the Kupiec test — τ_low = 0.05 for the lower quantile.

**Primary results discussion:** Focus on the 90% two-sided interval / 95% VaR comparison. This is the finance-standard horizon and provides the clearest Kupiec test power (~63 exceptions in 1,260 days at 5% rate).

---

## Reproducibility Rules

1. **Random seed:** `np.random.seed(42)` set globally at the start of all experiments.
2. **yfinance data pinned:** All data downloaded once and saved to `data/raw/spy_daily.csv` and `data/raw/vix_daily.csv`. No live API calls during experiments.
3. **Environment pinned:** `requirements.txt` or `environment.yml` specifies exact package versions.
4. **Results saved:** All interval time series, coverage indicators, width time series saved to `results/` as CSV files.
5. **No test peeking:** The test set data file is not loaded until the protocol is complete and all parameters are frozen.

---

## Leakage Prevention Checklist

- [ ] All rolling windows use `.shift(1)` before joining to target
- [ ] VIX feature uses t-1 value, not t-value
- [ ] Calibration quantiles computed before test set is loaded
- [ ] ACI γ selected on validation set only
- [ ] EnbPI β selected on validation set only
- [ ] Ridge λ selected on training set only (time-series CV)
- [ ] GARCH refitted on data strictly before the prediction date
- [ ] No future returns used in any feature at any step
- [ ] Walk-forward loop processes steps strictly sequentially (no batch vectorization over future)

---

## Stress-Period Sub-Analysis Protocol

After the full test set is evaluated, slice results into three sub-periods for conditional analysis:

| Sub-Period | Dates | VIX Threshold | Label |
|---|---|---|---|
| COVID Crisis | 2020-02-19 to 2020-04-30 | VIX > 30 | High Stress |
| 2022 Bear Market | 2022-01-01 to 2022-12-31 | VIX > 20 | Moderate Stress |
| Recovery / Bull | 2023-01-01 to 2024-12-31 | VIX < 20 | Low Stress |

Report all metrics separately for each sub-period. Do not re-run the models with different parameters for each sub-period — just slice the already-computed results.

---

## Final Output of the Protocol

For each method × confidence level × sub-period combination, the protocol produces:
1. Empirical coverage rate
2. Average interval width
3. Number of exceptions (VaR breaches)
4. Kupiec LR test p-value
5. Christoffersen conditional coverage test p-value
6. Rolling 60-day coverage time series
7. Interval width time series

These outputs feed directly into all metrics, charts, and the decision layer.
