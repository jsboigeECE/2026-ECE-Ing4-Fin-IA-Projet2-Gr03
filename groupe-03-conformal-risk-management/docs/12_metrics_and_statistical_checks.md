# 12 — Metrics and Statistical Checks

---

## Metric Framework Overview

Metrics fall into four layers. Each layer is non-optional. Reporting only one layer makes the project undergraduate-level.

```
Layer 1: Point Forecast Quality      (base learner sanity check)
Layer 2: Interval Coverage Metrics   (primary conformal evaluation)
Layer 3: Interval Width Metrics      (efficiency of intervals)
Layer 4: Decision Layer Metrics      (finance application value)
```

---

## Layer 1 — Point Forecast Quality

These metrics validate that the base learner is functioning correctly. They are NOT the main deliverable.

| Metric | Formula | Expected Value (SPY daily) | Purpose |
|---|---|---|---|
| RMSE | `sqrt(mean((y - ŷ)²))` | ~0.009–0.011 | Base learner sanity check |
| MAE | `mean(|y - ŷ|)` | ~0.007–0.009 | Robust to outliers |
| R² (out-of-sample) | `1 - SS_res/SS_tot` | ~0.001–0.02 (very low — expected) | Confirm model is above constant-mean baseline |
| Diebold-Mariano test | Comparison of MSE series between Ridge and constant-mean | p < 0.05 expected | Verify Ridge is a meaningful improvement |

**Note on low R²:** R² ≈ 1% for SPY daily returns is normal and expected. The Efficient Market Hypothesis implies limited predictability. If anyone challenges the low R², the correct answer is: "Low R² is correct for efficient markets at daily frequency. We are not here to beat the market — we are here to quantify forecast uncertainty."

---

## Layer 2 — Interval Coverage Metrics

These are the primary evaluation metrics of this project.

### 2A. Empirical Coverage Rate

```
Coverage(α) = (1/T) Σ_{t=1}^{T} 1{r_t ∈ [L_t, U_t]}
```

**Target:** ≥ 1−α (e.g., ≥ 0.90 for 90% nominal coverage)  
**Reported for:** all 7 methods × 3 confidence levels × 3 sub-periods = 63 values  
**Key table:** see `results/coverage_table.csv`

### 2B. Kupiec Proportion-of-Failures (POF) Test

Tests whether the observed exception rate equals the nominal level (1−α).

```
LR_POF = 2 log[ π̂^x (1-π̂)^{T-x} / α^x (1-α)^{T-x} ]
```

Where π̂ = exceptions/T, x = number of exceptions.  
Under H0, LR_POF ~ χ²(1).

| Zone | Exceptions (250 days, 99% VaR) | Interpretation |
|---|---|---|
| Green (Basel) | 0–4 | Model accepted |
| Yellow | 5–9 | Model under review |
| Red | ≥ 10 | Model rejected |

Report Kupiec p-value for all methods at all confidence levels. **This is the most finance-legible result in the entire study.**

### 2C. Christoffersen Conditional Coverage Test

Tests whether exceptions are independently distributed (not clustered).

```
LR_CC = LR_POF + LR_IND
```

Where LR_IND tests independence of exception indicators. Under H0, LR_CC ~ χ²(2).

**Finance relevance:** Clustered exceptions are more dangerous than dispersed ones. A model that misses 10 days in a row is worse than one that misses 10 random days — both fail Kupiec, but only the clustered one fails Christoffersen.

**Expected finding:** GARCH and HistSim should fail Christoffersen during COVID (clustered tail events). Conformal methods, especially ACI, should pass or be less severe.

### 2D. Rolling Coverage Rate (60-day window)

```
RollingCoverage_t = (1/60) Σ_{k=0}^{59} 1{r_{t-k} ∈ interval_{t-k}}
```

**Purpose:** Captures time-varying coverage reliability. A method that has 90% average coverage but drops to 60% for three months during COVID is far more dangerous than the average suggests.

**This is the most visually compelling metric in the project.** A well-designed rolling coverage plot is the "hero" figure of the presentation.

### 2E. Winkler Score (Optional but High-Value)

```
W_t = (U_t - L_t) + (2/α) max(L_t - r_t, 0) + (2/α) max(r_t - U_t, 0)
```

The Winkler score penalizes both wide intervals AND coverage violations simultaneously. It is a proper scoring rule that unifies coverage and efficiency into one number.

**Grading note:** Including Winkler score signals familiarity with probabilistic forecasting scoring theory (Gneiting & Raftery, 2007). High theoretical credibility, low implementation cost.

---

## Layer 3 — Interval Width Metrics

### 3A. Mean Interval Width (MIW)

```
MIW = (1/T) Σ_{t=1}^T (U_t - L_t)
```

**Interpretation:** Narrower intervals = more useful predictions, as long as coverage is maintained.  
**Expected ordering:** ACI > EnbPI ≈ HistSim > GARCH (in normal regime) > Split CP ≈ CQR

### 3B. Width Efficiency Ratio

```
WER(m) = MIW(m) / MIW(Split CP)
```

Normalizes all method widths against the static conformal baseline.  
**Values > 1.0:** method uses wider intervals than static CP  
**Values < 1.0:** method is more efficient

### 3C. Width Volatility (Std of Width)

```
WidthVol = std((U_t - L_t)_{t=1}^T)
```

**Purpose:** Measures how much the interval width varies over time. ACI and EnbPI should have higher width volatility (tracking volatility regimes). Static CP + GARCH should have lower width volatility.

### 3D. Width–Coverage Correlation

```
cor( RollingWidth_t, RollingCoverage_t )
```

**Hypothesis:** Better-adapted methods should have POSITIVE correlation between width and coverage (wider intervals → better coverage during stress). Poorly-adapted methods (static CP under stress) may show NEGATIVE correlation (intervals stay narrow exactly when they most need to be wide).

---

## Layer 4 — Decision Layer Metrics

### 4A. Sharpe Ratio of Uncertainty-Aware Strategy

```
SR = mean(daily_returns_strategy) / std(daily_returns_strategy) × sqrt(252)
```

Computed on the test set for: (1) buy-and-hold, (2) interval-width-adjusted position sizing.

### 4B. Maximum Drawdown

```
MaxDD = max_{t} [ max_{s≤t}(V_s) - V_t ] / max_{s≤t}(V_s)
```

**Target:** Interval-aware rule should reduce MaxDD during COVID and 2022 periods.

### 4C. Calmar Ratio

```
Calmar = Annualized_Return / MaxDD
```

Combines return and drawdown into one metric. Preferred over Sharpe when tail-risk management is the objective.

### 4D. Exception-Triggered Loss

```
ETL = mean(r_t | r_t < L_t)
```

Average return on days when the actual return breaches the lower prediction bound. This is the expected shortfall conditional on a VaR exception — a direct measure of interval severity.

---

## Statistical Checks to Run

| Check | Method | When |
|---|---|---|
| Residual autocorrelation (Ljung-Box) | On Ridge residuals in training | After model fitting |
| Heteroscedasticity (ARCH-LM test) | On Ridge residuals | After model fitting |
| Coverage stationarity (mean difference test per regime) | Coverage indicator time series | After test evaluation |
| Kupiec POF test | All methods, all α levels | After test evaluation |
| Christoffersen CC test | All methods, all α levels | After test evaluation |
| Diebold-Mariano (point forecast) | Ridge vs constant mean | After test evaluation |
| Wilcoxon width test | Pairwise method width comparison | After test evaluation |

---

## What Results Matter Most for Grading

**Rank 1 (essential):** Rolling 60-day coverage plot — visual demonstration of adaptive vs static coverage behavior across regimes. This is what evaluators remember.

**Rank 2 (essential):** Kupiec POF test table — quantitative regime comparison. Shows who "passes" the regulatory backtest and who doesn't.

**Rank 3 (high value):** Width efficiency table — demonstrates you understand the coverage/width tradeoff, not just that more coverage is always better.

**Rank 4 (differentiating):** Christoffersen test results — distinguishes methodological sophistication from simple coverage counting.

**Rank 5 (wow factor):** Decision layer Sharpe and MaxDD comparison — translates methodology into finance language.

**Rank 6 (bonus):** Winkler score table — signals deep scoring theory knowledge.
