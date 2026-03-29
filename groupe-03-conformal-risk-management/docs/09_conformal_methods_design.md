# 09 — Conformal Methods Design

---

## Conceptual Architecture

The three conformal methods in this project form a progression of increasing temporal adaptiveness:

```
Static assumption          Semi-adaptive              Fully adaptive
      │                        │                           │
      ▼                        ▼                           ▼
Split Conformal  ────►  EnbPI (rolling)  ────►  ACI (online update)
  (fixed cal.)        (ensemble scores)      (alpha correction)
```

Each method is testing a distinct conceptual claim. The comparison between them is itself the main scientific contribution.

---

## Method 1: Split Conformal Prediction (Static Baseline)

### What It Tests
Whether the distribution of calibration residuals from a stable period (2015–2017) provides valid coverage guarantees on a volatile test period (2020–2024). This is the **pure exchangeability test**.

### Formal Construction

Given:
- Training set: fit Ridge model `f`
- Calibration set: compute conformity scores `s_i = |r_i - f(X_i)|`, for i ∈ calibration
- Test day t: predict interval as `f(X_t) ± q̂_{α}` where `q̂_α = quantile_{⌈(1−α)(n+1)/n⌉}(s_1,...,s_n)`

**Coverage guarantee:** Under exchangeability, `P(r_{t+1} ∈ Ĉ_t) ≥ 1−α`. This is a finite-sample, distribution-free result (Vovk et al., 2005; Tibshirani et al., 2019).

### Expected Behavior
- Normal regime: good coverage, stable intervals correlated with volatility (because rv_t is a feature)
- Stress regime: **under-coverage** because calibration scores underestimate crisis-level residuals
- Interval widths: relatively constant (adapts only through the point forecast, not the quantile recalibration)

### Implementation
- Library: `MAPIE` — `MapieRegressor` with `method='base'`, `cv='prefit'`
- Conformity score: absolute residual `|y - ŷ|`
- Also implement **Conformalized Quantile Regression (CQR)** variant:  
  score = `max(q_low(X) - y, y - q_high(X))` using pre-fitted Linear QR.  
  CQR produces locally-adaptive widths that respond to feature-conditional variance — this serves as a bridge between static CP and full adaptive methods.

### Implementation Priority: **MANDATORY — implement first**

---

## Method 2: EnbPI — Ensemble Batch Prediction Intervals

### What It Tests
Whether ensemble-based rolling score adaptation can track distributional shift in financial time series without requiring a fully online update rule. This is the **semi-adaptive test**.

### Key Reference
Xu & Xie (2021), "Conformal prediction interval for dynamic time-series." Journal of Machine Learning Research.

### Conceptual Mechanism
EnbPI builds an ensemble of leave-one-out base learners over a bootstrap subsample set. At each test step:
1. Compute residuals using the ensemble's out-of-bag predictions on a rolling window of recent observations
2. Update the score distribution with the most recent residuals (sliding window of past β% observations)
3. Produce interval using the updated quantile

This allows the intervals to **narrow during calm periods** and **widen during volatile periods** — but the adaptation is retrospective (uses past residuals), not forward-looking.

### Key Parameter: Rolling Window Size β
- Default: β = 0.1 (use the most recent 10% of calibration+test observations for score update)
- For our test set of 1,260 days, this means approximately 126 observations in the rolling score distribution
- This corresponds to ~6 calendar months — a financially meaningful lookback

### Expected Behavior
- Better coverage during volatility transitions than static conformal
- Interval width tracks volatility clusters (widens during COVID, narrows during recovery)
- Slower adaptation than ACI during sudden shocks (β window must fill with new data)
- **Key weakness:** if the volatility shock is faster than the window, EnbPI lags

### Implementation
- Priority: **STRETCH** — implement only after Split CP, CQR, and ACI are validated and producing results
- Primary path: `MAPIE` `TimeSeriesRegressor` (wraps EnbPI-style rolling conformal for time series). Check MAPIE >= 0.8 for `EnbPI` support flag before starting.
- Fallback path: custom implementation following Xu & Xie (2021) Algo 1 (~80–100 lines of Python)
- Computational cost: moderate (bootstrap ensemble = 10–20 Ridge models)
- **If time-boxed**: drop from primary results and include in supplementary; the narrative is complete without EnbPI

---

## Method 3: Adaptive Conformal Inference (ACI)

### What It Tests
Whether an **online correction to the nominal confidence level** can maintain coverage guarantees even when the exchangeability assumption is violated by regime shift. This is the **theoretical guarantees under non-stationarity test**.

### Key Reference
Gibbs & Candès (2021), "Adaptive conformal inference under distribution shift." NeurIPS 2021.

### Conceptual Mechanism
ACI maintains a running target miscoverage level `α_t` that updates at each step:
```
α_{t+1} = α_t + γ (α − err_{t+1})
```
Where:
- `α` = nominal miscoverage level (e.g., 0.10 for 90% coverage)
- `err_{t+1}` = 1 if the true return falls outside the interval at step t+1, else 0
- `γ` > 0 = step size (learning rate for the online update)

**Interpretation:** If ACI is over-covering (too many correct predictions), it raises the effective miscoverage level (narrows intervals). If ACI is under-covering (too many exceptions), it lowers the effective miscoverage level (widens intervals).

**Theoretical guarantee:** Despite non-exchangeability, ACI guarantees:
```
(1/T) Σ_{t=1}^{T} 1{r_t ∉ Ĉ_t} → α  (almost surely)
```
This is a convergence guarantee on the time-average miscoverage, not a per-step guarantee.

### Key Parameter: Step Size γ
- Theory suggests γ = 0.005–0.05 for financial time series
- γ too small: slow adaptation to regime shifts (similar to static CP)
- γ too large: intervals oscillate rapidly (high variance in coverage)
- Run sensitivity analysis on γ ∈ {0.005, 0.01, 0.02, 0.05} on the validation set; fix γ before test

### Expected Behavior
- Best long-run coverage of all methods by design (convergence guarantee)
- Fastest recovery after a coverage violation cluster (COVID crash exceptions → immediate width expansion)
- Potential for oscillating interval widths in calm regimes if γ is too large
- **This is the method that should "win" on rolling coverage plots during stress periods**

### Implementation
- Priority: **MANDATORY** — implement second, after Split CP
- Implementation: custom (~20–30 lines of Python). ACI is not natively supported in MAPIE as a self-updating mechanism; the online update loop must be hand-coded. This is low risk given the algorithmic simplicity.
- Must track `α_t` time series alongside intervals for visualization
- Verify: MAPIE `MapieRegressor` with `cv='prefit'` is reused to generate the base conformal interval at each step; the ACI wrapper controls the input α level at each step.

---

## Where Each Method Is Expected to Win or Fail

| Condition | Static CP | CQR | EnbPI | ACI |
|---|---|---|---|---|
| Calm normal regime | ✓ tight | ✓✓ adaptive | ✓ | ✓ |
| Volatility transition (entering crisis) | ✗ lags | ✓ partial | ✓ partial | ✓✓ fast |
| Sustained stress period | ✗ undercoverage | ✓ partial | ✓ adapts slowly | ✓✓ corrects |
| Exit from crisis | ✗ still wide? | ✓ narrows | ✓ narrows | ✓✓ narrows |
| Long-run average coverage | ✓ by theory | ✓ by theory | ✓ | ✓✓ guaranteed |
| Interpretability | ✓✓ | ✓✓ | ✓ | ✓ |
| Implementation complexity | ✓✓ trivial | ✓✓ simple | ✓ medium | ✓✓ simple |

---

## What Counts as a Valid Adaptive Advantage

To claim that adaptive methods provide a meaningful advantage, we require:

1. **Coverage gap ≥ 3 percentage points** between static CP and ACI/EnbPI during at least one stress sub-period (COVID crash or 2022 bear). Not just 0.5%.

2. **Rolling coverage plot shows visible divergence** during identifiable crisis dates — not just statistical significance in a table.

3. **Width efficiency is not catastrophically worse:** ACI should not be 3× wider than static CP during calm periods to maintain coverage during stress. If it is, the adaptation is over-aggressive.

4. **Decision layer shows measurable improvement** in portfolio metrics using ACI intervals vs static CP intervals.

If only criteria 1 and 2 are met (not 3 or 4), we report ACI as "statistically superior but practically expensive." This is an honest and defensible finding.

---

## Implementation Priority Ranking

| Priority | Method | Status | Rationale |
|---|---|---|---|
| 1 | Split Conformal (static) | **MANDATORY** | Simplest, needed as the baseline for all other comparisons |
| 2 | ACI | **MANDATORY** | Highest theoretical value; ~25 lines of code; required for "excellent" criterion |
| 3 | Conformalized Quantile Regression (CQR) | **MANDATORY** | Bridges QR baseline with conformal framework; low implementation cost via MAPIE |
| 4 | EnbPI | **STRETCH** | Highest implementation risk; add only after 1–3 are fully validated and producing results |
