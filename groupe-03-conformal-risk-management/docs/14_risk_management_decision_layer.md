# 14 — Risk Management Decision Layer

---

## Why This Layer Is Non-Negotiable

A project that stops at coverage tables is a statistical study. A project with a decision layer is an applied finance project. The grading criteria explicitly require a "portfolio application" at the excellent level. This layer delivers exactly that in the simplest, most defensible form.

**The decision layer answers:** "If you believe ACI intervals, and you are a portfolio manager, what should you do differently — and does it pay off?"

---

## The Core Principle: Uncertainty as a Risk Signal

Prediction interval width is a direct measure of model uncertainty. Wide intervals = high uncertainty = high risk. Narrow intervals = low uncertainty = lower risk. A rational risk manager should reduce exposure when uncertainty increases — even before a loss occurs.

This principle is the foundation of:
- Kelly criterion (bet proportionally to expected edge / variance)
- Volatility targeting (adjust position size to hit a constant volatility target)
- Risk parity (equalize risk contribution across positions)

Our rule is a simple, principled implementation of volatility targeting using conformal interval width as the volatility proxy.

---

## The Decision Rule: Interval-Width Position Sizing

### Formal Definition

```
Let W_t = U_t - L_t  = conformal prediction interval width at day t
Let W_ref = median(W_{t ∈ calibration set})  = reference width (calm regime baseline)

position_size_t = min( W_ref / W_t , cap )
```

Where:
- `cap = 1.5` — prevents over-leveraging during apparently calm periods
- `floor = 0.1` — prevents going to near-zero exposure during extreme stress

### Strategy Return

```
strategy_return_t = position_size_{t-1} × r_t
```

Position size computed at time t−1 using interval available before market open on day t. No look-ahead.

### Why This Rule is Optimal for the Project

| Property | Assessment |
|---|---|
| Simple | Yes — 3 lines of code |
| Theoretically motivated | Yes — uncertainty-weighted betting |
| Defensible under questioning | Yes — direct extension of vol targeting |
| Finance-meaningful | Yes — every quant fund uses exposure scaling |
| Implementable with our output | Yes — W_t is produced by all conformal methods |
| No hyperparameter overfitting | Yes — W_ref is computed on calibration, cap/floor are standard |

---

## Baseline Policies for Comparison

**Policy 0 (Buy and Hold):** `return_{BH,t} = r_t`. Full 100% allocation at all times. No risk management.

**Policy 1 (Fixed-Threshold Rule):** Reduce exposure to 50% when VIX > 25, full exposure otherwise. Simple, intuitive, but uses VIX (an observable signal) rather than the model's uncertainty.

**Policy 2 (Static CP Width Sizing):** Position sizing using Split CP interval widths. Tests whether any conformal-based sizing adds value.

**Policy 3 (ACI Width Sizing):** Position sizing using ACI interval widths. The primary test — does adaptive conformal produce better uncertainty signals for risk management?

**Policy 4 (GARCH Vol Target):** Position sizing using GARCH conditional volatility estimate. The industry benchmark for vol targeting. ACI should beat this during crisis periods.

---

## Evaluation of the Decision Layer

### Primary Metrics (Test Set 2020–2024)

| Metric | Formula | Why It Matters |
|---|---|---|
| Annualized Sharpe Ratio | `mean(r_strategy) / std(r_strategy) × √252` | Risk-adjusted return |
| Maximum Drawdown | `max peak-to-trough loss` | Tail risk protection |
| Calmar Ratio | `Annualized Return / Max Drawdown` | Combined performance |
| Average Position Size | `mean(position_size_t)` | Sanity check — should be ~0.7–1.0 |
| Turnover | `mean(|position_size_t - position_size_{t-1}|)` | Transaction cost proxy |

### Expected Results

| Policy | Expected Sharpe | Expected MaxDD | Expected Calmar | Notes |
|---|---|---|---|---|
| Buy and Hold | ~0.65 | ~−34% (COVID) | ~0.3 | Reference |
| VIX Threshold | ~0.70 | ~−22% | ~0.5 | Simple benchmark |
| Static CP Sizing | ~0.72 | ~−25% | ~0.45 | Static CP has lagged width signal |
| ACI Sizing | ~0.78 | ~−20% | ~0.6 | ACI reacts faster to uncertainty spikes |
| GARCH Vol Target | ~0.74 | ~−21% | ~0.55 | Strong benchmark — GARCH vol is responsive |

**Note:** These are pre-implementation hypotheses. The actual results will differ. If ACI sizing does NOT beat GARCH vol targeting, the correct conclusion is: "At daily frequency, GARCH-based uncertainty signals are competitive with conformal uncertainty signals for position sizing — but conformal provides formal coverage guarantees that GARCH cannot." This is still a defensible finding.

---

## Crisis Performance Focus

The most important sub-period analysis: **February–April 2020 (COVID crash).**

**Expected sequence for ACI sizing:**
1. Pre-crash: intervals narrow (calm market, VIX ~15), position size near cap (1.5× base)
2. First exceptions (Feb 24–28): α_t begins falling, intervals begin widening
3. Peak crisis (Mar 9–23): α_t at minimum, intervals at maximum width, position size near floor (0.1× base)
4. Recovery (Apr): α_t recovers, intervals narrow, position size increases

**Expected sequence for GARCH vol targeting:**
1. Pre-crash: narrow intervals, full exposure
2. First shock: GARCH σ_t updates overnight, exposure reduces with 1-day lag
3. Peak crisis: exposure responding to yesterday's vol, still partially exposed
4. Recovery: lagging exposure reduction may continue after crash is over

**Expected sequence for Static CP:**
1. Pre-crash: fixed width from calibration (2015–2017 calm period), full exposure
2. Crisis: width NEVER changes (calibration quantile is frozen), stays fully invested
3. Post-crisis: still looks the same — static CP has NO crisis response at all

**This stark contrast (ACI reduced exposure, Static CP stayed invested) is the most dramatic and finance-relevant finding in the entire project.**

---

## What This Layer Is NOT

- NOT a backtested trading strategy claiming alpha generation
- NOT a demonstration that conformal prediction improves market-timing
- NOT a claim that the strategy is profitable after transaction costs
- NOT a recommendation to use this rule in live trading

**The claim is narrow and defensible:** "Uncertainty-aware position sizing using ACI intervals reduces drawdown during the test period relative to buy-and-hold and static CP sizing, at a modest cost to average exposure." Nothing more.

---

## Implementation Blueprint

```
INPUTS:
  - interval_series[method][alpha]: DataFrame with columns [date, lower, upper, width]
  - spy_returns_test: Series of daily log-returns in test set

COMPUTATION:
  1. w_ref = median(calibration_widths)  (from calibration set, for each method separately)
  2. position_t = clip( w_ref / width_t, 0.1, 1.5 )  (applied to t-1 position for day t)
  3. strategy_returns_t = position_{t-1} × spy_returns_t
  4. compute_metrics(strategy_returns_t) → Sharpe, MaxDD, Calmar

OUTPUT:
  - results/decision_layer_metrics.csv
  - figures/equity_curves.png  (5 policies × test period)
  - figures/position_sizing_covid.png  (zoom into Feb–Apr 2020)
```
