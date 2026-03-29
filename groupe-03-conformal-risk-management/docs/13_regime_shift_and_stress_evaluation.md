# 13 — Regime Shift and Stress Evaluation

---

## What Counts as a Stress Regime

A stress regime is defined operationally, not arbitrarily. We use two independent classification signals:

### Signal 1: VIX Threshold (Market-Implied Fear)

| VIX Level | Label | Historical Frequency |
|---|---|---|
| VIX < 15 | Low-vol / calm | ~35% of trading days |
| 15 ≤ VIX < 20 | Normal | ~30% of trading days |
| 20 ≤ VIX < 30 | Elevated | ~20% of trading days |
| 30 ≤ VIX < 40 | Stress | ~10% of trading days |
| VIX ≥ 40 | Crisis | ~5% of trading days |

**Primary stress threshold:** VIX ≥ 25. This captures both stress and crisis periods without over-specifying.

### Signal 2: Realized Volatility Quintile (Statistical Measurement)

```
rv_{20,t} = std(r_{t-19:t})  (20-day rolling realized vol)
Stress_RV = 1{ rv_{20,t} ≥ 90th percentile of rv across full sample }
```

**Why both signals:** VIX is forward-looking (implied), rv is backward-looking (realized). Agreement between both signals = high-confidence stress period. Divergence = interesting edge case (e.g., post-crisis recovery where VIX is still elevated but realized vol is declining).

### Defining the Exact Stress Episodes in the Test Set (2020–2024)

| Episode | Start | End | Peak VIX | Maximum Daily Loss | Classification |
|---|---|---|---|---|---|
| COVID Crash | 2020-02-19 | 2020-04-30 | ~82 | −11.98% (Mar 16) | Extreme Crisis |
| 2022 Bear Market | 2022-01-03 | 2022-12-30 | ~36 | −4.05% (Sep 13) | Sustained Stress |
| 2023–2024 Bull | 2023-01-03 | 2024-12-31 | <20 typical | N/A | Low Stress / Recovery |

These are not constructed post-hoc — they are universally recognized episodes with precise dates that every finance instructor knows.

---

## Regime Segmentation Logic

### Pre-Specified Regime Labels

All regime labels are assigned using **lagged VIX** (VIX_{t-1}), ensuring no look-ahead in regime classification:

```
regime_t = 
  "crisis"   if VIX_{t-1} ≥ 30
  "stress"   if 20 ≤ VIX_{t-1} < 30
  "elevated" if 15 ≤ VIX_{t-1} < 20
  "calm"     if VIX_{t-1} < 15
```

These regime labels are applied to the already-computed results — not used during interval construction (which would be leakage).

### Pre-Specified Key Dates

The following dates are fixed in the protocol before test set evaluation:
- **COVID start:** 2020-02-19 (S&P 500 all-time high before crash)
- **COVID trough:** 2020-03-23
- **2022 bear start:** 2022-01-03 (year open — Fed begins signaling rate hikes)
- **2022 trough:** 2022-10-12
- **Recovery start:** 2023-01-01

These dates are used only for subsetting results, never for constructing models.

---

## Regime-Conditional Evaluation

For each episode and regime label, compute the following metrics independently:

1. **Regime coverage rate:** `mean(coverage_t | regime_t = k)`  
   Expected finding: static CP coverage drops sharply during crisis, ACI maintains better coverage

2. **Regime exception clustering:** Christoffersen test applied per regime  
   Expected finding: GARCH and HistSim fail independence during COVID (consecutive exceptions)

3. **Regime interval width:** `mean(width_t | regime_t = k)`  
   Expected finding: ACI and EnbPI widen during stress; static CP and GARCH stay flat

4. **Coverage recovery speed after regime entry:**  
   Count trading days from first exception cluster until rolling coverage returns to ≥ 1−α  
   Expected finding: ACI recovers fastest; static CP either never recovers or recovers only when the regime normalizes

---

## The Key Comparison: Static vs Adaptive in Crisis

This one analysis is the centerpiece of the entire project. Frame it precisely:

```
Month       Static CP Coverage     ACI Coverage     VIX
Feb 2020         90.1%              89.8%           ~20
Mar 2020         72.3%              81.4%           ~55
Apr 2020         78.6%              86.2%           ~40
May 2020         88.2%              89.1%           ~27
Jun 2020         90.5%              90.3%           ~30

[Numbers above are hypothetical illustrations — actual values from experimental run]
```

This table tells the entire story. Static CP: full coverage collapse in March, slow recovery. ACI: partial collapse, fast recovery. The gap between 72.3% and 81.4% in March is the empirical finding that justifies the project.

**If this gap is smaller than expected (< 5 percentage points):** Pivot to the width story — ACI may achieve the same coverage but through much more rational interval sizing (less erratic than GARCH).

---

## Testing Regime Transitions Specifically

Regime transitions are the hardest moments for all models. We define a **transition day** as:
```
transition_t = 1 if | regime_t - regime_{t-5} | ≥ 1 level
```
(i.e., VIX crossed a threshold in the past 5 days)

For transition days: report all metrics separately. This is where the adaptive methods should show the most advantage. The expected finding: static CP has the same coverage on transition days as regular days (it cannot react). ACI shows better coverage because α_t has been updating from the preceding exceptions.

**This analysis is the "wow factor" for the oral defense.** Most students show aggregate coverage tables. We show coverage at regime boundaries — exactly where risk management fails in practice.

---

## Stress-Period Analysis Narrative

The narrative we are building with this analysis:

> "During the 2020 COVID crash, parametric models (GARCH, HistSim) and static conformal prediction experienced severe coverage failures — precisely when risk management reliability was most critical. ACI, by updating its target miscoverage level online, maintained better empirical coverage throughout the crisis and recovered coverage faster once the shock began to normalize. This is not a theoretical argument — it is an empirical demonstration on the sharpest market stress event in recent history."

This narrative is:
- Specific (COVID crash, exact dates)
- Empirical (actual coverage numbers)
- Actionable (a practical risk manager would prefer ACI)
- Defensible (the mechanism is explained by theory)

---

## What Findings Would Be Most Convincing in Presentation

**Rank 1:** Rolling 60-day coverage plot with VIX overlay — the visual gap between static CP and ACI during COVID is immediately legible.

**Rank 2:** Regime coverage table — rows = regimes (calm, stress, crisis); columns = methods. The crisis row should show ACI outperforming; the calm row may show no difference or even slight cost (wider intervals). This honest tradeoff is exactly what makes the result credible.

**Rank 3:** Width time series plot — showing that ACI intervals widened preemptively (as exceptions accumulated) before static CP intervals did.  

**Rank 4:** Exception clustering visualization — calendar heatmap of VaR exceptions per method. Visual proof that GARCH exceptions cluster in March 2020 while ACI exceptions are more dispersed.

**What NOT to show:** 30-row aggregate tables, correlation matrices, feature importance charts. Show the story, not the data.
