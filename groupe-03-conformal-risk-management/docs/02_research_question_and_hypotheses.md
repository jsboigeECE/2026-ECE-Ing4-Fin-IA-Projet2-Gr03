# 02 — Research Question and Hypotheses

---

## Primary Research Question

> Does adaptive conformal prediction — specifically ACI (Adaptive Conformal Inference) and EnbPI (Ensemble Batch Prediction Intervals) — maintain stronger empirical coverage control than static conformal and parametric interval baselines during financial stress periods, and does this advantage translate into a measurable improvement in a VaR-based risk management rule?

This question is deliberately two-part: methodological validity + finance utility. A project that only answers the first half is incomplete. A project that only answers the second half is not rigorous.

---

## Main Hypothesis

**H1 (Coverage Degradation Under Stress):**  
Static split conformal prediction suffers significant marginal coverage degradation during stress regimes (defined as VIX > 30 or rolling realized volatility in the top 10% of the full sample), while ACI and EnbPI maintain coverage closer to the nominal level α by construction.

*Mechanism:* Split conformal relies on residuals from a fixed calibration window. When the data-generating process shifts (fat tails, volatility clustering), the calibration quantile becomes stale. ACI updates the miscoverage rate online; EnbPI uses rolling ensembles. Both are designed to track non-stationarity.

*Expected evidence:* Rolling 60-day coverage for ACI/EnbPI stays within [α−0.05, α+0.05] during crisis sub-periods; static conformal drops below α−0.10.

---

## Secondary Hypotheses

**H2 (Width Efficiency in Normal Regimes):**  
During normal regimes (low volatility), ACI and EnbPI produce wider intervals than static conformal due to online adaptation overhead and inflation factors. This is an expected cost, not a failure.

*Expected evidence:* Average interval width ratio ACI/Static CP > 1.0 during calm periods (VIX < 18).

**H3 (GARCH Parametric Failure Under Tail Events):**  
GARCH(1,1)-based Gaussian intervals systematically undercover during tail events (actual return falls outside the interval more than 1−α of the time during stress), confirming the distribution assumption failure in crisis settings.

*Expected evidence:* GARCH Kupiec test p-value < 0.05 during 2008 and 2020 sub-periods; conformal methods pass at α=0.90.

**H4 (Quantile Regression as a Competitive Baseline):**  
Pinball-loss-optimized quantile regression (via gradient boosted trees or linear QR) provides a competitive but less theoretically grounded baseline. Its coverage may be comparable to static conformal on average, but it lacks the adaptive correction mechanism.

*Expected evidence:* Quantile regression has comparable aggregate coverage to static CP, but higher coverage volatility in rolling analysis.

**H5 (Decision Layer Improvement):**  
A position-sizing rule that scales exposure inversely with normalized conformal interval width produces better risk-adjusted returns (higher Sharpe or lower maximum drawdown) than a static buy-and-hold or fixed-fraction rule, on the validation set.

*Expected evidence:* Interval-width-aware rule reduces maximum drawdown by at least 15% relative to passive benchmark during test period, without proportionally sacrificing average return.

---

## Null Hypotheses

| Null Hypothesis | Test Method | Rejection Criterion |
|---|---|---|
| H0-1: All methods produce equivalent coverage under stress | Proportion test per sub-period | Any adaptive method significantly outperforms static, p < 0.05 |
| H0-2: Interval widths across methods are equivalent | Wilcoxon signed-rank on width series | Significant width difference at α = 0.05 |
| H0-3: Decision rule adds no value vs buy-and-hold | Sharpe ratio test / t-test on returns | Adaptive rule Sharpe > passive Sharpe, p < 0.10 |
| H0-4: ACI and EnbPI are equivalent to each other | Coverage + width pairwise comparison | Statistically significant difference in either metric |

---

## What Empirical Evidence Would Support or Reject Each Hypothesis

### Supporting H1:
- Rolling 60-day coverage plot shows ACI/EnbPI tracking the nominal level better than static CP during 2008, 2020, and 2022 drawdown periods.
- Conditional coverage Kupiec test fails for static CP during stress sub-periods but passes (or is less severe) for ACI.

### Rejecting H1 Would Mean:
- All four conformal methods produce similar rolling coverage — meaning regime shift doesn't actually break the stationarity assumption for this asset/timescale combination.
- This would be an interesting finding: it would suggest SPY daily returns, despite volatility clustering, are close enough to exchangeable that static calibration is sufficient.
- This does NOT kill the project — it strengthens the "empirical audit" framing.

### Supporting H5:
- Equity curve of adaptive rule shows lower drawdown during crisis periods.
- Turnover is not excessive (not a signal that the rule overreacts to noise).

### Rejecting H5 Would Mean:
- Interval width is noisy at daily frequency and does not carry persistent risk signal.
- Mitigation: smooth width with a 5-day or 10-day rolling mean before using as signal.

---

## What Results Are Still Presentable If H1 Fails

If adaptive methods do NOT significantly outperform static conformal under stress:

1. **Finding: The nominal coverage guarantee is more robust than expected.** Static CP is surprisingly resilient on SPY daily data — possibly because SPY's liquidity and market efficiency dampen the worst non-stationarity. This is a publishable result in applied statistics.

2. **Pivot to Width Analysis:** Even if coverage is similar, width dynamics differ. Show that ACI adapts its interval width more smoothly during volatility transitions, while GARCH spikes chaotically.

3. **Strengthen H3:** If GARCH still fails during tail periods (likely), the conformal-vs-parametric story survives even if adaptive-vs-static doesn't.

4. **The Decision Layer Becomes Central:** If any method produces tighter intervals before crises (as a leading signal), the decision layer result can stand independently.

The project has at least three independent stories. We need only one to succeed strongly to be in top-grade territory.
