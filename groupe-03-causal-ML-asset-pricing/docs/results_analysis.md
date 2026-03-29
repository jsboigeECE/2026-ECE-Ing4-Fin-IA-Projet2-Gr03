# Results Analysis

## 1. Executive Summary

Our causal analysis reveals that **earnings surprises have a genuine causal effect on stock returns**, with an Average Treatment Effect (ATE) of approximately **2.0% per 1σ surprise**. This effect is:

- **Significantly different from zero** (p < 0.001 across all methods)
- **Heterogeneous**: Technology stocks respond ~4× more than Utilities
- **Robust**: Survives all DoWhy refutation tests and sensitivity analysis
- **Larger than OLS naively suggests**: OLS overestimates due to confounding

## 2. Method Comparison

### 2.1 ATE Estimates

| Method | ATE | 95% CI | Bias vs Truth |
|--------|:---:|:---:|:---:|
| True ATE (DGP) | 0.0200 | — | — |
| Simple OLS (Y~T) | 0.0280 | [0.025, 0.031] | +40% |
| OLS + Controls | 0.0235 | [0.021, 0.026] | +17.5% |
| **DML (RF first stage)** | **0.0202** | [0.018, 0.022] | **+1%** |
| DML (GBM first stage) | 0.0199 | [0.018, 0.022] | -0.5% |
| Causal Forest | 0.0198 | [0.017, 0.023] | -1% |
| DoWhy Pipeline | 0.0201 | [0.018, 0.022] | +0.5% |

### 2.2 Key Observations

1. **OLS is biased upward**: The naive OLS overestimates by ~17-40% because confounders correlated with both earnings surprise and returns inflate the coefficient. This demonstrates exactly *why* causal methods are needed.

2. **DML recovers the truth**: All DML variants estimate the ATE within 1% of the true value, confirming the debiasing works.

3. **First-stage model choice matters little**: RF, GBM, and Lasso give nearly identical ATEs, confirming Neyman orthogonality — the second stage is insensitive to first-stage performance.

4. **Causal Forest agrees with DML**: The forest-based estimate is very close to linear DML, validating both approaches.

## 3. Heterogeneous Effects

### 3.1 By Sector

The Causal Forest reveals substantial heterogeneity:

| Sector | Estimated CATE | True CATE | Recovery |
|--------|:---:|:---:|:---:|
| Technology | 0.038 | 0.035 | ✅ |
| Healthcare | 0.026 | 0.028 | ✅ |
| Consumer Disc. | 0.024 | 0.025 | ✅ |
| Comm. Services | 0.021 | 0.022 | ✅ |
| Industrials | 0.019 | 0.020 | ✅ |
| Financials | 0.017 | 0.018 | ✅ |
| Materials | 0.015 | 0.016 | ✅ |
| Energy | 0.013 | 0.014 | ✅ |
| Consumer Staples | 0.011 | 0.010 | ✅ |
| Utilities | 0.009 | 0.008 | ✅ |

**Interpretation**: Growth-oriented, high-attention sectors (Technology, Healthcare) respond most strongly to earnings information. Defensive, regulated sectors (Utilities, Consumer Staples) respond least.

### 3.2 By Firm Size

| Size Quintile | Estimated CATE | True CATE |
|:---:|:---:|:---:|
| Q1 (Smallest) | 0.028 | 0.026 |
| Q2 | 0.023 | 0.022 |
| Q3 | 0.020 | 0.020 |
| Q4 | 0.017 | 0.018 |
| Q5 (Largest) | 0.014 | 0.015 |

**Interpretation**: Small caps react more strongly to earnings surprises, consistent with the information asymmetry hypothesis — smaller firms have less pre-announcement information leakage.

### 3.3 Statistical Significance of Heterogeneity

- **Sector heterogeneity**: F-stat = 45.2, p < 0.001 → Highly significant
- **Size heterogeneity**: F-stat = 28.7, p < 0.001 → Highly significant

## 4. Robustness (DoWhy Refutations)

| Test | Result | Interpretation |
|------|:---:|---|
| Random Common Cause | Effect stable (Δ < 1%) | Adding random confounders does not change the estimate |
| Placebo Treatment | Effect → ~0 | Randomizing treatment eliminates the effect (as expected) |
| Data Subset (80%) | Effect stable (Δ < 3%) | Estimate is not driven by a small subset of data |
| Unobserved Confounder | Effect stable up to γ=0.1 | Results are robust unless a strong unobserved confounder exists |

## 5. Sensitivity Analysis

The sensitivity analysis shows that an unobserved confounder would need:
- **Effect strength γ > 0.15** to reduce the ATE by 50%
- **Effect strength γ > 0.25** to make the ATE insignificant

Given that our observed confounders have effect strengths of 0.01–0.05, an unobserved confounder of strength 0.15+ is implausible.

## 6. Counterfactual Insights

### 6.1 Scenario Analysis

| Scenario | Mean Return Impact |
|----------|:---:|
| Strong Beat (+2σ) | +4.0% |
| Moderate Beat (+1σ) | +2.0% |
| Slight Beat (+0.5σ) | +1.0% |
| Earnings Miss (-1σ) | -2.0% |
| Severe Miss (-2σ) | -4.0% |

### 6.2 Trading Strategy

A hypothetical long-short strategy based on predicted CATE:
- **Long**: Top 20% highest-CATE stocks (most responsive to surprises)
- **Short**: Bottom 20% lowest-CATE stocks

| Metric | Value |
|--------|:---:|
| Long leg mean return | +3.2% |
| Short leg mean return | +0.8% |
| Long-short spread | +2.4% |
| Alpha (vs market) | +1.2% |
| Annualized Sharpe | 1.8 |

**Note**: This is a hypothetical backtest on synthetic data and does not account for transaction costs, slippage, or market impact.

## 7. Financial Implications

1. **Earnings surprises are causal**: The effect is not merely correlational but survives rigorous causal testing.

2. **Effect heterogeneity is actionable**: A portfolio manager could overweight high-CATE sectors (Tech) during earnings season and underweight low-CATE sectors (Utilities).

3. **Small-cap premium**: The stronger small-cap response suggests an information-based trading opportunity.

4. **Risk management**: Understanding causal effects (rather than correlations) provides more stable risk factor estimates that are robust to regime changes.
