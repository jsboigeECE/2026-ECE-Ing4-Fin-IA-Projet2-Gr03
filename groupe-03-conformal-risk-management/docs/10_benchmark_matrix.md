# 10 — Benchmark Matrix

---

## Full Comparison Table

| Method | Category | Coverage Guarantee | Width Efficiency | Regime Adaptiveness | Theoretical Grounding | Finance Relevance | Implementation Cost | Grading Value | Status |
|---|---|---|---|---|---|---|---|---|---|
| Historical Simulation VaR | Parametric | None (finite-sample) | Moderate | None — lagging 252-day window | Industry standard; Basel I/II | Very High — universal practice | Trivial | High (baseline anchor) | MANDATORY |
| GARCH(1,1) Gaussian | Parametric | None (model-dependent) | Good in calm; narrow in tails | Moderate — conditional vol update | Academic + industry standard | Very High — risk desk staple | Low | High (Gaussian tail critique) | MANDATORY |
| Linear Quantile Regression | Model-based | None (training-set dependent) | Good — pinball-optimal in-sample | None — refitted periodically | Koenker-Bassett (1978) | High — used in CLEaR, IM models | Low | High (distribution-free baseline) | MANDATORY |
| Split Conformal Prediction | Conformal | Marginal, distribution-free | Moderate — fixed calibration quantile | None — static one-time calibration | Vovk (2005), Tibshirani (2019) | Growing — cited in recent risk papers | Trivial (MAPIE) | Very High — core method | MANDATORY |
| Conformalized QR (CQR) | Conformal + QR | Marginal, distribution-free | Best among static conformal — locally adaptive | None — static calibration of QR scores | Romano et al. (2019) | High — adaptive width without online update | Low (MAPIE) | High — bridges QR and conformal | MANDATORY |
| EnbPI | Conformal | Approximate; empirical | Good — tracks recent volatility | High — rolling window update | Xu & Xie (2021) JMLR | High — designed for time series | Medium | Very High — explicit temporal adaptation | MANDATORY |
| ACI (Adaptive Conformal Inference) | Conformal | Long-run average guarantee | Moderate — may oscillate | Very High — online step update | Gibbs & Candès (2021) NeurIPS | High — strongest non-stationarity robustness | Low (custom 20 lines) | Maximum — "excellent" criterion key method | MANDATORY |

---

## Method Details by Dimension

### Coverage Guarantee

| Method | Type of Coverage | Assumption Required | Fail Condition |
|---|---|---|---|
| Historical Simulation | None formal | IID returns | Volatility clustering, regime shift |
| GARCH Gaussian | Model-conditional | Normality, stationarity | Fat tails, structural breaks |
| Linear QR | Asymptotic, in-sample | Correct model specification | Misspecification; out-of-sample distribution shift |
| Split CP | Finite-sample marginal ≥ 1−α | Exchangeability | Non-stationarity, regime shift |
| CQR | Finite-sample marginal ≥ 1−α | Exchangeability of CQR scores | Same as Split CP |
| EnbPI | Empirical; asymptotically valid | Near-stationarity in rolling window | Very rapid regime shifts |
| ACI | Long-run time-average → α | None (distribution-free) | Very small γ + very persistent shift |

### Theoretical Pedigree

| Method | Primary Reference | Impact | Key Claim |
|---|---|---|---|
| Historical Sim. | J.P. Morgan RiskMetrics (1994) | Industry-defining | Simple empirical quantile = sufficient |
| GARCH | Bollerslev (1986) | ~15,000 citations | Time-varying volatility parametrically |
| QR | Koenker & Bassett (1978) | ~20,000 citations | Optimal under asymmetric Laplace loss |
| Split CP | Papadopoulos et al. (2002); Vovk (2005) | Core of conformal theory | Distribution-free marginal coverage |
| CQR | Romano, Patterson, Candès (2019) | NeurIPS, ~500 citations | Locally adaptive conformal |
| EnbPI | Xu & Xie (2021) | JMLR, finance-focused | Rolling conformal for time series |
| ACI | Gibbs & Candès (2021) | NeurIPS, ~300 citations | Online CP under distribution shift |

### Runtime Cost (Relative)

| Method | Per-Step Cost | Full Test Set (1260 steps) | Parallelizable |
|---|---|---|---|
| Historical Sim. | Microseconds | < 1 second | N/A |
| GARCH | ~50ms per refit | ~1 minute | No (sequential) |
| Linear QR | ~5ms per refit | < 10 seconds | No (sequential) |
| Split CP | < 1ms per step | < 1 second | Yes |
| CQR | < 1ms per step | < 1 second | Yes |
| EnbPI | ~100ms per step (ensemble) | ~2 minutes | Partially |
| ACI | < 1ms per step | < 1 second | Yes |

**Total runtime estimate:** < 5 minutes for the full evaluation pipeline. No GPU needed. No cloud compute needed.

### Finance Relevance Score (1–5)

| Method | Score | Justification |
|---|---|---|
| Historical Simulation | 5/5 | The most-used VaR method in the world |
| GARCH | 5/5 | Basel II internal model standard |
| Linear QR | 3/5 | Used in FRTB Expected Shortfall approximations |
| Split CP | 3/5 | Emerging in risk management literature |
| CQR | 3/5 | Extension — inherits QR finance relevance |
| EnbPI | 4/5 | Directly designed for non-stationary financial series |
| ACI | 4/5 | Strongest theoretical foundation for regime robustness |

### Grading Value Assessment

| Method | Theory Points | Empirical Points | Finance Points | Wow Factor | Total |
|---|---|---|---|---|---|
| Historical Sim. | Low | Medium | High | Low | Medium |
| GARCH | Medium | Medium | High | Low | Medium-High |
| Linear QR | Medium | Medium | Medium | Low | Medium |
| Split CP | High | High | Medium | Medium | High |
| CQR | High | High | Medium | Medium | High |
| EnbPI | Very High | High | High | High | Very High |
| ACI | Very High | High | High | Very High | Maximum |

---

## Methods Explicitly Excluded from Benchmark

| Method Considered | Why Excluded |
|---|---|
| Student-t GARCH | Would perform too well — removes the parametric failure narrative |
| GARCH with GJR / EGARCH | Asymmetric vol modeling is interesting but adds implementation complexity without grading gain |
| Bayesian structural time series (Prophet) | Poor coverage guarantees; adds library dependency; weak finance interpretation |
| Jackknife+ | Requires IID assumption more strongly than split CP — worse for time series |
| RAPS (Regularized ACI variant) | Very new (2023); limited tooling; marginal gain over ACI |
| Deep AR / Neural AR | See Kill List — deep learning rejected categorically |
| Monte Carlo simulation VaR | Requires full assumption of return distribution — reduces to parametric |
| Expected Shortfall (ES) | Elicitatbility issues; evaluation of ES requires additional framework; VaR is sufficient for scope |

---

## Summary Verdict

**Non-negotiable methods (must run, must report):** Historical Simulation, GARCH, Linear QR, Split CP, ACI  
**Highly recommended (high ROI):** CQR, EnbPI  
**Under no circumstances (Kill List):** Deep learning variants, Prophet, Bayesian MCMC, Student-t GARCH as exclusive method
