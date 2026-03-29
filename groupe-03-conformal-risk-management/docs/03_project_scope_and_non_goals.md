# 03 — Project Scope and Non-Goals

---

## What This Project Is

A focused, reproducible, empirical study of conformal prediction interval methods applied to SPY daily log-return forecasting, evaluated against parametric and quantile baselines, with explicit stress-period analysis and a single concrete risk management decision layer.

---

## Scope Definition (Hard Boundaries)

### Assets
- **Exactly one primary asset:** SPY (SPDR S&P 500 ETF Trust)
- **Exactly one secondary asset for context only:** VIX index (used as a regime indicator, not a forecast target)
- No other assets are in scope

**Rationale:** SPY is the most liquid, most studied equity instrument. Daily data is clean, free, and unambiguous. VIX provides a credible regime signal without requiring a second full forecasting pipeline. Adding GLD, BTC, or individual stocks adds noise, complexity, and fragility for zero grading gain.

### Target Variable
- **Exactly one:** SPY 1-day ahead log-return: `r_t = log(P_t / P_{t-1})`
- This maps cleanly to 1-day VaR estimation

### Time Period
- **Full span:** January 2004 — December 2024 (approximately 21 years, ~5,200 trading days)
- **Train:** 2004–2014 (10 years)
- **Calibration:** 2015–2017 (3 years)
- **Validation:** 2018–2019 (2 years, includes Dec 2018 drawdown for early stress signal)
- **Test:** 2020–2024 (4.5 years, includes COVID crash and 2022 rate-hike bear market)

### Methods In Scope
| Method | Role |
|---|---|
| Ridge Regression | Primary point forecast base learner |
| Historical Simulation VaR | Parametric baseline 1 |
| GARCH(1,1) Gaussian VaR | Parametric baseline 2 |
| Quantile Regression (Linear) | Interval baseline |
| Split Conformal Prediction | Static conformal baseline |
| EnbPI | Time-series adaptive conformal |
| ACI (Adaptive Conformal Inference) | Online adaptive conformal |

### Decision Layer
- Exactly one rule: volatility-scaled position sizing based on normalized conformal interval width
- Evaluated on Sharpe ratio and maximum drawdown during the test period

---

## What This Project Explicitly Is NOT

### Not a Portfolio Optimization Project
No multi-asset allocation, no covariance estimation, no mean-variance frontier. The portfolio application mentioned in the "excellent" grading criteria is satisfied by the decision layer (position sizing = single-asset portfolio construction).

**Why:** Portfolio optimization adds 3× the implementation complexity for no additional methodological insight. The decision layer achieves the same grading point with 20% of the effort.

### Not a Deep Learning Project
No LSTM, Transformer, TCN, or neural network of any kind. The base forecaster is Ridge Regression.

**Why:** Deep learning adds hyperparameter sensitivity, training instability, and explainability debt. The conformalization is the contribution — the base learner is intentionally simple and auditable. Any reviewer who asks "why not LSTM?" gets a precise answer: "base learner variance is irrelevant to conformal coverage — we chose Ridge for reproducibility and speed."

### Not a Real-Time System
No streaming data pipeline, no API integration, no live trading simulation. Everything is batch offline evaluation.

**Why:** Live execution is a completely different engineering problem. It adds no academic value and creates massive delivery risk.

### Not an Options Pricing Project
No Black-Scholes derivatives, no implied volatility surface, no Greeks.

**Why:** Option pricing requires a completely different data pipeline, is not needed to demonstrate conformal interval validity, and diverts focus from the core VaR story.

### Not a Multi-Horizon Study
No 5-day or 10-day ahead forecasting. Only 1-day horizon.

**Why:** Multi-horizon forecasting compounds errors, complicates evaluation, and dilutes the conformal coverage analysis. 1-day is the standard regulatory horizon for VaR.

### Not a Hyperparameter Search Study
No grid search over feature sets, model architectures, or conformal parameters across every method.

**Why:** This is not an AutoML project. One clean, pre-specified model per method, with fixed hyperparameters chosen from theory or simple cross-validation on the training set only.

### Not a Full Bayesian Comparison
No full MCMC posterior inference, no variational inference, no probabilistic programming framework.

**Why:** Full Bayesian inference on a time series requires substantial implementation time and is not the focus. The comparison is framed as: "conformal makes fewer assumptions and is faster — is it also better calibrated?" A sentence referencing Gneiting & Raftery (2007) is sufficient as a theoretical positioning.

---

## Anti-Overengineering Rules (Enforced)

These are binding constraints, not suggestions:

1. **Rule of One:** One asset, one target, one decision rule. Whenever tempted to add a second, articulate the grading gain first. If you cannot name a specific grading gain, reject it.

2. **Rule of Simplicity:** If a method requires more than 50 lines of non-boilerplate code to implement, question whether it belongs in scope.

3. **Rule of Defensibility:** Every architectural choice must survive a 30-second hostile explanation. "We did it because it seemed interesting" is not a valid answer.

4. **Rule of Chronology:** All train/calibration/test splits are strictly chronological. No random shuffling, no k-fold cross-validation on returns data. Ever.

5. **Rule of Completeness over Breadth:** A complete, rigorously evaluated experiment on one method beats a superficial comparison of five methods. Depth wins.

6. **Rule of No Cosmetic Complexity:** No dashboards, Streamlit apps, interactive widgets, custom CSS, or anything that does not directly serve the scientific comparison.

---

## Scope Decisions Explicitly Made

| Decision | Choice Made | Alternative Rejected | Reason |
|---|---|---|---|
| Asset | SPY only | SPY + GLD or SPX futures | Cleaner data, single pipeline, no leakage risk |
| Frequency | Daily | Intraday (1min, 5min) | Intraday requires microstructure handling, bid-ask correction; regulatory VaR is daily |
| Base learner | Ridge Regression | Random Forest, XGBoost | Interpretable, fast, numerically stable residuals for conformal |
| Conformal library | MAPIE | Custom implementation | MAPIE is well-tested and industry-relevant; custom wastes time |
| Decision layer | Position sizing | Delta hedging, stop-loss | Simplest instrument that demonstrates RM utility |
| Bayesian comparison | GARCH as proxy | Full MCMC | GARCH IS the industry Bayesian-in-practice for volatility |
