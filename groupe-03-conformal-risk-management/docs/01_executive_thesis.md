# 01 — Executive Thesis

**Project:** Adaptive Conformal Prediction for Financial Risk Management  
**Team:** Group 03 — Lewis, Orel, Thomas Nassar  
**Institution:** ECE Paris — Ing4 Finance / IA Probabiliste  
**Topic Reference:** A.5 — Conformal Prediction pour Risk Management

---

## One-Line Research Thesis

> This project tests whether adaptive conformal prediction (ACI and EnbPI) maintains better empirical coverage control than static conformal and parametric baselines (GARCH, Historical Simulation) during financial stress periods — and whether this reliability difference, if confirmed, translates into improved risk management via uncertainty-aware position sizing on SPY daily returns.

---

## The Promise of This Project

This project delivers three things that no ordinary student project delivers simultaneously:

1. **Statistical soundness**: every interval method we test carries a formal coverage guarantee or is explicitly evaluated against its failure conditions under regime shift.
2. **Finance relevance**: the output is not a generic forecasting comparison — it directly maps to the VaR problem that every risk desk operates daily, with a realistic decision layer that tests practical utility.
3. **Methodological honesty**: we do not just show which method "wins". We rigorously evaluate where each method struggles, with a particular focus on the 2020 COVID crash and the 2022 rate-shock bear market — the stress periods in our test set.

---

## Why This Topic Matters in Finance

Value-at-Risk remains the regulatory anchor of Basel III/IV. Yet the dominant implementations — parametric GARCH, historical simulation — carry implicit distribution assumptions that regularly fail during tail events. Bayesian alternatives are theoretically elegant but computationally fragile and require prior specification.

Conformal prediction enters this space as a **model-agnostic, distribution-free** framework. It makes exactly one assumption: exchangeability (or a controlled relaxation thereof for time series). This is a weaker assumption than normality, GARCH stationarity, or any Bayesian prior.

The core academic question is: **does this theoretical advantage survive contact with real financial data?** Specifically, does it survive regime change?

This project forces that question into an empirical test under rigorous conditions.

---

## Why This Project is High Academic ROI

| Dimension | Why This Works |
|---|---|
| Theory | Conformal prediction has a strong, recent mathematical literature (Vovk, Tibshirani, Barber, Angelopoulos). We cite real papers. |
| Finance | VaR backtesting has a precise regulatory framework (Kupiec, Christoffersen). We borrow their language. |
| Empirics | SPY daily data is publicly available, clean, and has documented crisis periods. No data procurement risk. |
| Execution | The full pipeline can be implemented with scikit-learn + MAPIE + scipy. No exotic dependencies. |
| Presentation | The central question is compelling and falsifiable: "Does adaptive conformal maintain coverage when static methods fail?" |
| Defense | Every design decision has a one-sentence justification. No apologetics needed. |

---

## What Must Be True for the Project to Feel Exceptional

The project earns top-grade territory if and only if:

- [ ] Coverage tables show a **real, interpretable difference** between static and adaptive methods during stress periods — not just marginal noise.
- [ ] The ACI method **demonstrably recovers** after the initial shock of a regime shift, while static conformal does not.
- [ ] The VaR decision layer shows that interval width carries **risk-relevant information** — not just interval coverage.  
- [ ] The rolling coverage plot is **legible and dramatic** — a key visual that makes the argument without explanation.
- [ ] Every methodological choice can be defended in under 30 seconds under hostile questioning.
- [ ] The repository looks like a quant research output, not a homework submission.

---

## What Would Make This Project Mediocre

- Reporting only average coverage without conditional or rolling analysis.
- Failing to define a concrete finance decision layer — ending at coverage tables.
- Using deep learning as the base forecaster (adds variance, hurts explainability, loses time).
- Adding too many assets and losing depth for breadth.
- Showing results only on the full test period without isolating crisis sub-periods.
- Treating conformal prediction as a "new" method without connecting to the existing VaR backtesting literature.
- A README that looks like a notebook dump.

---

## Scope in One Sentence

One asset (SPY), one target (1-day log-return), one use case (VaR), four interval methods (Split CP, EnbPI, ACI, Quantile Regression), two parametric baselines (GARCH, Historical Simulation), rolling evaluation with crisis-period segmentation, and one position-sizing decision rule — executed cleanly and defended crisply.
