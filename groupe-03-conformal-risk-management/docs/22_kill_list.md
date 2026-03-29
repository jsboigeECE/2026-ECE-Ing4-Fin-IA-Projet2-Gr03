# 22 — Kill List

## Purpose

This document exists to protect the project from self-sabotage. Every item below is an idea that seems attractive — intellectually interesting, technically impressive, or academically credible — but would actively harm this project's grade-per-hour ratio and overall quality. These items are killed permanently.

If anyone on the team suggests implementing an item from this list, the correct response is: "It's on the kill list. We decided."

---

## KILL 1 — LSTM / Transformer as Base Learner

**Why it's tempting:** Deep learning on financial time series sounds sophisticated. The base learner quality affects residual structure, and a better model should give better conformal scores.

**Why it must be killed:**
- Training instability means results may not reproduce reliably — fatal for a reproducibility-focused project
- Requires GPU or long training time in the walk-forward loop
- LSTM residuals are harder to characterize theoretically
- The contribution of this project is the conformal layer, NOT the forecasting quality
- A reviewer who asks "why LSTM?" will then spend the entire Q&A on LSTM hyperparameters instead of the coverage analysis
- If LSTM happens to overfit, the coverage analysis is invalidated
- Ridge Regression is a stronger methodological statement: "we chose simplicity because the base learner is not the point"

**Verdict: Do not implement. Do not discuss. If asked during defense, use the prepared answer in doc 19.**

---

## KILL 2 — GLD, BTC, or Any Second Asset as Full Pipeline

**Why it's tempting:** The "excellent" grading criterion mentions portfolio applications. Adding a second asset feels like it addresses that criterion.

**Why it must be killed:**
- A second asset requires doubling the data pipeline, feature engineering, model fitting, conformal calibration, and evaluation
- Portfolio VaR requires covariance estimation — a separate major topic
- Results for a second asset would be shallower than results for one asset, reducing depth
- The decision layer (position sizing on SPY with uncertainty-weighted exposure) ALREADY satisfies the portfolio application criterion
- VIX is already included as a regime signal — that's enough for "multi-signal" framing

**Verdict: One asset. SPY. Forever.**

---

## KILL 3 — Multi-Horizon Forecasting (5-day, 10-day)

**Why it's tempting:** Basel III uses 10-day VaR for capital calculation. Including a 10-day horizon feels more institutionally relevant.

**Why it must be killed:**
- Multi-step ahead conformal prediction requires compounding uncertainty — a non-trivial extension that invalidates the simple coverage guarantee derivation
- Overlapping return windows (5-day or 10-day) introduce autocorrelation that requires Hansen-Hodrick correction — a complete detour from the main story
- The industry approximates 10-day VaR by scaling 1-day VaR by √10 — which we can mention in one sentence without implementing it
- Adding multi-horizon doubles the evaluation matrix for zero additional methodological insight

**Verdict: 1-day horizon only. Mention the √10 scaling as a footnote.**

---

## KILL 4 — Streamlit / Interactive Dashboard

**Why it's tempting:** A live interactive dashboard looks impressive in a demo and might increase "wow factor."

**Why it must be killed:**
- Dashboards are engineering, not research — they demonstrate Streamlit proficiency, not statistical rigor
- A broken dashboard during a demo is catastrophically worse than no dashboard
- All key results are already captured as static figures that are reproducible in two commands
- The time cost of building and deploying a dashboard could instead produce two additional analysis notebooks
- No professional research paper has a Streamlit app — they have figures and CSV files

**Verdict: No dashboard. Pre-generated PNG figures only. Demo = showing figures and running the experiment script.**

---

## KILL 5 — Bayesian MCMC / Full Posterior Inference

**Why it's tempting:** Conformal vs. Bayesian is listed as a comparison criterion in the excellent-level grading. Full Bayesian uncertainty quantification would make the comparison rigorous.

**Why it must be killed:**
- Full MCMC requires PyMC or Stan — a new library dependency with steep setup cost
- MCMC in a walk-forward loop requires re-sampling at every step — computationally prohibitive
- We already include GARCH as a "parametric Bayesian-in-spirit" baseline — which is the practical industry analog of Bayesian inference for volatility
- The theoretical argument (conformal requires no prior; Bayesian requires prior specification) can be made in one paragraph without implementation
- A sentence citing Gneiting & Raftery (2007) on proper scoring rules is sufficient academic positioning

**Verdict: Use GARCH as the parametric/Bayesian proxy. Frame the argument theoretically in one paragraph. Do not implement MCMC.**

---

## KILL 6 — Student-t GARCH as Primary Parametric Baseline

**Why it's tempting:** Student-t GARCH is more realistic than Gaussian GARCH. It would actually perform reasonably well on fat-tailed data.

**Why it must be killed:**
- If Student-t GARCH performs well, it reduces the contrast with conformal methods — weakening the central narrative
- Gaussian GARCH IS the historical industry standard (Basel I internal model, RiskMetrics) — testing it is methodologically honest, not strawmanning
- Including Student-t GARCH as a supplementary variant dilutes the clean comparison matrix
- The argument "we test the method practitioners actually used" is both correct and compelling

**Verdict: GARCH(1,1)-Gaussian only. Mention Student-t as a known extension in the limitations slide.**

---

## KILL 7 — Option Pricing Application

**Why it's tempting:** Options are central to finance and conformal prediction could bound option prices with coverage guarantees — a genuinely interesting application.

**Why it must be killed:**
- Requires options data (not available via yfinance for free in usable form)
- Black-Scholes pricing model and Greeks require a separate modeling layer entirely
- Connecting equity return intervals to option bounds requires implied volatility translation — a non-trivial derivation
- We would need to explain both conformal prediction AND options theory — doubling the conceptual load
- The VaR application is cleaner, more defensible, and already well-connected to the existing coverage literature

**Verdict: No derivatives. VaR only.**

---

## KILL 8 — Rolling Window Re-Calibration for Static CP

**Why it's tempting:** Re-calibrating static conformal every 90 days might give it "adaptive" properties without needing ACI, which could undermine the novelty claim.

**Why it must be killed:**
- We WANT static CP to fail during stress — that's the point. Improving it manually blurs the comparison.
- A rolling re-calibration is a manual approximation of what ACI does automatically and provably. ACI is strictly theoretically superior.
- Showing that hand-tuned rolling calibration beats naive static CP BUT underperforms ACI is an interesting finding — but requires implementing a third conformal variant that has no clean theoretical backing
- The comparison is: "static (provably wrong assumption) vs. adaptive (theoretically robust)" — keep it clean

**Verdict: Static CP uses the one-time 2015–2017 calibration set. No re-calibration. If needed, mention "rolling recalibration" as a known middle ground in the limitations slide.**

---

## KILL 9 — Sentiment / NLP / Alternative Data Features

**Why it's tempting:** Adding news sentiment or social media signals would show cross-disciplinary sophistication.

**Why it must be killed:**
- Requires NLP pipeline (data download, preprocessing, embedding, alignment) — a completely separate project
- Sentiment data for 2004–2024 from a free source is essentially absent or unreliable
- Feature importance for sentiment in daily equity return prediction is weak and noisy
- Adds a dimension where graders may be skeptical and challenge data quality
- The 7-feature model is already theoretically justified — adding sentiment weakens the argument for feature minimalism

**Verdict: No alternative data. VIX is our contextual market signal. Enough.**

---

## KILL 10 — Expected Shortfall as Primary Risk Metric

**Why it's tempting:** FRTB mandates ES. Using ES would signal awareness of the current regulatory framework.

**Why it must be killed:**
- ES (Conditional VaR) is not directly elicitable — its formal backtesting requires specialized tests (Acerbi, McNeil) that add significant evaluation complexity
- VaR has the Kupiec test — a direct, well-known, one-sentence-explainable formal backtest
- The coverage guarantee of conformal prediction maps directly to VaR (lower bound coverage = VaR estimate) but does NOT directly produce an ES estimate
- Including ES would require computing Expected Shortfall from the conformal prediction interval — beyond the scope of the current framework
- We can mention ES as a natural extension in the conclusions slide without implementing it

**Verdict: VaR only. Mention ES once in limitations. Do not compute, report, or defend ES results.**

---

## KILL 11 — Intraday Data

**Why it's tempting:** Intraday data would give higher statistical power and allow testing conformal prediction at finer time scales.

**Why it must be killed:**
- Intraday data requires microstructure correction (bid-ask bounce, price impact)
- 1-minute SPY data for 20 years = ~30 million rows — memory and compute overhead
- Intraday conformal prediction has a different theoretical setup (what does "coverage" mean at 1-minute frequency?)
- The regulatory VaR benchmark is explicitly daily — intraday results are disconnected from the finance application
- yfinance intraday data is limited to the last ~60 days for free — no historical crisis data accessible

**Verdict: Daily frequency only. Mention intraday as a future extension, nothing more.**

---

## Summary Kill List (For Quick Reference)

| # | Killed Item | One-Line Reason |
|---|---|---|
| 1 | LSTM / Transformers | Base learner complexity is irrelevant to coverage; destroys reproducibility |
| 2 | Second asset full pipeline | Depth > breadth; decision layer already satisfies portfolio criterion |
| 3 | Multi-horizon forecasting | Breaks coverage derivation; √10 scaling is a one-sentence mention |
| 4 | Streamlit dashboard | Engineering ego; no research value; demo risk |
| 5 | Bayesian MCMC | GARCH is the practical proxy; full Bayes requires new library + compute |
| 6 | Student-t GARCH as primary | Would reduce contrast with conformal; Gaussian GARCH IS the historical standard |
| 7 | Option pricing application | Requires options data pipeline + B-S theory; separate project |
| 8 | Rolling re-calibration of static CP | Manually weakens the comparison that motivates ACI |
| 9 | Sentiment / NLP features | Separate project; weak signal; data quality risk |
| 10 | Expected Shortfall primary metric | Not directly elicitable; no clean conformal connection; Kupiec is cleaner |
| 11 | Intraday data | Microstructure complexity; no crisis history available free |
