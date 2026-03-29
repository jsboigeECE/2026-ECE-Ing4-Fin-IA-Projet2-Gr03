# 23 — Top Grade Checklist

---

## Purpose

This is the final binary gate before submission. Every item marked **REQUIRED** must be true. Items marked **NICE-TO-HAVE** add marginal value. Items marked **DANGER** instantly downgrade the project.

This document is not aspirational — it is a quality contract.

---

## Section 1 — REQUIRED: Scientific Rigor

- [ ] **Chronological splits are strictly enforced.** No data from the test set was used for training, calibration, or hyperparameter selection.
- [ ] **All features use lagged data only.** No look-ahead in any rolling window computation. `.shift(1)` verified on feature engineering code.
- [ ] **All conformal intervals are computed on unseen test data only.** Calibration set was not included in test evaluation.
- [ ] **ACI γ was selected on the validation set, not the test set.**
- [ ] **EnbPI β was selected on the validation set, not the test set.**
- [ ] **Ridge λ was selected via time-series CV within the training set only.**
- [ ] **GARCH is re-fitted monthly using only data available before each prediction date.**
- [ ] **All metrics are computed on the test set [2020-01-02, 2024-12-31] only.** No test set results are in the training, calibration, or validation sections.
- [ ] **Random seed is set globally and documented.**
- [ ] **Results are reproducible from saved CSVs** — running `generate_figures.py` does not re-run the experiment.

---

## Section 2 — REQUIRED: Coverage and Statistical Results

- [ ] **Empirical coverage rates are reported for all mandatory methods** (Split CP, CQR, ACI, GARCH, HistSim, Linear QR) at both primary and secondary two-sided coverage levels (80%, 90%, 95%).
- [ ] **Kupiec POF test p-values are reported** for all methods at all confidence levels.
- [ ] **Christoffersen conditional coverage test is reported** for at least the primary methods.
- [ ] **Rolling 60-day coverage plot (Fig01) is generated** and shows clear visual differentiation between adaptive and static methods during at least one stress period.
- [ ] **Regime-conditional coverage table (Fig03 heatmap) is generated** with calm / elevated / stress / crisis rows.
- [ ] **ACI α_t time series is plotted** and shows plausible dynamic behavior (increases during crisis, stabilizes during recovery).
- [ ] **Coverage results are presented at the regime level**, not only as full-period aggregates.

---

## Section 3 — REQUIRED: Finance Application

- [ ] **A concrete VaR framing is established** and maintained throughout — the lower bound of the 90% two-sided interval serves as the 95% VaR estimate (α/2 = 5% exceedance), with this correspondence explicitly stated in the results.
- [ ] **Kupiec test is explicitly connected to Basel Traffic Light** — green/yellow/red zone interpretation is present.
- [ ] **A decision layer (position sizing rule) is implemented and evaluated** with Sharpe ratio and MaxDD metrics.
- [ ] **At least 4 policies are compared in the decision layer**: Buy-and-Hold, Static CP sizing, ACI sizing, and at least one parametric baseline (GARCH or VIX threshold).
- [ ] **The COVID crash (Feb–Mar 2020) is explicitly analyzed** as the primary stress test period.
- [ ] **The 2022 bear market is mentioned** as a secondary stress period.
- [ ] **The finance narrative is present in the README**, the slides, and the defense prep — this is NOT just an ML comparison.

---

## Section 4 — REQUIRED: Repository Quality

- [ ] **Full directory structure exists** per doc 16 (src/, notebooks/, scripts/, results/, docs/, data/, slides/).
- [ ] **All notebooks execute top-to-bottom without errors** (Restart Kernel & Run All tested before submission).
- [ ] **`requirements.txt` is pinned** with exact versions and tested in a clean environment.
- [ ] **`scripts/run_experiment.py` executes without errors** and produces all results CSVs.
- [ ] **`scripts/generate_figures.py` executes without errors** and produces all figures.
- [ ] **All figures are committed to the repo** as PNG (and ideally SVG).
- [ ] **README.md is complete** per doc 17 blueprint (abstract, key results table, figure embedded, quickstart, references).
- [ ] **All 23 docs/ files are committed** and well-formatted.
- [ ] **No temporary files, `.ipynb_checkpoints/`, or `__pycache__/`** are committed (covered by `.gitignore`).

---

## Section 5 — REQUIRED: Presentation Quality

- [ ] **Presentation has maximum 9 slides** (7 content + title + backup).
- [ ] **Slide 1 opens with the problem, not with method explanation.**
- [ ] **Fig01 (rolling coverage plot) has a full dedicated slide** with ≥ 90 seconds of presentation time.
- [ ] **Kupiec test results appear** in the presentation with Basel Traffic Light framing.
- [ ] **Decision layer equity curve (Fig06) appears** in the presentation.
- [ ] **At least one limitation is stated proactively** during the presentation.
- [ ] **Presentation references real papers by name**: Gibbs & Candès (2021), Xu & Xie (2021), Romano et al. (2019), Kupiec (1995).
- [ ] **Presentation contains no code snippets.**
- [ ] **All figures in the presentation are high-resolution and legible when projected.**

---

## Section 6 — REQUIRED: Oral Defense Readiness

- [ ] **All team members can explain ACI's α_t update formula** and what γ controls.
- [ ] **All team members can define the Kupiec test** and interpret a p-value result.
- [ ] **All team members can explain why Ridge was chosen** over LSTM/XGBoost.
- [ ] **All team members can state one honest limitation** of the project.
- [ ] **All team members know the exact coverage numbers** from the crisis sub-period table.
- [ ] **All team members can explain the decision layer rule** (position sizing formula and rationale) in under 60 seconds.
- [ ] **The defense strategy document (doc 19) has been read and rehearsed by all team members.**

---

## Section 7 — NICE-TO-HAVE (Add if time permits)

- [ ] **EnbPI implemented and results reported** (STRETCH — only after Split CP, CQR, ACI fully gated)
- [ ] Winkler score table computed and reported
- [ ] Exception calendar heatmap (Fig08) generated
- [ ] Width–coverage scatter / efficiency frontier (Fig07) generated
- [ ] ACI γ sensitivity analysis reported on validation set (shows robustness of parameter selection)
- [ ] Diebold-Mariano test explicitly reported for base learner comparison
- [ ] Short section in README or docs comparing conformal to parametric approaches — note GARCH is frequentist MLE, not Bayesian (one paragraph)
- [ ] `data/raw/spy_daily.csv` committed directly (removes yfinance dependency for reproducers)
- [ ] Kupiec test reported at both 5% AND 2.5% exceedance rates for completeness

---

## Section 8 — DANGER: What Instantly Makes This Project Feel Ordinary

These mistakes, if present, signal undergraduate-level work regardless of the scientific content:

- ❌ **Presenting only average coverage without regime-conditional breakdown** — this is the single most common failure in student-level uncertainty quantification work
- ❌ **Calling this "a machine learning project"** instead of "an uncertainty quantification study for financial risk management"
- ❌ **Not implementing the Kupiec test** — using coverage rates alone without formal backtesting is insufficient for "finance" framing
- ❌ **Having a broken notebook or unexecuted cells** at submission
- ❌ **Presenting the conformal methods without framing why the exchangeability assumption matters** — talking about conformal without addressing non-stationarity = missing the finance-specific contribution
- ❌ **Showing results only on the full test period** without isolating crisis sub-periods
- ❌ **Opening the presentation with a definition of conformal prediction** rather than the finance problem
- ❌ **Not including the decision layer** — without it, this is a methodology comparison, not applied finance research
- ❌ **Repository with a single `analysis.ipynb` file** — monolithic notebooks signal no software engineering discipline
- ❌ **README that lists libraries used but has no scientific narrative**

---

## Section 9 — What Would Make This Project Memorable

These are the elements that push from "strong project" to "projects they remember at grading time":

- ✅ **The rolling coverage plot shows a dramatic visible gap** between adaptive and static methods during COVID — a figure that needs no explanation to make its point
- ✅ **The Kupiec p-value table has at least one method clearly failing** (GARCH during COVID, p < 0.01) contrasted with ACI passing (p > 0.10) — quantified regulatory consequences
- ✅ **The ACI α_t plot shows a clear adaptive response to the COVID crash** — a tangible demonstration of the theoretical mechanism functioning as designed
- ✅ **The decision layer shows measurable drawdown reduction** during the most severe crisis in the test period
- ✅ **The limitations are stated proactively and precisely** — shows scientific maturity, not defensiveness
- ✅ **The repository structure looks like a quant research package**, not a homework submission
- ✅ **The README opens with the key figure** and a crisp one-paragraph abstract — a reviewer who spends 60 seconds on the README already understands the contribution
- ✅ **Real paper citations used fluently** during the defense — "as Gibbs and Candès proved in NeurIPS 2021..." is instantly credible
