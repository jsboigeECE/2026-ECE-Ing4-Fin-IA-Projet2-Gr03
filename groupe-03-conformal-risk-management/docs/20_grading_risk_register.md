# 20 — Grading Risk Register

---

## Framework

Each risk is assessed on:
- **Probability:** Low / Medium / High — likelihood of occurring given current plan
- **Severity:** Low / Medium / High / Critical — grade impact if it occurs
- **Mitigation:** Concrete pre-emptive action

Risks are ranked by `Severity × Probability`.

---

## Risk Category 1 — Methodological Risks

### R1.1 — ACI result is not statistically significant
**Probability:** Medium | **Severity:** High

The coverage gap between ACI and static conformal during COVID may be < 3 percentage points, making the main thesis underpowered.

**Mitigation:**
- Pre-specify the threshold (3 pp) in the experimental protocol before running
- If the gap is < 3 pp: pivot to width dynamics and Kupiec test results as primary findings
- Have a prepared narrative: "The exchangeability assumption is more robust than expected on SPY daily data — which is itself a finding that de-risks the use of static conformal for liquid equity indices"
- Include Winkler score as backup metric — it may show ACI superiority even if raw coverage differences are small

---

### R1.2 — All methods under-cover during COVID (general failure)
**Probability:** Medium | **Severity:** Medium

If ALL methods — including ACI — show severe undercoverage during the COVID crash, the adaptive advantage disappears into a general "everything failed" result.

**Mitigation:**
- This actually strengthens the "crisis is unforeseeable" narrative
- Reframe: "No interval method fully survived the COVID shock — but ACI recovered coverage fastest, which is the relevant risk management property"
- Show DAYS-TO-RECOVERY metric: how many trading days until rolling coverage returns to ≥ nominal level after the shock

---

### R1.3 — EnbPI implementation is buggy or produces degenerate intervals
**Probability:** Medium | **Severity:** Medium

EnbPI is the most complex method to implement correctly. Bootstrap ensemble + rolling score update has several potential edge cases.

**Mitigation:**
- Implement EnbPI LAST (after Split CP and ACI are validated)
- Unit test: on IID Gaussian data, EnbPI coverage should match the nominal level closely
- If EnbPI results look anomalous: exclude it from presentation, keep in supplementary
- Do NOT delay ACI for debugging EnbPI — ACI is higher grading priority

---

### R1.4 — GARCH fails to re-fit during walk-forward (numeric instability)
**Probability:** Low | **Severity:** Medium

GARCH optimization may fail to converge on some expanding window fits, especially early in the series.

**Mitigation:**
- Use `arch` library with `show_warnings=False` and `options={'maxiter': 1000}`
- Set fallback: if GARCH fails to converge, use previous period's σ̂ estimate
- Log all convergence failures — document in supplementary if any occur

---

### R1.5 — Data leakage introduced during feature engineering
**Probability:** Low | **Severity:** Critical

Look-ahead in feature computation would invalidate all results.

**Mitigation:**
- Code review specifically focused on `.shift()` placement in all rolling computations
- Unit test: features at any test date must use only prices strictly before that date
- Separate data loading notebook (notebook 00) that outputs the features CSV — audit this CSV manually on 3 random dates

---

## Risk Category 2 — Storytelling Risks

### R2.1 — Presentation sounds like a conformal prediction tutorial
**Probability:** Medium | **Severity:** High

If slides 1–2 spend more than 2 minutes on "what is conformal prediction," the project feels like a survey rather than research.

**Mitigation:**
- Follow the storyline in doc 18 strictly
- Lead with the PROBLEM (VaR models fail in crises), not with the METHOD
- Time the presentation in rehearsal — max 90 seconds on slide 2 (methods)

---

### R2.2 — Results presented without proper finance context
**Probability:** Medium | **Severity:** High

Showing coverage tables without connecting to Basel requirements, Kupiec tests, or VaR decision rules makes the project look like an ML comparison, not applied finance.

**Mitigation:**
- Every results slide must containat least one finance-domain term: "regulatory backtest," "VaR exception," "exception clustering," "drawdown"
- Kupiec test p-value MUST appear in the presentation — it is the finance credential
- Decision layer equity curves MUST appear — without them, the project has no finance punchline

---

### R2.3 — Wide intervals dismissed as "not useful" by evaluator
**Probability:** Low | **Severity:** Medium

If ACI intervals are significantly wider than GARCH intervals, an evaluator may challenge whether wider = less useful in practice.

**Mitigation:**
- Pre-empt: "A wider interval that covers is infinitely more useful than a narrow interval that doesn't. A risk manager who uses GARCH 95% VaR during COVID had 20% exceptions — which is an operational, regulatory, and capital disaster."
- Show the coverage-width efficiency frontier (Fig07) — ACI sits on the Pareto frontier
- Prepare the Winkler score slide as backup: it proves wider-but-covers beats narrow-but-misses under any proper scoring rule

---

## Risk Category 3 — Repository Quality Risks

### R3.1 — Notebooks have unexecuted cells or errors at submission
**Probability:** Medium | **Severity:** High

A grader who opens a notebook with unrun cells or visible error messages immediately loses confidence.

**Mitigation:**
- Run all notebooks top-to-bottom with `Restart Kernel & Run All` before committing
- Notebooks must complete without errors
- Use `try/except` around GARCH convergence failures and log warnings, never crash

---

### R3.2 — `requirements.txt` is incomplete or has version conflicts
**Probability:** Medium | **Severity:** Medium

If the grader cannot reproduce the environment, all claims of reproducibility are void.

**Mitigation:**
- Generate `requirements.txt` from a clean virtual environment: `pip freeze > requirements.txt`
- Test on a second clean environment before submission
- Pin all critical packages: `numpy==x.y.z`, `scikit-learn==x.y.z`, `arch==x.y.z`, `mapie==x.y.z`

---

### R3.3 — Data download fails (yfinance API changes)
**Probability:** Low | **Severity:** Medium

yfinance API is unofficial and could break.

**Mitigation:**
- Commit `data/raw/spy_daily.csv` and `data/raw/vix_daily.csv` directly to the repository
- Add download script as backup only
- Document the download date in the data files' header

---

## Risk Category 4 — Demo / Execution Risks

### R4.1 — Live demo crashes during presentation
**Probability:** Low | **Severity:** High

Running code live during a presentation almost always has a failure mode.

**Mitigation:**
- **Do not run live code during the presentation**
- All figures are pre-generated PNGs embedded in slides
- Figures are reproducible offline from saved results CSVs — demonstrate this in README, not during defense

---

### R4.2 — Evaluator asks for a result not pre-computed
**Probability:** Medium | **Severity:** Low

"What happens if you use a 30-day rolling window for coverage?" — a response "we haven't computed that" is weak.

**Mitigation:**
- Pre-compute sensitivity analyses on validation set: rolling window = 30, 60, 90 days
- Pre-compute sensitivity for γ ∈ {0.005, 0.01, 0.02, 0.05} on validation
- Keep these results in the supplementary backup slide

---

## Risk Category 5 — Time Management Risks

### R5.1 — EnbPI implementation consumes too much time
**Probability:** High | **Severity:** Low

EnbPI is the most complex method but not the highest marginal grading value.

**Mitigation:**
- Time-box EnbPI to maximum 20% of total implementation effort
- If over budget: drop EnbPI from primary results, keep in supplementary (mention in presentation as "partially implemented")
- ACI + Static CP + CQR already cover the core narrative

---

### R5.2 — Decision layer results are weak and undermine the finance story
**Probability:** Medium | **Severity:** Medium

If interval-width position sizing does NOT improve Sharpe or MaxDD relative to buy-and-hold, the finance application punchline disappears.

**Mitigation:**
- Smooth width signal with 5-day rolling mean before computing position size — reduces noise
- If Sharpe improvement is absent but MaxDD improvement is present, lead with MaxDD (the more risk-relevant metric)
- If BOTH are absent: report honestly as "the uncertainty signal does not translate to portfolio improvement at daily frequency — confirming that daily CP widths carry limited exploitable information at this timescale, which is an empirical finding"

---

## Overall Risk Summary Table

| Risk | Probability | Severity | Priority | Mitigation Status |
|---|---|---|---|---|
| R1.5 — Data leakage | Low | Critical | 1 | Prevent via test in feature engineering |
| R2.2 — No finance context | Medium | High | 2 | Enforce in slide review |
| R1.1 — Weak ACI result | Medium | High | 3 | Pre-specify threshold; prepare pivot |
| R3.1 — Broken notebooks | Medium | High | 4 | Run all before commit |
| R2.1 — Tutorial presentation | Medium | High | 5 | Follow doc 18 strictly |
| R5.1 — EnbPI time overrun | High | Low | 6 | Time-box; drop if needed |
| R1.3 — EnbPI bugs | Medium | Medium | 7 | Implement last; have fallback |
| R5.2 — Weak decision layer | Medium | Medium | 8 | Smooth width; lead with MaxDD |
