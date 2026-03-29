# Brutal Final Verdict

---

## 1. Current Design Strength

This design is **genuinely strong**. It is not flattery — it earns the description.

The project has:
- A **precise, singularly focused research question** with a falsifiable hypothesis and pre-specified rejection criteria
- A **strict chronological experimental protocol** that eliminates the leakage risks that plague 90% of student-level applied ML projects
- **Seven carefully chosen methods** in a hierarchy that tells a clear story: parametric fails → static conformal holds on average → adaptive conformal holds under stress
- A **finance application** (VaR / Kupiec / decision layer) that connects method performance to real regulatory and portfolio consequences
- A **visual strategy** anchored on one hero figure (rolling coverage plot) that makes the central argument without requiring explanation
- An **oral defense preparation** that anticipates every hostile question with precise, non-apologetic answers
- A **kill list** that actively prevents the six most common forms of student project self-sabotage
- An **execution roadmap** that sequences implementation to minimize risk and maximize early deliverables

The project as designed is significantly better than the median Ing4 finance/AI project. It would be among the best submissions if executed correctly.

---

## 2. Biggest Remaining Weakness

**The main empirical result is uncertain before code is run.**

The entire project narrative depends on ACI showing meaningfully better empirical coverage than static conformal during the COVID crash and 2022 bear market. If that gap is small (< 3 percentage points), the central thesis weakens substantially.

SPY daily returns are among the most efficient, most liquid financial series in the world. They may be close enough to exchangeable at the calibration timescale (2015–2017) that static conformal does not fail dramatically in 2020. This is a real possibility, not paranoia.

The project has fallback narratives (width dynamics, Winkler score, GARCH parametric failure, decision layer), but none of them are as dramatic or as directly tied to the thesis as the coverage degradation result.

**This risk cannot be designed away — it can only be managed through preparation.**

---

## 3. What Would Stop This Project from Reaching Top-Grade Territory

Three failure modes, in decreasing order of likelihood:

**Failure Mode A — Results are run on the test set before the protocol is frozen.**  
Even one decision made after seeing test results — choosing γ that maximizes ACI coverage on the test set, trimming the stress period to improve the heatmap coloring — invalidates the scientific integrity of the entire study. This is the most dangerous failure. It is undetectable post-hoc and it is career-defining to avoid.

**Failure Mode B — The finance framing is dropped during implementation.**  
The technical team focuses on making the conformal methods run correctly and forgets to implement the Kupiec test, the decision layer, and the regime-conditional tables. The final submission has correct coverage results but no connection to VaR, no regulatory interpretation, and no finance decision application. The project becomes an ML comparison that earns a solid but not top grade.

**Failure Mode C — The presentation sounds like a tutorial.**  
Slides 1–2 spend four minutes explaining what conformal prediction is. The evaluators spend the rest of the talk mentally categorizing this as "competent reproduction of the literature" rather than "applied research." The rolling coverage plot is shown for 30 seconds instead of 90 seconds. The Kupiec comparison disappears into a table appendix. The oral defense is technically correct but not memorable.

**All three of these failures are avoidable, preventable, and entirely within the team's control.**

---

## 4. The Single Highest-ROI Upgrade

**Implement the Christoffersen conditional coverage test and present it with explicit Basel Traffic Light framing.**

Specifically: create a table that shows, for each method during the COVID sub-period, the number of VaR exceptions, whether they are clustered, the Christoffersen CC test p-value, and the Basel Traffic Light color (green/yellow/red).

This single table:
- Demonstrates knowledge of the regulatory VaR backtesting framework beyond Kupiec
- Shows that GARCH fails BOTH the unconditional test (too many exceptions) AND the independence test (they are clustered) — a devastating double failure
- Shows that ACI passes the unconditional test and performs better on independence — clustered exceptions disappear faster because α_t updates immediately
- Uses language that every finance professor recognizes as regulatory-grade rigor

Implementation cost: ~15 lines of Python for the Christoffersen test. Grading impact: moves the project from "strong methodology section" to "finance-grade empirical validation." This is the highest grade-per-hour upgrade available at this stage.

---

## 5. The Single Most Dangerous Mistake to Avoid

**Running any experiment on the test set before all design decisions are frozen — then using the test set results to adjust any methodological choice, however innocently.**

This is the most dangerous mistake because:
1. It is invisible to evaluators — they cannot detect it from the code or results
2. It corrupts the scientific validity of every result in the study
3. It is especially likely under time pressure ("let's just check if this works before committing to it")
4. The temptation is highest for γ selection (ACI) and stress period boundary selection

**Prevention protocol:**  
Before running anything on the test set, every team member must answer: "Have all hyperparameters been fixed? Are all split boundaries frozen? Has the list of sub-periods and their exact dates been written down?" If the answer to any of these is "not sure," stop and write it down first.

The `.gitignore` for `data/raw/spy_daily.csv` may help: if the test set data is not even loaded until Phase 4 of implementation, accidental test-set peeking is prevented by construction.

---

## 6. Overall Design Assessment

**This design is in top-grade territory — conditionally.**

The condition is execution quality. The architecture is elite. The scope is appropriately narrow. The theoretical foundation is solid. The finance framing is concrete and defensible. The visualization strategy is focused and impactful. The oral defense preparation is thorough.

But architecture gives grades only after it is implemented correctly. A poorly executed version of this design could still land at a mediocre grade if:
- The implementation has leakage
- The finance framing is abandoned during coding
- The presentation does not deliver the rolling coverage figure with sufficient impact
- The defense team cannot answer basic questions about ACI's convergence guarantee or Kupiec test interpretation

Executed to 90% of the design quality: **strong project, 16–18 territory**  
Executed to 100% of the design quality with the Christoffersen upgrade: **top-grade territory, 18–20 realistic**  
Executed with leakage or missing finance layer: **12–15 regardless of method complexity**

---

## 7. Exact Next Action for Roo Code Mode

**Switch to Code mode. Begin Phase 0 of doc 21 immediately.**

The first task is exact:
1. Create the full directory structure from doc 16
2. Create `requirements.txt` with all pinned dependencies
3. Create `src/__init__.py` and all sub-package `__init__.py` files
4. Create `scripts/download_data.py` that downloads SPY and VIX via yfinance and saves to `data/raw/`
5. Run `download_data.py` and verify the CSVs contain clean data (no missing dates, reasonable price ranges)

Do NOT start implementing conformal methods before the data pipeline is verified and the feature engineering leakage test has passed.

The gate before moving to Phase 1: feature matrix shows zero NaN values for all rows in the training set, and a manual spot-check confirms that the feature value on any given test date uses only data from strictly before that date.

**Start there. The architecture is done. Build it.**

---

*End of architecture design. Total: 23 specification documents + this verdict.*  
*Group 03 — Lewis, Orel, Thomas Nassar — ECE Paris Ing4 Finance / IA Probabiliste*
