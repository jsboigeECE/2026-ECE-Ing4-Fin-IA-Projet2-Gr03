# 19 — Oral Defense Strategy

---

## Mindset for the Defense

The defense is not a viva examination of weaknesses — it is an opportunity to demonstrate command of the material. A hostile question is not an attack; it is an invitation to show that you thought harder about the problem than a casual observer.

**Rules:**
1. Never apologize for methodological choices — explain them.
2. Never say "we didn't have time to..." — say "we deliberately excluded X because Y."
3. If you don't know the answer, say: "That's an interesting extension — our validation data would need to be re-examined to answer that precisely."
4. Lead with numbers when challenged on claims. Numbers end arguments.

---

## Category 1 — Methodology Defense

**Q: Why didn't you use a more powerful base learner like XGBoost or LSTM?**

> "The base learner is irrelevant to the conformal coverage guarantee — coverage holds for any base learner under exchangeability. We chose Ridge Regression for two reasons: interpretability of the residuals, and speed of re-training in the walk-forward loop. A more complex model would have produced similar conformal coverage behavior but with opaque residuals and a fragile training pipeline. The contribution is in the conformal layer, not the forecasting layer."

**Q: Your R² is essentially zero — why bother with Ridge at all?**

> "Low R² on daily equity returns is the expected result under the Efficient Market Hypothesis. We are not building an alpha model — we are building a risk measurement tool. What matters is that Ridge produces mean-zero, approximately heteroscedastic residuals that serve as stable conformal scores. An R² of 1.5% is meaningfully better than a constant-mean model for residual stability — and the Diebold-Mariano test confirms this."

**Q: The exchangeability assumption is clearly violated for financial time series. Doesn't that invalidate conformal prediction?**

> "You're correct that perfect exchangeability fails in financial time series due to volatility clustering. This violation is the central motivation for our study. Static split conformal relies on exchangeability and we show it fails empirically under stress — that's a finding, not a failure of the methodology. ACI and EnbPI are specifically designed to work under this violation, with ACI providing a long-run time-average coverage guarantee without any exchangeability requirement. The project empirically tests whether this theoretical advantage materializes."

**Q: ACI's γ parameter is a hyperparameter — didn't you tune it on the test set?**

> "No. γ was selected on the validation set (2018–2019), which is strictly separated from the test set (2020–2024). The validation set covers the December 2018 drawdown — a relevant stress signal — but predates all COVID and 2022 data. γ is frozen before the test set is opened. This is explicit in the experimental protocol."

**Q: Why only marginal coverage? Conditional coverage is more important in finance.**

> "We report both marginal coverage (unconditional) and conditional coverage via the Christoffersen test, which specifically tests whether exceptions are independent (a condition for valid conditional coverage). We also report regime-conditional coverage separately for calm, stress, and crisis periods — which is a form of conditional coverage analysis. Full conditional coverage would require much larger samples per conditioning set and is left as a formal extension."

---

## Category 2 — Finance Defense

**Q: Banks use Expected Shortfall (ES), not VaR, under FRTB. Why focus on VaR?**

> "You're correct that FRTB uses ES. However, VaR remains the standard regulatory measure under Basel III's standardized approach and is the foundational quantity that ES is built from. More importantly, VaR has a directly testable coverage property — the Kupiec test — which makes it the ideal object for evaluating prediction interval quality. ES is useful but has eliticability issues that make fair comparison across methods harder. Our VaR focus is methodologically cleaner and directly connects to the coverage guarantee literature."

**Q: Your Sharpe ratio improvement is modest. Is this economically significant?**

> "The decision layer is not designed to generate alpha — it is designed to demonstrate risk control. The relevant metric is maximum drawdown reduction and Calmar ratio, reported in our results table. A risk manager is willing to sacrifice some return to avoid a large drawdown, because that level of loss triggers redemptions, margin calls, and regulatory review — consequences that cannot be recovered from easily. The Calmar ratio captures this tradeoff more precisely than Sharpe alone."

**Q: Why SPY? Conformal prediction would be more interesting on a less liquid, less efficient market.**

> "We chose SPY precisely because it is the most challenging environment for any interval method — it is the most studied, most efficient, most liquid market in the world. If we had used a small-cap ETF or an emerging market index, results would be harder to criticize AND harder to trust. SPY results that show ACI advantage set a high bar — the effect must be real, not driven by data quality issues or thin markets. That said, the next natural extension is to apply the same protocol to a less liquid instrument — which we note as a future direction."

**Q: Your position sizing rule assumes you can trade at the daily close. Is that realistic?**

> "The position size is computed using interval width from closing data on day t and applied to closing returns on day t+1 — which corresponds to a market-on-open order on the next trading day. This is conservative: it accounts for the information available before the next session opens. In practice, a fund would use tomorrow's open, not close, but the qualitative result is the same. We acknowledge this execution assumption is simplified."

---

## Category 3 — Why Conformal vs Alternatives?

**Q: Why not just use a Bayesian approach with full uncertainty quantification?**

> "A full Bayesian approach requires specifying a prior distribution over the DGP — which either replicates a parametric assumption (defeating the purpose) or requires MCMC sampling at each walk-forward step (computationally expensive and hyperparameter-sensitive). GARCH is a classical frequentist model estimated by maximum likelihood — it is not Bayesian in any technical sense, but it represents the industry-standard parametric benchmark and we include it as such. Conformal prediction makes a strictly weaker assumption — exchangeability — with a finite-sample, prior-free coverage guarantee, without requiring any distributional prior. On the same 20-year time series, conformal's lower assumption cost is the correct tradeoff."

**Q: Why is ACI better than just using a longer calibration window for static conformal?**

> "A longer calibration window dilutes recent regime-specific information. If we use 2004–2019 for calibration (including the GFC), the quantile includes stressed-period residuals that makes static CP always wide — sacrificing efficiency. ACI adapts online at the exact rate needed, tightening in calm regimes and widening in stress, without requiring manual window choice. The γ parameter controls the adaptation speed — a far simpler hyperparameter than 'optimal calibration window length.'"

**Q: Has this approach been validated on other assets? Isn't this just data-snooping on SPY?**

> "Single-asset studies are common in methodological finance papers — Kupiec's original 1995 VaR backtest paper used a single bond portfolio. Our claim is about the method, not the asset. The hypothesis we test — that adaptive conformal should maintain better coverage during regime shifts — is coherently predicted by theory a priori. We chose SPY precisely because it is the hardest environment to show any effect; a positive result here is more credible than on an obscure illiquid instrument. Generalization to other assets is the natural next step."

---

## Category 4 — Limitations and Failure Modes

**Q: What would cause your main result to be spurious?**

> "Three scenarios: (1) If the COVID crash were shorter than the ACI adaptation window, ACI might not have time to correct before the crash ends. (2) If γ were accidentally tuned on test data — which we explicitly prevented. (3) If the choice of 60-day rolling window for coverage reporting were cherry-picked to show favorable results — we can check robustness with 30-day and 90-day windows on request."

**Q: What are the honest limitations of this work?**

> "Three: First, single asset. We cannot claim the result generalizes without additional experiments. Second, ACI's long-run guarantee is on the time average — not on any specific window. During a rapid 3-day crash, ACI's per-step coverage may still fail, even if the running average recovers. Third, transaction costs in the decision layer are ignored. At daily rebalancing, costs would reduce the strategy's practical advantage, though not eliminate the risk control benefit."

---

## How to Sound Elite and Credible

1. **Cite specific papers by name:** "Gibbs and Candès showed in NeurIPS 2021 that..." — not "a recent paper showed..."
2. **Use regulatory language naturally:** "Kupiec test," "Basel Traffic Light," "exception clustering," "FRTB" — these signal domain fluency
3. **Differentiate coverage types:** Marginal vs conditional vs rolling vs regime-conditional. Students who conflate these are identified immediately.
4. **Acknowledge the limitations first** when asked about them — don't wait to be pushed. Proactive acknowledgment is more credible than defensive explanation.
5. **Use exact numbers from your results:** State actual empirical coverage values from the results table — e.g., "Coverage dropped from X% to Y% during the COVID period" — rather than qualitative claims. Prepare these numbers from `results/metrics/coverage_table.csv` before the defense. Do not memorize fabricated numbers.
