# 17 — README Blueprint

---

## Design Principle

The README is the project's handshake. A reviewer must understand the entire contribution within 90 seconds. The README is NOT documentation — it is a persuasion document that earns the reader's attention and navigates them to the key result.

**Structure:** Thesis → Key Finding → Reproduce → Navigate  
**Tone:** Crisp, technical, self-assured. No apologetics, no "we tried our best."

---

## README Structure (Section-by-Section)

---

### Section 1: Header

```markdown
# Adaptive Conformal Prediction for Financial Risk Management
### Distribution-Free VaR Estimation Under Regime Shifts — SPY Daily Returns, 2020–2024

Group 03 | ECE Paris — Ing4 Finance / IA Probabiliste
Lewis · Orel · Thomas Nassar
```

Followed immediately by: one key figure embedded (Fig01 rolling coverage plot).  
No preamble. The figure IS the opening statement.

---

### Section 2: One-Paragraph Abstract

Content:
- What we study (conformal prediction for VaR estimation on SPY)
- Key finding (ACI + EnbPI maintain better coverage during COVID crash and 2022 bear market than static conformal and GARCH)
- Finance application (interval-width position sizing reduces maximum drawdown vs buy-and-hold)
- Method summary (Split CP, CQR, EnbPI, ACI vs GARCH, HistSim, Linear QR — walk-forward evaluation, 2020–2024 test)

Length: 5 sentences maximum. No fluff.

---

### Section 3: Key Results Table

Embed the actual coverage table from `results/metrics/coverage_table.csv` after experiments are run. The table structure must be:

```markdown
| Method         | Full Coverage | Crisis Coverage | Avg Width | Kupiec p (crisis) |
|----------------|:-------------:|:---------------:|:---------:|:-----------------:|
| HistSim        | XX.X%         | XX.X%           | X.XX%     | X.XXX             |
| GARCH-Gaussian | XX.X%         | XX.X%           | X.XX%     | X.XXX             |
| Linear QR      | XX.X%         | XX.X%           | X.XX%     | X.XXX             |
| Split CP       | XX.X%         | XX.X%           | X.XX%     | X.XXX             |
| CQR            | XX.X%         | XX.X%           | X.XX%     | X.XXX             |
| **ACI**        | **XX.X%**     | **XX.X%**       | X.XX%     | **X.XXX**         |

Nominal: 90% two-sided interval (≡ 95% VaR lower bound). Crisis = COVID crash (Feb 19 – Apr 30, 2020).
Kupiec test at 5% exceedance rate. p < 0.05 = model rejected at 95% confidence.
```

**Do not insert fabricated numbers in the README under any circumstances.** Fill this table only from `results/metrics/coverage_table.csv` after the experiment is complete. The table structure and column definitions above are fixed.

---

### Section 4: Repository Structure

Embed the repository tree (abbreviated, top 2 levels only). Full tree is in `docs/16_repository_architecture.md`.

---

### Section 5: Quickstart / Reproducibility

```markdown
## Reproduce Results

### Requirements
Python 3.10+, all dependencies in `requirements.txt`

pip install -r requirements.txt

### Download Data
python scripts/download_data.py

### Run Full Experiment
python scripts/run_experiment.py
# Outputs saved to results/

### Generate Figures
python scripts/generate_figures.py
# Figures saved to results/figures/

### Explore in Notebooks
Open notebooks/ in order (00 through 05) for full reproducible analysis.
```

Simple. No Docker. No conda environment YAML. No cloud infrastructure. `pip install` + run = done.

---

### Section 6: Methods Summary

A concise bullet list describing each method in one line:

- **Split Conformal Prediction:** Static calibration using 2015–2017 residuals. Coverage guarantee under exchangeability.
- **CQR (Conformalized Quantile Regression):** Locally adaptive widths via Linear QR conformalization.
- **EnbPI:** Rolling ensemble-based conformal intervals tracking recent score distribution.
- **ACI (Adaptive Conformal Inference):** Online α-update that guarantees long-run average coverage even under distribution shift. *[Gibbs & Candès, NeurIPS 2021]*
- **GARCH(1,1) Gaussian:** Conditional volatility parametric baseline. Standard Basel-era method.
- **Historical Simulation VaR:** Empirical rolling-252-day quantile. Industry baseline.
- **Linear QR:** Pinball-optimal interval baseline without coverage guarantee.

---

### Section 7: Academic References

```markdown
## Key References

- Gibbs & Candès (2021). *Adaptive Conformal Inference Under Distribution Shift.* NeurIPS.
- Xu & Xie (2021). *Conformal Prediction Interval for Dynamic Time-Series.* JMLR.
- Romano, Patterson & Candès (2019). *Conformalized Quantile Regression.* NeurIPS.
- Tibshirani et al. (2019). *Conformal Prediction Under Covariate Shift.* NeurIPS.
- Vovk, Gammerman & Shafer (2005). *Algorithmic Learning in a Random World.* Springer.
- Kupiec (1995). *Techniques for Verifying the Accuracy of Risk Measurement Models.* FEDS.
- Christoffersen (1998). *Evaluating Interval Forecasts.* International Economic Review.
- Bollerslev (1986). *Generalized Autoregressive Conditional Heteroscedasticity.* Journal of Econometrics.
```

8 references. All primary sources. No textbooks. No Wikipedia. This reference list alone signals academic seriousness.

---

### Section 8: Links

```markdown
## Documentation

Full architecture: [docs/01_executive_thesis.md](docs/01_executive_thesis.md)
Experimental protocol: [docs/11_experimental_protocol.md](docs/11_experimental_protocol.md)
Finance application: [docs/14_risk_management_decision_layer.md](docs/14_risk_management_decision_layer.md)
```

---

## What Screenshots/Figures to Include in README

| Position | Figure | Purpose |
|---|---|---|
| Top of README (hero) | `fig01_rolling_coverage.png` | Instant impact — shows the core result |
| Key Results section | `fig03_regime_coverage_heatmap.png` | Quantitative confirmation |
| Finance Application section | `fig06_equity_curves.png` | Practical impact |

Three embedded figures maximum. More than three makes the README a gallery rather than a document.

---

## Tone and Narrative

- **DO:** "We show that..." / "Results demonstrate..." / "ACI maintains coverage..."
- **DON'T:** "We tried to..." / "We hope that..." / "It would be interesting if..."
- **DO:** State specific numbers in the abstract and key results
- **DON'T:** Vague claims like "improved performance" without numerical backing
- **DO:** Reference specific papers by author+year, not just methods by name
- **DON'T:** "We used a machine learning approach to predict returns"

The README is the face of the project. It must be written as if it were being submitted to a minor workshop, not an undergrad assignment.

---

## What a Reviewer Must Understand in Under 90 Seconds

1. This is a coverage and risk management study, not a prediction accuracy study.
2. The central result: ACI maintains better empirical coverage than static methods during crisis periods.
3. The finance application: uncertainty-aware position sizing.
4. The experiment is clean: one asset, strict chronological splits, no look-ahead.
5. Results are reproducible in two commands.

If any of these five points requires more than 90 seconds of reading to understand, the README needs revision.
