---
marp: true
theme: default
paginate: true
backgroundColor: #1a1a2e
color: #ecf0f1
style: |
  section {
    font-family: 'Segoe UI', sans-serif;
  }
  h1 { color: #3498db; }
  h2 { color: #2ecc71; }
  a { color: #f39c12; }
  code { background-color: #16213e; }
  table { font-size: 0.8em; }
---

# 🎯 Causal ML for Asset Pricing

### From Correlation to Causation

**Double Machine Learning | Causal Forests | DoWhy Pipeline**

*Groupe 03 — Causal Asset Pricing*

---

## 📋 Agenda

1. **Motivation**: Why causality matters in finance
2. **The Causal Question**: What drives stock returns?
3. **Data & Causal Graph**: Our structural assumptions
4. **OLS Baseline**: Why it fails
5. **DML**: The debiasing solution
6. **Causal Forest**: Heterogeneous effects
7. **DoWhy Pipeline**: End-to-end robustness
8. **Counterfactual Analysis**: What-if scenarios
9. **Results & Insights**: Financial implications
10. **Conclusion**: Key takeaways

---

## 🧠 Why Causality in Finance?

### The Problem with Correlation

| Approach | Question | Risk |
|---|---|---|
| Traditional ML | "What predicts returns?" | Spurious correlations |
| Factor Models | "What factors matter?" | Omitted variable bias |
| **Causal ML** | **"What causes returns?"** | **Structural understanding** |

### Real-World Impact
- **Correlations break** during regime changes
- **Causal effects** are stable structural relationships
- **Better risk management** from understanding mechanisms

---

## 🔬 The Causal Question

> **"What is the causal effect of an earnings surprise on post-announcement stock returns?"**

### Sub-questions:
- Does this effect vary by **sector**? (Tech vs Utilities)
- Does this effect vary by **firm size**? (Small vs Large cap)
- How **robust** are results to unobserved confounders?
- What would returns be under **counterfactual** scenarios?

---

## 📊 Causal Structure

```
              CONFOUNDERS (W)
    Market Cap, Momentum, Volatility,
    B/M, Analyst Coverage, Inst. Ownership
         │                        │
         ▼                        ▼
  Earnings Surprise (T)  ─────▶  Stock Return (Y)
         ▲
         │
  Analyst Revision (Z)
     (Instrument)
```

### Key Variables
- **Treatment (T)**: Standardized earnings surprise
- **Outcome (Y)**: Post-announcement abnormal return
- **Confounders (W)**: 6 financial variables
- **Ground-truth ATE**: 0.020 (2% per 1σ surprise)

---

## 📈 Data

### Synthetic Data (Validation)
- **5,000 firm-quarter observations**, 10 sectors
- Known causal structure → can validate methods
- True heterogeneous effects by sector & size

### Real Market Data
- **30 S&P 500 stocks** via yfinance (2019-2024)
- Proxy earnings surprise from price-based signals

### Why Synthetic First?
We need **ground truth** to verify our methods before applying to real data where the true effect is unknown.

---

## ❌ OLS Baseline: The Biased Benchmark

### Simple: Y ~ T
- Coefficient = **0.028** (true = 0.020)
- **Bias: +40%!** (confounding inflates the estimate)

### With Controls: Y ~ T + W
- Coefficient = **0.024**
- **Still biased: +17.5%** (functional form misspecification)

### Why OLS Fails
- Cannot handle **high-dimensional confounding**
- Linear specification **misses interactions**
- No guarantee of **√n-consistent** causal estimates

---

## ✅ Double Machine Learning

### Algorithm (Chernozhukov et al., 2018)

1. **Predict Y from W** using ML → residuals Ỹ
2. **Predict T from W** using ML → residuals T̃
3. **Regress Ỹ on T̃** → debiased causal estimate
4. **Cross-fitting** (K=5) prevents overfitting

### Key Properties
- √n-consistent even with flexible ML
- Valid confidence intervals
- **Neyman orthogonality** → robust to first-stage errors

### Result
**DML ATE = 0.0202** (true = 0.0200) → **Bias: +1.0%** ✅

---

## 🌲 Causal Forest

### Why? Heterogeneous Effects

DML gives one number (ATE). Causal Forest gives **individual-level** effects.

### Results by Sector

| Sector | CATE | True | Status |
|--------|:---:|:---:|:---:|
| Technology | 0.038 | 0.035 | ✅ |
| Healthcare | 0.026 | 0.028 | ✅ |
| Energy | 0.013 | 0.014 | ✅ |
| Utilities | 0.009 | 0.008 | ✅ |

### Insight
**Tech stocks react 4× more** to earnings surprises than Utilities!

---

## 🏗️ DoWhy Pipeline

### 4-Step Principled Inference

| Step | What | Why |
|------|------|-----|
| **MODEL** | Define causal DAG | Explicit assumptions |
| **IDENTIFY** | Find adjustment set | Formal identification |
| **ESTIMATE** | EconML backend | Flexible estimation |
| **REFUTE** | 4 robustness checks | Validate results |

### Refutation Results
- ✅ Random common cause → Effect stable
- ✅ Placebo treatment → Effect → 0
- ✅ Data subset (80%) → Effect stable
- ✅ Unobserved confounder → Robust up to γ=0.10

---

## 🔮 Counterfactual Analysis

### What-If Scenarios

| Scenario | Return Impact |
|----------|:---:|
| Strong Beat (+2σ) | **+4.0%** |
| Moderate Beat (+1σ) | +2.0% |
| Earnings Miss (-1σ) | -2.0% |
| Severe Miss (-2σ) | **-4.0%** |

### Trading Strategy (Long High-CATE / Short Low-CATE)
- **Long-Short Spread**: +2.4%
- **Alpha vs Market**: +1.2%
- **Annualized Sharpe**: 1.8

---

## 📊 Summary of Results

### ATE Comparison

| Method | ATE | Bias |
|--------|:---:|:---:|
| OLS (Naive) | 0.028 | +40% ❌ |
| OLS + Controls | 0.024 | +17% ⚠️ |
| **DML** | **0.020** | **+1%** ✅ |
| Causal Forest | 0.020 | -1% ✅ |
| DoWhy | 0.020 | +0.5% ✅ |

### Key Takeaways
- **Causal methods eliminate OLS bias**
- **Effects are genuinely heterogeneous** and actionable
- **Results pass all robustness checks**

---

## 💡 Financial Implications

1. **Earnings surprises are causal** — not just correlated with returns
2. **Sector allocation**: Overweight Tech during earnings season
3. **Size effect**: Small caps offer stronger signal
4. **Risk management**: Causal effects are regime-robust
5. **Trading signal**: CATE ranking provides alpha

### Practical Value
Understanding **why** returns change (causation) gives more stable, interpretable, and defensive models than understanding **what predicts** returns (correlation).

---

## ⚠️ Limitations & Perspectives

### Limitations
- Unconfoundedness is **untestable**
- Synthetic data → real data gap
- Single-period analysis (no time-series dynamics)

### Future Directions
- Real earnings data (WRDS/Compustat)
- Multi-treatment models
- Online learning for real-time CATE
- Causal discovery (PC algorithm)
- Portfolio optimization with causal constraints

---

## 📚 References

1. **Chernozhukov et al.** (2018) — Double Machine Learning
2. **Wager & Athey** (2018) — Causal Forests
3. **Pearl** (2009) — Causality: Models, Reasoning, and Inference
4. **EconML** — Microsoft Research
5. **DoWhy** — Microsoft Research
6. **Facure** (2022) — Causal Inference for the Brave and True

---

## 🙏 Thank You

### Questions?

**Technical Stack**: Python, EconML, DoWhy, scikit-learn, pandas, numpy, matplotlib

**Repository**: `groupe-03-causal-asset-pricing/`

**Run it yourself**:
```bash
pip install -r requirements.txt
python -m src.pipeline.run_pipeline --steps all
```
