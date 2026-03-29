# Causal ML for Asset Pricing

> **From Correlation to Causation**: Identifying the true causal drivers of stock returns using Double Machine Learning, Causal Forests, and principled causal inference pipelines.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![EconML](https://img.shields.io/badge/EconML-0.15+-green.svg)](https://econml.azurewebsites.net/)
[![DoWhy](https://img.shields.io/badge/DoWhy-0.11+-orange.svg)](https://py-why.github.io/dowhy/)

---

## 📋 Table of Contents

- [Context & Motivation](#-context--motivation)
- [The Causal Question](#-the-causal-question)
- [Data](#-data)
- [Methodology](#-methodology)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Limitations](#-limitations)
- [Perspectives](#-perspectives)
- [References](#-references)

---

## 🎯 Context & Motivation

Traditional asset pricing models rely on **correlational analysis** — regressing stock returns on factors to identify predictors. However, correlation ≠ causation:

- **Confounding**: Firm size affects both earnings quality and returns
- **Reverse causality**: High-returning stocks attract analyst attention
- **Spurious correlation**: Data mining across hundreds of factors

**This project applies Causal Machine Learning** to move from "what predicts returns?" to **"what causes returns?"**, specifically studying the causal effect of earnings surprises on stock returns.

### Why Causal ML in Finance?

| Traditional ML | Causal ML |
|---|---|
| "Factor X predicts returns" | "Factor X *causes* returns to change" |
| Breaks under regime change | Robust to distribution shift |
| Correlational associations | Structural relationships |
| Black-box predictions | Interpretable mechanisms |

---

## 🔬 The Causal Question

> **"What is the causal effect of an earnings surprise on post-announcement stock returns, and how does this effect vary across sectors and firm sizes?"**

### Causal Structure

```
                    ┌─────────────────────────────────────────────────┐
                    │              CONFOUNDERS (W)                    │
                    │  Market Cap, Momentum, Volatility,              │
                    │  B/M Ratio, Analyst Coverage, Inst. Ownership   │
                    └────────┬─────────────────────┬─────────────────┘
                             │                     │
                             ▼                     ▼
  Analyst Revision (Z) → Earnings Surprise (T) → Stock Return (Y)
         (Instrument)        (Treatment)           (Outcome)
```

- **Treatment (T)**: Standardized Unexpected Earnings (earnings surprise)
- **Outcome (Y)**: Post-announcement abnormal stock return
- **Confounders (W)**: Market cap, book-to-market, momentum, volatility, analyst coverage, institutional ownership
- **Instrument (Z)**: Pre-announcement analyst revision

---

## 📊 Data

### Synthetic Dataset (Validation)

We generate a **synthetic dataset with known causal structure** (ground truth) to validate that our methods correctly recover true causal effects:

- **5,000 firm-quarter observations**
- **10 sectors** (GICS-inspired classification)
- **True ATE ≈ 0.020** (2% return per 1σ earnings surprise)
- Heterogeneous effects: Technology responds ~4× more than Utilities

### Real Market Dataset

We also construct a real-world dataset using **yfinance**:
- **30 representative S&P 500 stocks** across 10 sectors
- **2019–2024** historical data
- Quarterly windows with proxy earnings-surprise measures

---

## 🧠 Methodology

### 1. OLS Baseline (Biased)

Standard regression `Y ~ T + W` provides a **biased** estimate because:
- OLS cannot fully adjust for high-dimensional confounding
- Functional form misspecification biases the coefficient
- This baseline demonstrates **why we need causal methods**

### 2. Double Machine Learning (DML)

Based on **Chernozhukov et al. (2018)**:

1. **First stage**: Use ML (Random Forest, GBM) to predict both Y and T from confounders W
2. **Residualize**: Compute residuals `Ỹ = Y - E[Y|W]` and `T̃ = T - E[T|W]`
3. **Second stage**: Regress `Ỹ` on `T̃` → debiased causal estimate
4. **Cross-fitting**: K-fold to prevent overfitting bias

**Key property**: √n-consistent and asymptotically normal, even with flexible ML first stages.

### 3. Causal Forest

Based on **Wager & Athey (2018)**:

- Extends DML with a **forest-based final stage**
- Each tree splits to **maximize treatment effect heterogeneity**
- Provides individual-level CATE estimates with **confidence intervals**
- Reveals which sectors and firm sizes respond most to earnings surprises

### 4. DoWhy Pipeline (End-to-End)

Full principled causal inference following the PyWhy framework:

1. **Model**: Define causal DAG from financial theory
2. **Identify**: Automatically find the backdoor adjustment set
3. **Estimate**: Use EconML estimators as computation backend
4. **Refute**: Run 4 robustness checks:
   - Random common cause
   - Placebo treatment
   - Data subset refuter
   - Unobserved common cause sensitivity

### 5. Counterfactual Analysis

- **What-if scenarios**: "What if earnings surprise = +2σ vs −1σ?"
- **Individual Treatment Effects**: Rank stocks by predicted causal responsiveness
- **Trading strategy**: Long high-CATE / Short low-CATE portfolio

---

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/ErwanSi/2026-ECE-Ing4-Fin-IA-Projet2-Gr03.git
cd groupe-03-causal-ML-asset-pricing

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.9+
- EconML ≥ 0.15.0
- DoWhy ≥ 0.11.0
- scikit-learn ≥ 1.3.0
- pandas, numpy, matplotlib, seaborn, networkx, plotly

---

## 🚀 Usage

### Run the full pipeline

```bash
# Full pipeline with synthetic data (recommended for validation)
python -m src.pipeline.run_pipeline --steps all --data-source synthetic --output outputs/

# Full pipeline with real market data
python -m src.pipeline.run_pipeline --steps all --data-source real --output outputs/

# Run specific steps
python -m src.pipeline.run_pipeline --steps data,ols,dml --output outputs/

# Custom number of observations
python -m src.pipeline.run_pipeline --steps all --n-obs 10000 --output outputs/
```

### Available steps

| Step | Description |
|------|-------------|
| `data` | Generate/download dataset |
| `ols` | OLS baseline regression |
| `dml` | Double Machine Learning |
| `forest` | Causal Forest |
| `dowhy` | DoWhy 4-step pipeline |
| `counterfactual` | What-if analysis |
| `sensitivity` | Robustness checks |
| `visualize` | Generate all plots |

### Run individual modules

```bash
# Generate synthetic data
python -m src.data.generator

# Run OLS baseline
python -m src.models.baseline_ols

# Run DML estimation
python -m src.models.dml_estimator

# Run Causal Forest
python -m src.models.causal_forest

# Run DoWhy pipeline
python -m src.models.dowhy_pipeline
```

---

## 📈 Results

### Key Findings (Synthetic Data)

| Method | ATE Estimate | 95% CI | Bias vs Truth |
|--------|:---:|:---:|:---:|
| **True ATE** | 0.0200 | — | — |
| OLS (Naive) | 0.0235 | [0.021, 0.026] | +17.5% |
| **DML (Linear)** | **0.0202** | [0.018, 0.022] | **+1.0%** |
| Causal Forest | 0.0198 | [0.017, 0.023] | -1.0% |
| DoWhy Pipeline | 0.0201 | [0.018, 0.022] | +0.5% |

**Key insight**: DML and Causal Forest successfully debias the OLS estimate, recovering the true causal effect within confidence intervals.

### Heterogeneous Effects

| Sector | Estimated CATE | True CATE |
|--------|:---:|:---:|
| Technology | 0.038 | 0.035 |
| Healthcare | 0.026 | 0.028 |
| Consumer Discretionary | 0.024 | 0.025 |
| Energy | 0.013 | 0.014 |
| Utilities | 0.009 | 0.008 |

Small-cap stocks show ~2× stronger response to earnings surprises compared to large-caps.

### Robustness

All DoWhy refutation tests pass:
- ✅ Random common cause: No significant change
- ✅ Placebo treatment: Effect drops to ~0
- ✅ Data subset: Stable across 80% subsamples
- ✅ Unobserved confounder: Robust up to γ = 0.1

---

## 📁 Project Structure

```
groupe-03-causal-asset-pricing/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── src/
│   ├── data/
│   │   ├── generator.py               # Synthetic data with known DGP
│   │   ├── real_data.py               # Real market data (yfinance)
│   │   └── preprocessor.py            # Feature engineering, encoding
│   ├── models/
│   │   ├── baseline_ols.py            # OLS regression (biased baseline)
│   │   ├── dml_estimator.py           # Double Machine Learning
│   │   ├── causal_forest.py           # Causal Forest (heterogeneous effects)
│   │   └── dowhy_pipeline.py          # DoWhy 4-step pipeline
│   ├── analysis/
│   │   ├── counterfactual.py          # What-if / intervention analysis
│   │   ├── sensitivity.py             # Robustness to unobserved confounding
│   │   └── heterogeneity.py           # Effects by sector, size, interactions
│   ├── visualization/
│   │   ├── causal_graphs.py           # DAG visualization
│   │   ├── effects_plots.py          # ATE, CATE, comparison plots
│   │   └── financial_insights.py      # Trading strategy visualization
│   └── pipeline/
│       └── run_pipeline.py            # CLI end-to-end pipeline
├── docs/
│   ├── methodology.md                 # DML, Causal Forest theory
│   ├── causal_hypotheses.md           # DAG justification
│   └── results_analysis.md            # Detailed results interpretation
├── slides/
│   └── presentation.md                # Slide deck (Marp-compatible)
├── data/                              # Generated datasets
├── outputs/
│   └── figures/                       # All generated visualizations
└── notebooks/                         # Optional interactive analysis
```

---

## ⚠️ Limitations

1. **Synthetic data limitations**: While the DGP is calibrated to realistic financial parameters, real markets exhibit non-stationarity, fat tails, and regime changes not captured here.

2. **Unconfoundedness assumption**: All causal methods assume no unobserved confounders. Although sensitivity analysis suggests robustness, this cannot be formally tested.

3. **Earnings surprise proxy**: For real data, the earnings surprise is approximated from price-based signals rather than actual analyst forecast errors (which require premium data access).

4. **Linear treatment assumption**: Some estimators assume linearity in the treatment effect. While Causal Forests relax this, the treatment variable itself is continuous and unbounded.

5. **Single-period analysis**: The current framework treats each observation independently, ignoring potential time-series dependencies and persistence effects.

---

## 🔮 Perspectives

1. **True earnings data**: Integrate WRDS/Compustat for actual analyst forecasts and earnings surprises
2. **Panel causal methods**: Extend to difference-in-differences or synthetic control for time-series data
3. **Multi-treatment**: Jointly estimate effects of multiple factors (revenue surprise, guidance, etc.)
4. **Online learning**: Adaptive CATE estimation for real-time trading signal generation
5. **Causal discovery**: Apply PC/FCI algorithms to learn the DAG structure from data rather than imposing it
6. **Risk management**: Use counterfactual analysis for portfolio stress testing

---

## 📚 References

1. **Chernozhukov, V., et al.** (2018). "Double/debiased machine learning for treatment and structural parameters." *The Econometrics Journal*, 21(1), C1-C68.

2. **Wager, S., & Athey, S.** (2018). "Estimation and inference of heterogeneous treatment effects using Random Forests." *Journal of the American Statistical Association*, 113(523), 1228-1242.

3. **EconML**: Microsoft Research. https://econml.azurewebsites.net/

4. **DoWhy**: Microsoft Research. https://py-why.github.io/dowhy/

5. **Pearl, J.** (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.

6. **Imbens, G. W., & Rubin, D. B.** (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press.

7. **Facure, M.** (2022). *Causal Inference for the Brave and True*. https://matheusfacure.github.io/python-causality-handbook/

---

## 👥 Team Organization

| Member | Responsibilities |
|--------|-----------------|
| Erwan Simon | Architecture, pipeline, DML, Causal Forest, sensitivity |
| Hamza Ouadoudi | Data generation, preprocessing, Plots, slides, documentation |

### Git Workflow

- `main`: Stable, reviewed code
- `develop`: Integration branch
- `feature/*`: Individual feature branches
- All changes via Pull Requests with code review

---


