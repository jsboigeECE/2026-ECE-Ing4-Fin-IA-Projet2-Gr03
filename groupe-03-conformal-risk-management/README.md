````markdown
# Adaptive Conformal Prediction for Financial Risk Management

**ECE Paris — Ing4 Finance / IA Probabiliste / Machine Learning**  
**Group 03 — Lewis Orel, Thomas Nassar**

This project studies the use of **adaptive conformal prediction** for **financial risk management** on **SPY daily returns**. The objective is not to build a high-frequency alpha model or a complex forecasting engine, but to answer a narrower and more relevant question for risk control:

> **Can adaptive conformal methods provide more reliable uncertainty estimates than static interval methods and standard parametric baselines under regime shifts and crisis periods?**

Our work focuses on the calibration, robustness, and financial usefulness of prediction intervals under non-stationary market conditions, with particular attention to major stress episodes such as the **COVID crash (2020)** and the **2022 bear market**.

---

## Abstract

Classical financial risk models often fail precisely when they are most needed: during abrupt volatility shifts, tail events, and crisis regimes. In such environments, static interval procedures and parametric models may produce overly optimistic downside risk estimates, leading to poor coverage and weak risk control.

This project investigates whether **adaptive conformal prediction** can improve the reliability of return uncertainty estimation on **SPY daily log-returns**. We compare static conformal prediction, **Adaptive Conformal Inference (ACI)**, and several baselines including **historical simulation**, **Gaussian GARCH(1,1)**, **Ridge regression**, and **quantile regression**.

The project is framed as an **applied research study in uncertainty quantification for finance**. The emphasis is not on maximizing raw predictive accuracy, but on producing intervals that remain informative, calibrated, and useful when market conditions deteriorate. We also examine whether interval-derived uncertainty can support a simple **risk-aware exposure adjustment rule**.

---

## Research Question

The central question of the project is the following:

> **Do adaptive conformal prediction methods maintain more reliable coverage than static interval methods under financial stress, and can this improvement be translated into a meaningful risk management signal?**

This question is intentionally twofold:

1. **Statistical reliability**  
   Can adaptive methods preserve or recover interval coverage more effectively than static approaches when the distribution of returns changes abruptly?

2. **Financial relevance**  
   Can interval-based uncertainty be used to improve risk control, for example through a simple exposure scaling rule?

---

## Project Scope

The project deliberately adopts a **narrow, rigorous, and reproducible setup**.

### Asset Universe
- **Primary asset:** SPY (SPDR S&P 500 ETF Trust)
- **Regime indicator:** VIX

### Prediction Target
- **1-day ahead SPY log-return**

### Frequency
- **Daily**

### Time Span
- **2004–2024**

### Core Focus
- interval estimation under non-stationarity
- coverage reliability during crisis periods
- downside risk interpretation
- simple uncertainty-aware financial decision layer

This is **not**:
- a portfolio optimization project
- a deep learning project
- a live trading system
- an option pricing project
- an intraday forecasting project

The aim is to prioritize **clarity, defensibility, and methodological quality** over unnecessary complexity.

---

## Financial Motivation

In financial markets, uncertainty is not constant. Volatility clusters, tail risks emerge suddenly, and models calibrated on calm periods often fail when regimes shift. This makes interval reliability central for risk management.

The intuition behind the project is simple:

- a **prediction interval** gives a plausible range for tomorrow’s return,
- its **lower bound** can be interpreted as a downside risk proxy,
- and its **width** can be interpreted as a measure of uncertainty.

This gives the project a direct connection to practical financial risk control:
- if intervals become wider,
- uncertainty is higher,
- and a risk manager may rationally reduce exposure.

The project therefore sits at the intersection of:
- **uncertainty quantification**
- **financial econometrics**
- **risk management**

---

## Methods

We compare several methods of increasing sophistication.

### Baseline Forecast / Risk Models
- **Constant Mean**
- **Ridge Regression**
- **Historical Simulation**
- **Gaussian GARCH(1,1)**
- **Linear Quantile Regression**

### Conformal / Adaptive Methods
- **Split Conformal Prediction**
- **Conformalized Quantile Regression (CQR)** *(if included in the final version)*
- **Adaptive Conformal Inference (ACI)**
- **EnbPI** *(optional / stretch depending on final implementation)*

The comparison is designed to reflect a meaningful progression:

- standard statistical / econometric baselines,
- static conformal methods,
- adaptive conformal methods designed for distribution shift.

---

## Data and Experimental Design

The experimental protocol is fully **chronological** and designed to avoid look-ahead bias.

### Data Sources
- **SPY daily prices** (adjusted close)
- **VIX daily close**

### Features
The feature set is intentionally compact and interpretable:
- lagged returns
- rolling realized volatility
- lagged VIX
- short-term return aggregates
- squared returns as a volatility proxy

### Chronological Splits
- **Train:** 2004–2014
- **Calibration:** 2015–2017
- **Validation:** 2018–2019
- **Test:** 2020–2024

This split is intentionally chosen to evaluate robustness on major out-of-sample stress periods, especially:
- the **COVID market shock**
- the **2022 rate-driven drawdown**

The goal is not to maximize in-sample fit, but to test whether interval methods remain credible when markets become unstable.

---

## Evaluation Philosophy

This project is not judged primarily by point prediction accuracy. In daily equity data, low predictability is normal and expected. The main object of interest is instead:

- **coverage**
- **interval behavior across regimes**
- **robustness under stress**
- **financial usability of uncertainty estimates**

The project therefore emphasizes:
- chronological evaluation
- regime-sensitive analysis
- out-of-sample robustness
- simplicity of interpretation

When relevant, the lower interval bound is interpreted as a downside risk estimate, and interval width is studied as a possible uncertainty signal.

---


The repository is organized to make the workflow as clear and reproducible as possible:

* `src/` contains the core code
* `scripts/` contains reproducible entry points
* `data/` stores raw and processed datasets
* `results/` stores computed outputs
* `slides/` contains the presentation material

---

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Reproducibility

### 1. Download raw market data

```bash
python scripts/download_data.py
```

### 2. Build processed datasets

```bash
python scripts/build_dataset.py
```

### 3. Run the project pipeline

Use the experiment and evaluation scripts provided in the repository to reproduce the baseline and conformal workflow.

If your local version contains dedicated scripts for:

* baseline models,
* conformal evaluation,
* metric generation,
* or result export,

run them after dataset construction.

---

## Methodological Safeguards

This project was built with strict safeguards to preserve validity:

* **strict chronological splitting**
* **no random shuffling**
* **lagged feature construction only**
* **explicit leakage checks**
* **cached raw data for reproducibility**
* **clear separation between training, calibration, validation, and test**

The objective is not to inflate apparent performance, but to produce a setup that remains **methodologically defensible** under scrutiny.

---

## Project Deliverables

This repository contains:

* the **full source code**
* the **data pipeline**
* the **feature engineering logic**
* the **baseline and conformal methods**
* the **evaluation framework**
* the **technical documentation**
* the **presentation slides**

---

## Presentation Material

Slides are available in the `slides/` directory.

---

## Strengths of the Project

This project is designed to be strong on several dimensions:

* it addresses a **clear and finance-relevant problem**
* it compares **static and adaptive uncertainty methods**
* it studies performance under **real market stress periods**
* it keeps the setup **interpretable and reproducible**
* it focuses on **calibration and reliability**, not only on point prediction
* it links uncertainty estimation to a concrete **risk management use case**

---

## Limitations

The project also has explicit and important limitations:

* the study is conducted on **one primary asset (SPY)**
* the horizon is limited to **1-day ahead**
* the financial application is intentionally **simple**
* adaptive methods improve robustness, but they do **not eliminate crisis risk**
* conclusions should remain specific to this setup unless validated on additional assets and frequencies

These limitations are acknowledged deliberately: the project prioritizes **depth and rigor** over breadth.

---

## Team

**Group 03**

* Lewis Orel — GitHub: `@LewisOrel`
* Thomas Nassar — GitHub: `@thomasnassar`



---

## Topic Reference

**A.5 — Conformal Prediction for Risk Management**
**2026 — ECE — Ing4 — Finance — IA Probabiliste, Théorie des Jeux et Machine Learning**

---

## Submission Note

All project files are intentionally organized under the required group subdirectory:

```text
groupe-03-conformal-risk-management/
```

This structure follows the course submission requirements:

* dedicated group subfolder
* source code
* documentation
* slides
* reproducible project organization

---

```
```
