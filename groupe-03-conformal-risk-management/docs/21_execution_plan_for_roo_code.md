# 21 — Execution Plan for Roo Code

---

## Philosophy

Roo Code executes in **Code mode**. The architect layer is complete. All design decisions are frozen. Implementation must follow this plan exactly — no scope changes, no feature additions, no "while we're at it" diversions.

**Principle:** Build in layers. Each layer must be tested before the next layer begins. Never implement method N+1 before method N is verified to work correctly.

---

## Phase 0 — Repository Scaffolding (Do First)

**Goal:** Create the full directory structure and placeholder files. Establish the import hierarchy.

**Roo Code tasks:**
1. Create all directories: `src/`, `src/models/`, `src/conformal/`, `src/evaluation/`, `notebooks/`, `scripts/`, `data/raw/`, `data/processed/`, `results/intervals/`, `results/metrics/`, `results/figures/`, `slides/`
2. Create `__init__.py` in all `src/` subdirectories
3. Create `requirements.txt` with pinned versions:
   - `numpy>=1.24`
   - `pandas>=2.0`
   - `scikit-learn>=1.3`
   - `yfinance>=0.2.28`
   - `arch>=6.2`
   - `statsmodels>=0.14`
   - `mapie>=0.8`
   - `matplotlib>=3.7`
   - `seaborn>=0.12`
   - `scipy>=1.11`
4. Create `.gitignore` (standard Python + `data/raw/*.csv` optional)
5. Create empty `README.md` placeholder

**Gate:** Directory structure visible in terminal. All `__init__.py` importable.

---

## Phase 1 — Data Pipeline (Build Second)

**Goal:** Reproducible data download → clean features → saved to CSV.

**Roo Code tasks — implement in this order:**

1. **`src/data_loader.py`**
   - `download_spy(start, end)` → downloads SPY adjusted close via yfinance, caches to `data/raw/spy_daily.csv`
   - `download_vix(start, end)` → downloads VIX, caches to `data/raw/vix_daily.csv`
   - `load_raw()` → loads from cache, returns aligned DataFrame

2. **`src/feature_engineering.py`**
   - `compute_log_returns(prices)` → `r_t = log(p_t / p_{t-1})`
   - `compute_features(returns, vix)` → 7 features with `.shift(1)` applied correctly
   - `split_data(df, splits_config)` → returns train/cal/val/test subsets
   - `save_processed(features, targets)` → saves to `data/processed/`

3. **`scripts/download_data.py`** — calls `download_spy` and `download_vix`, prints confirmation

4. **`notebooks/00_data_exploration.ipynb`** — load data, plot returns distribution, VIX time series, rolling vol, regime frequency table. Confirm no missing values, no leakage.

**Gate:** Run `python scripts/download_data.py` → no errors. Open notebook 00 and execute all cells → plots appear, no NaN in features.

---

## Phase 2 — Base Learner and Parametric Methods (Build Third)

**Goal:** Ridge + GARCH + HistSim working with ModelWrapper interface.

**Roo Code tasks:**

1. **`src/models/base_learner.py`**
   - `RidgeWrapper` class → `.fit()`, `.predict()`, `.get_residuals()`
   - Includes `RidgeCV` for λ selection within training set only

2. **`src/models/garch_model.py`**
   - `GARCHWrapper` class → `.fit(returns)`, `.predict_interval(alpha)` using `arch` library
   - Monthly re-fit logic with fallback on convergence failure

3. **`src/models/historical_simulation.py`**
   - `HistSimWrapper` class → rolling 252-day quantile, `.predict_interval(alpha)`

4. **`src/models/quantile_regression.py`**
   - `QuantileRegressionWrapper` → `statsmodels QuantReg`, fit at τ_low and τ_high

5. **`notebooks/01_model_training.ipynb`** — fit Ridge on training set, plot residuals, check Ljung-Box and ARCH-LM tests on residuals. Confirm Ridge beats constant mean (Diebold-Mariano).

**Gate:** All wrappers importable. Ridge residuals are approximately white noise on training set. GARCH fits without errors on training set. HistSim intervals look reasonable.

---

## Phase 3 — Conformal Methods (Build Fourth)

**Goal:** Split CP, CQR, and ACI working and producing valid intervals on calibration set. EnbPI is STRETCH — only implement after this phase is fully gated.

**Roo Code tasks (MANDATORY):**

1. **`src/conformal/split_cp.py`**
   - `SplitConformalPredictor` → `.calibrate(cal_residuals)`, `.predict_interval(y_hat, alpha)`
   - `CQRPredictor` → `.calibrate(cal_cqr_scores)`, `.predict_interval(q_low, q_high, alpha)`
   - Note: `alpha` here = two-sided interval level (0.10 for 90% two-sided coverage)

2. **`src/conformal/aci.py`**
   - `ACIPredictor` → `.initialize(cal_scores, alpha, gamma)`, `.step(y_hat, y_true)` → returns interval + updates `alpha_t`
   - Store `alpha_t` history as a time series

3. **`notebooks/02_conformal_methods.ipynb`** — sanity check: apply Split CP, CQR, ACI on calibration set. Verify Split CP coverage = nominal ± 0.01 by construction. Verify ACI α_t updates correctly (increases after exception, decreases after correct prediction).

**Roo Code tasks (STRETCH — only if Phases 3, 4, 5 are fully complete):**

4. **`src/conformal/enbpi.py`**
   - First attempt: use `MAPIE TimeSeriesRegressor` if available in installed MAPIE version (check `mapie.time_series_regression`)
   - Fallback: custom implementation following Xu & Xie (2021) Algo 1
   - Time-box: maximum 4 hours total implementation + debugging
   - If over budget: document as "not implemented" in results, mention in supplementary section only

**Gate (mandatory before Phase 4):** Split CP coverage on calibration set = nominal ± 0.01 (by conformal guarantee). ACI α_t series increases after each exception and decreases after each correct prediction. EnbPI not required for gate passage.

---

## Phase 4 — Full Walk-Forward Evaluation (Build Fifth)

**Goal:** Run the complete experimental protocol on the test set.

**Roo Code tasks:**

1. **`src/evaluation/coverage_metrics.py`**
   - `compute_coverage(intervals, y_true)` → empirical coverage
   - `kupiec_test(n_exceptions, n_obs, alpha)` → LR statistic + p-value
   - `christoffersen_test(coverage_indicators)` → CC test p-value
   - `rolling_coverage(coverage_indicators, window=60)` → rolling Series

2. **`src/evaluation/width_metrics.py`**
   - `mean_interval_width(intervals)`, `width_efficiency_ratio(intervals, baseline_intervals)`
   - `winkler_score(intervals, y_true, alpha)` — optional

3. **`scripts/run_experiment.py`**
   - Full walk-forward loop: for each test day t → fit models → compute intervals → record coverage, width
   - Save all interval time series to `results/intervals/*.csv`
   - Save all metrics to `results/metrics/*.csv`
   - Run at EXACTLY three alpha levels: 0.10, 0.05, 0.01

4. **`notebooks/03_full_evaluation.ipynb`** — load results CSVs, build coverage tables, Kupiec tables, width tables. Check all metrics.

**Gate:** `python scripts/run_experiment.py` completes without errors. `results/intervals/*.csv` all populated. Coverage tables look reasonable (conformal methods at or above nominal level on average).

---

## Phase 5 — Regime Analysis and Decision Layer (Build Sixth)

**Goal:** Regime-conditional results + portfolio application metrics.

**Roo Code tasks:**

1. **`src/evaluation/decision_layer.py`**
   - `position_sizing(widths, w_ref, cap=1.5, floor=0.1)` → position size series
   - `strategy_returns(spy_returns, position_sizes)` → strategy return series
   - `portfolio_metrics(returns)` → Sharpe, MaxDD, Calmar, avg position
   - Compare all 4 policies (BH, VIX threshold, Static CP, ACI)

2. **`notebooks/04_stress_analysis.ipynb`** — regime-conditional coverage heatmap, Kupiec by regime, rolling coverage plot with VIX overlay

3. **`notebooks/05_decision_layer.ipynb`** — equity curves, position sizing dynamics (COVID zoom), decision layer metrics table

**Gate:** Regime coverage heatmap populated. Decision layer metrics table shows all 4 policies. Equity curves plot renders correctly.

---

## Phase 6 — Visualization and Final Output (Build Last)

**Goal:** All 9 figures generated. README complete. Repo ready for PR.

**Roo Code tasks:**

1. **`src/visualization.py`**
   - One function per figure: `plot_rolling_coverage(...)`, `plot_widths_covid_zoom(...)`, etc.
   - All functions read from `results/` CSVs — no inline computation
   - Color palette enforced: ACI=green, Static CP=red, GARCH=orange, HistSim=grey, EnbPI=blue, CQR=purple

2. **`scripts/generate_figures.py`** — calls all visualization functions, saves to `results/figures/`

3. **`README.md`** — implement per doc 17 blueprint

4. Final PR checklist: all notebooks run clean, all figures committed, requirements pinned

**Gate:** `python scripts/generate_figures.py` → all 9 figures saved. README renders correctly on GitHub.

---

## What Can Be Parallelized

The following can be implemented simultaneously by different sessions:
- Phase 2 (parametric models) and Phase 3 (conformal methods) are independent once Phase 1 is done
- `src/evaluation/coverage_metrics.py` and `src/evaluation/width_metrics.py` can be written once the interface is agreed

## What CANNOT Be Parallelized

- Phase 1 must complete before Phase 2/3 begin
- Phase 4 requires Phase 2 and Phase 3 to be complete
- Figures require Phase 4 and Phase 5 to be complete

## Mandatory Gates Before Moving to Next Phase

| Phase | Gate Condition |
|---|---|
| After Phase 0 | All directories exist, all imports work |
| After Phase 1 | Data loads, features have zero NaN, leakage check passes |
| After Phase 2 | Ridge R² > 0.005 on validation, GARCH converges |
| After Phase 3 | Split CP calibration coverage = nominal ± 0.01 |
| After Phase 4 | All results CSVs populated, no NaN in coverage tables |
| After Phase 5 | Equity curves plot cleanly, regime heatmap readable |
| After Phase 6 | All notebooks run clean from top-to-bottom |

---

## Debug Mode Audit Checklist

When switching to Debug mode, prioritize these investigations:

1. Feature leakage audit (`.shift()` verification — highest priority)
2. ACI `alpha_t` behavior: does it decrease after exceptions and increase after correct intervals?
3. GARCH convergence rate: what % of monthly re-fits converge?
4. Walk-forward loop off-by-one errors: verify prediction uses only data strictly before the target date
5. Coverage vs nominal level on calibration set: must match by construction for Split CP
