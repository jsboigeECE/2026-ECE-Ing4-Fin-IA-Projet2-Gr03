# 16 — Repository Architecture

---

## Philosophy

The repository must communicate professionalism before a single line of code is read. A grader who opens the GitHub PR and sees a flat folder of `.ipynb` files with names like `untitled3_final_v2.ipynb` will mentally downgrade the project before evaluating the science.

**The repository must look like a quant research output.**

---

## Full Repository Tree

```
groupe_03/                             ← ⚠️ NAMING TO CONFIRM with course coordinator before PR submission
│                                         Current workspace: PROJET_2/ — rename to ECE submission convention
│                                         Confirm expected format (e.g., groupe_03/, A5_group03/, etc.)
├── README.md                          ← Primary landing page (see doc 17)
├── requirements.txt                   ← Pinned dependencies
├── .gitignore                         ← Standard Python + data ignores
│
├── data/
│   ├── raw/
│   │   ├── spy_daily.csv              ← SPY OHLCV + adjusted close (yfinance download)
│   │   ├── vix_daily.csv              ← VIX daily close
│   │   └── .gitkeep                   ← Tracks empty dirs; actual CSVs in .gitignore
│   └── processed/
│       ├── features.csv               ← Feature matrix X (all dates)
│       ├── targets.csv                ← Target series y (all dates)
│       └── splits.json                ← Split boundary dates as ISO strings
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 ← yfinance download + caching
│   ├── feature_engineering.py         ← All feature construction (shift, rolling)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_learner.py            ← Ridge wrapper with ModelWrapper interface
│   │   ├── quantile_regression.py     ← Linear QR wrapper
│   │   ├── garch_model.py             ← GARCH(1,1) wrapper (arch library)
│   │   └── historical_simulation.py   ← HistSim VaR wrapper
│   ├── conformal/
│   │   ├── __init__.py
│   │   ├── split_cp.py                ← Split Conformal + CQR
│   │   ├── enbpi.py                   ← EnbPI implementation
│   │   └── aci.py                     ← ACI implementation (core algorithm)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── coverage_metrics.py        ← Coverage, Kupiec, Christoffersen
│   │   ├── width_metrics.py           ← MIW, WER, Winkler
│   │   └── decision_layer.py          ← Position sizing + portfolio metrics
│   └── visualization.py               ← All figure generation functions
│
├── notebooks/
│   ├── 00_data_exploration.ipynb      ← EDA: returns distribution, VIX, regimes
│   ├── 01_model_training.ipynb        ← Base learner fit + residual analysis
│   ├── 02_conformal_methods.ipynb     ← CP methods implementation + sanity checks
│   ├── 03_full_evaluation.ipynb       ← Complete experimental protocol
│   ├── 04_stress_analysis.ipynb       ← Regime-conditional results
│   └── 05_decision_layer.ipynb        ← Portfolio application
│
├── scripts/
│   ├── download_data.py               ← One-command data download
│   ├── run_experiment.py              ← Full pipeline execution (command-line)
│   └── generate_figures.py            ← Reproduce all figures from saved results
│
├── results/
│   ├── intervals/
│   │   ├── split_cp_intervals.csv     ← [date, lower_90, upper_90, lower_95, upper_95, ...]
│   │   ├── aci_intervals.csv          ← Same format + aci_alpha_t column
│   │   ├── enbpi_intervals.csv
│   │   ├── cqr_intervals.csv
│   │   ├── garch_intervals.csv
│   │   └── histsim_intervals.csv
│   ├── metrics/
│   │   ├── coverage_table.csv         ← Method × alpha level × period
│   │   ├── width_table.csv
│   │   ├── kupiec_table.csv
│   │   ├── christoffersen_table.csv
│   │   └── decision_layer_metrics.csv
│   └── figures/
│       ├── fig01_rolling_coverage.png
│       ├── fig01_rolling_coverage.svg
│       ├── fig02_widths_covid_zoom.png
│       ├── fig03_regime_coverage_heatmap.png
│       ├── fig04_kupiec_pvalues.png
│       ├── fig05_aci_alpha_dynamics.png
│       ├── fig06_equity_curves.png
│       ├── fig07_coverage_width_scatter.png
│       ├── fig08_exception_calendar.png
│       └── fig09_position_sizing_covid.png
│
├── docs/
│   ├── 01_executive_thesis.md
│   ├── 02_research_question_and_hypotheses.md
│   ├── 03_project_scope_and_non_goals.md
│   ├── 04_finance_problem_formulation.md
│   ├── 05_data_strategy.md
│   ├── 06_target_definition.md
│   ├── 07_feature_strategy.md
│   ├── 08_modeling_strategy.md
│   ├── 09_conformal_methods_design.md
│   ├── 10_benchmark_matrix.md
│   ├── 11_experimental_protocol.md
│   ├── 12_metrics_and_statistical_checks.md
│   ├── 13_regime_shift_and_stress_evaluation.md
│   ├── 14_risk_management_decision_layer.md
│   ├── 15_visualization_masterplan.md
│   ├── 16_repository_architecture.md
│   ├── 17_readme_blueprint.md
│   ├── 18_presentation_storyline.md
│   ├── 19_oral_defense_strategy.md
│   ├── 20_grading_risk_register.md
│   ├── 21_execution_plan_for_roo_code.md
│   ├── 22_kill_list.md
│   └── 23_top_grade_checklist.md
│
└── slides/
    ├── presentation_draft.pdf          ← Export from slides tool
    └── assets/                         ← Figures used in slides (symlinks to results/figures)
```

---

## File Naming Conventions

| Rule | Example |
|---|---|
| Snake_case for all Python files | `feature_engineering.py` |
| Zero-padded numbering for notebooks | `00_data_exploration.ipynb`, `01_model_training.ipynb` |
| Human-readable CSV names | `coverage_table.csv`, `kupiec_table.csv` |
| Numbered figures matching masterplan | `fig01_rolling_coverage.png` |
| Zero-padded doc numbers | `01_executive_thesis.md` |

---

## What Belongs Where

### `src/` — Production-quality modular code
Pure Python functions and classes. No inline visualizations. No data loading side-effects at import time. Every function is unit-testable. This is the code that Roo Code implements.

### `notebooks/` — Sequential documented analysis
Notebooks are numbered and linear. They import from `src/` — they do NOT re-implement logic. They serve as the readable, reproducible analytical record. Each maps to one major phase of the experimental protocol.

### `scripts/` — Reproducibility entry-points
A grader who wants to reproduce results runs: `python scripts/run_experiment.py` and gets all results CSVs. Then `python scripts/generate_figures.py` for all figures. Two commands = full reproduction.

### `results/` — All computed outputs
Saved once, loaded by visualization scripts. This ensures figures are reproducible without re-running the full experiment (which may take several minutes). Git-commit results CSVs; do NOT commit large raw data files.

### `data/raw/` — Committed or reproducibly downloadable
Option A: Commit `spy_daily.csv` and `vix_daily.csv` directly (< 5MB; acceptable for research repos).  
Option B: `.gitignore` them and provide `scripts/download_data.py` for reproduction.  
**Recommendation: Option A.** Graders should never need internet access to reproduce results.

---

## What the Final PR Must Look Like

The GitHub PR for submission must include:
1. All `src/` code committed and functional
2. All `notebooks/` with executed outputs (cells run, results visible)
3. All `results/` CSVs committed (computed outputs)
4. All `results/figures/` PNG files committed
5. All `docs/` Markdown files committed
6. `requirements.txt` with all dependencies pinned
7. `README.md` complete and accurate

The PR description must include:
- One-sentence project summary
- Link to `docs/01_executive_thesis.md`
- Link to key figure (`fig01_rolling_coverage.png`)
- Instructions to reproduce: `pip install -r requirements.txt` + `python scripts/run_experiment.py`

---

## Repository Quality Signals (For the Evaluator)

A grader who opens this repo should notice within 30 seconds:
- Clean top-level structure (no file soup)
- Professional README with a key figure visible
- Numbered, sequential documentation
- Modular `src/` code (not just notebooks)
- Results already computed (no "run this notebook in order" instructions)

These signals are not cosmetic. They demonstrate software engineering maturity and research reproducibility — both of which are implicit grading criteria.
