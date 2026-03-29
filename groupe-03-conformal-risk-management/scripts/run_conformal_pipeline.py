"""Full conformal risk management pipeline.

Covers all three objective levels:
  Minimum  : Split Conformal Prediction with Ridge, RF and GBR base models
  Bon      : Adaptive Conformal Inference (ACI) with dynamic coverage tracking
  Excellent: Portfolio (CPPS), Bayesian and quantile regression comparison,
             crisis period evaluation

Usage
-----
    python scripts/run_conformal_pipeline.py [--rebuild] [--save-plots]

Options
-------
--rebuild     : force rebuild of the processed dataset (re-downloads data)
--save-plots  : save figures to figures/ instead of showing them interactively
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FIGURES_DIR = PROJECT_ROOT / "figures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_splits() -> tuple:
    """Load processed features/targets and return (X, y, splits) NumPy arrays."""
    from src.data_loader import load_aligned_market_data
    from src.feature_engineering import (
        FEATURE_COLUMNS,
        TARGET_COLUMN,
        SPLITS,
        build_feature_frame,
        save_processed_dataset,
        split_by_date_range,
    )

    processed_features = PROJECT_ROOT / "data" / "processed" / "features.csv"
    processed_targets = PROJECT_ROOT / "data" / "processed" / "targets.csv"

    if processed_features.exists() and processed_targets.exists():
        features = pd.read_csv(processed_features, index_col="date", parse_dates=True)
        targets = pd.read_csv(processed_targets, index_col="date", parse_dates=True)
        frame = pd.concat([features, targets], axis=1)
    else:
        print("Processed dataset not found — building from raw data...")
        raw = load_aligned_market_data(force=False)
        frame = build_feature_frame(raw)
        save_processed_dataset(frame)
        print("Dataset built.")

    subsets = split_by_date_range(frame)
    return frame, subsets, FEATURE_COLUMNS, TARGET_COLUMN


def _to_arrays(subset: pd.DataFrame, feature_cols: list, target_col: str):
    X = subset[feature_cols].to_numpy(dtype=float)
    y = subset[target_col].to_numpy(dtype=float)
    return X, y


def _show_or_save(fig, name: str, save: bool) -> None:
    if save:
        FIGURES_DIR.mkdir(exist_ok=True)
        path = FIGURES_DIR / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {path}")
    else:
        import matplotlib.pyplot as plt
        plt.show()
    import matplotlib.pyplot as plt
    plt.close(fig)


def _print_section(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


# ---------------------------------------------------------------------------
# Objective 1 — Minimum: Split Conformal Prediction
# ---------------------------------------------------------------------------


def run_split_conformal(subsets, feature_cols, target_col, alpha, save_plots) -> dict:
    from src.conformal import SplitConformalRegressor
    from src.models import make_gbr_pipeline, make_rf_pipeline, make_ridge_pipeline
    from src.evaluation import winkler_score, mean_interval_width, marginal_coverage
    from src.visualization import plot_prediction_intervals

    _print_section("Objective 1 — Split Conformal Prediction")

    X_train, y_train = _to_arrays(subsets["train"], feature_cols, target_col)
    X_cal, y_cal = _to_arrays(subsets["calibration"], feature_cols, target_col)
    X_test, y_test = _to_arrays(subsets["test"], feature_cols, target_col)
    dates_test = subsets["test"].index

    base_models = {
        "Ridge": make_ridge_pipeline(),
        "RandomForest": make_rf_pipeline(n_estimators=200),
        "GradientBoosting": make_gbr_pipeline(),
    }

    results_summary = {}

    for name, model in base_models.items():
        cp = SplitConformalRegressor(model=model, alpha=alpha)
        cp.fit(X_train, y_train, X_cal, y_cal)

        lo, hi = cp.predict_interval(X_test)
        y_pred = cp.predict(X_test)

        cov = marginal_coverage(y_test, lo, hi)
        width = mean_interval_width(lo, hi)
        ws = winkler_score(y_test, lo, hi, alpha)

        print(f"\n  [{name}]")
        print(f"    q_hat (half-width): {cp.q_hat_:.5f}  ({cp.q_hat_ * 100:.3f}%)")
        print(f"    Coverage          : {cov:.3%}  (target {1 - alpha:.0%})")
        print(f"    Mean width        : {width * 100:.4f}%")
        print(f"    Winkler score     : {ws:.6f}")

        results_summary[name] = {
            "coverage": cov,
            "mean_width": width,
            "winkler": ws,
            "lower": lo,
            "upper": hi,
            "y_pred": y_pred,
            "model": cp,
        }

        fig = plot_prediction_intervals(
            dates_test, y_test, y_pred, lo, hi,
            title=f"Split Conformal — {name}",
            alpha=alpha,
        )
        _show_or_save(fig, f"split_conformal_{name.lower()}", save_plots)

    return results_summary


# ---------------------------------------------------------------------------
# Objective 2 — Bon: Adaptive Conformal Inference
# ---------------------------------------------------------------------------


def run_aci(subsets, feature_cols, target_col, alpha, gamma, save_plots) -> "ACIResult":
    from src.aci import AdaptiveConformalInference
    from src.models import make_gbr_pipeline
    from src.visualization import plot_prediction_intervals, plot_aci_diagnostics

    _print_section("Objective 2 — Adaptive Conformal Inference (ACI)")

    X_train, y_train = _to_arrays(subsets["train"], feature_cols, target_col)
    X_cal, y_cal = _to_arrays(subsets["calibration"], feature_cols, target_col)
    X_test, y_test = _to_arrays(subsets["test"], feature_cols, target_col)
    dates_test = subsets["test"].index

    aci = AdaptiveConformalInference(
        model=make_gbr_pipeline(),
        alpha=alpha,
        gamma=gamma,
    )
    result = aci.run(
        X_train, y_train,
        X_cal, y_cal,
        X_test, y_test,
        dates_test=dates_test,
    )

    print(f"\n  Overall coverage   : {result.empirical_coverage:.3%}  (target {1 - alpha:.0%})")
    print(f"  Mean width         : {result.mean_width * 100:.4f}%")
    print(f"  Winkler score      : {result.winkler_score:.6f}")
    print(f"  alpha_t range      : [{result.alpha_seq.min():.4f}, {result.alpha_seq.max():.4f}]")

    fig1 = plot_prediction_intervals(
        result.dates, result.y_true, result.y_pred, result.lower, result.upper,
        title="Adaptive Conformal Inference (ACI) — GBR base model",
        alpha=alpha,
    )
    _show_or_save(fig1, "aci_intervals", save_plots)

    fig2 = plot_aci_diagnostics(result)
    _show_or_save(fig2, "aci_diagnostics", save_plots)

    return result


# ---------------------------------------------------------------------------
# Objective 3 — Excellent: Portfolio + Baseline Comparison + Crisis Analysis
# ---------------------------------------------------------------------------


def run_portfolio(subsets, feature_cols, target_col, alpha, aci_result, save_plots) -> None:
    from src.conformal import SplitConformalRegressor
    from src.models import make_gbr_pipeline
    from src.portfolio import run_binary_cpps, run_vol_scaled_cpps, portfolio_period_breakdown
    from src.visualization import plot_portfolio, plot_conformal_var

    _print_section("Objective 3a — CPPS Portfolio Strategies")

    X_train, y_train = _to_arrays(subsets["train"], feature_cols, target_col)
    X_cal, y_cal = _to_arrays(subsets["calibration"], feature_cols, target_col)
    X_test, y_test = _to_arrays(subsets["test"], feature_cols, target_col)
    dates_test = subsets["test"].index

    # Fit split conformal (GBR) to get intervals
    cp = SplitConformalRegressor(model=make_gbr_pipeline(), alpha=alpha)
    cp.fit(X_train, y_train, X_cal, y_cal)
    lo, hi = cp.predict_interval(X_test)

    # --- Binary CPPS (VaR-controlled) ---
    binary = run_binary_cpps(dates_test, y_test, lo, hi, risk_budget=0.015)
    print("\n  Binary CPPS (VaR budget = 1.5%/day):")
    for k, v in binary.metrics().items():
        print(f"    {k:<35}: {v}")

    fig1 = plot_portfolio(binary)
    fig1.suptitle("Binary CPPS (invest when ConfVaR < 1.5%/day)", fontsize=11)
    _show_or_save(fig1, "portfolio_binary", save_plots)

    # --- Vol-scaled CPPS ---
    vol_scaled = run_vol_scaled_cpps(dates_test, y_test, lo, hi, target_vol=0.10)
    print("\n  Volatility-scaled CPPS (target vol = 10% p.a.):")
    for k, v in vol_scaled.metrics().items():
        print(f"    {k:<35}: {v}")

    fig2 = plot_portfolio(vol_scaled)
    fig2.suptitle("Volatility-Scaled CPPS (inverse uncertainty sizing)", fontsize=11)
    _show_or_save(fig2, "portfolio_vol_scaled", save_plots)

    # Conformal VaR plot
    fig3 = plot_conformal_var(dates_test, lo, y_test, alpha=alpha)
    _show_or_save(fig3, "conformal_var", save_plots)

    # Period breakdown
    print("\n  Period breakdown — Binary CPPS:")
    print(portfolio_period_breakdown(binary).to_string(index=False))


def run_comparison(subsets, feature_cols, target_col, alpha, aci_result, save_plots) -> None:
    from src.conformal import SplitConformalRegressor
    from src.models import (
        make_gbr_pipeline,
        BayesianRidgeInterval,
        QuantileRegressionInterval,
    )
    from src.evaluation import (
        marginal_coverage, mean_interval_width, winkler_score,
        evaluate_by_period, conditional_coverage_by_vix, CRISIS_PERIODS,
    )
    from src.visualization import (
        plot_method_comparison, plot_conditional_coverage,
    )

    _print_section("Objective 3b — Method Comparison (Conformal vs Bayesian vs Quantile)")

    X_train, y_train = _to_arrays(subsets["train"], feature_cols, target_col)
    X_cal, y_cal = _to_arrays(subsets["calibration"], feature_cols, target_col)
    X_val, y_val = _to_arrays(subsets["validation"], feature_cols, target_col)
    X_test, y_test = _to_arrays(subsets["test"], feature_cols, target_col)
    dates_test = subsets["test"].index

    # For Bayesian/QR: train on train+calibration (they don't need a held-out cal set)
    X_train_full = np.vstack([X_train, X_cal])
    y_train_full = np.concatenate([y_train, y_cal])

    methods: dict[str, tuple] = {}  # name -> (lower, upper)

    # 1. Split Conformal (GBR)
    cp = SplitConformalRegressor(make_gbr_pipeline(), alpha=alpha)
    cp.fit(X_train, y_train, X_cal, y_cal)
    lo_sc, hi_sc = cp.predict_interval(X_test)
    methods["Split Conformal\n(GBR)"] = (lo_sc, hi_sc)

    # 2. ACI (from already-computed result)
    methods["ACI\n(GBR, g=0.005)"] = (aci_result.lower, aci_result.upper)

    # 3. Bayesian Ridge
    bay = BayesianRidgeInterval(alpha=alpha)
    bay.fit(X_train_full, y_train_full)
    lo_bay, hi_bay = bay.predict_interval(X_test)
    methods["Bayesian Ridge"] = (lo_bay, hi_bay)

    # 4. Quantile Regression (GBR)
    qr = QuantileRegressionInterval(alpha=alpha)
    qr.fit(X_train_full, y_train_full)
    lo_qr, hi_qr = qr.predict_interval(X_test)
    methods["Quantile Regression\n(GBR)"] = (lo_qr, hi_qr)

    # Compute summary metrics
    names, coverages, widths, winklers = [], [], [], []
    for name, (lo, hi) in methods.items():
        cov = marginal_coverage(y_test, lo, hi)
        wid = mean_interval_width(lo, hi)
        ws = winkler_score(y_test, lo, hi, alpha)
        names.append(name)
        coverages.append(cov)
        widths.append(wid)
        winklers.append(ws)
        print(f"\n  {name.replace(chr(10), ' '):<35}  coverage={cov:.3%}  width={wid*100:.3f}%  winkler={ws:.5f}")

    fig = plot_method_comparison(names, coverages, widths, winklers, target_coverage=1 - alpha)
    _show_or_save(fig, "method_comparison", save_plots)

    # --- Crisis period evaluation (Split Conformal) ---
    _print_section("Objective 3c — Crisis Period Evaluation (Split Conformal)")
    crisis_df = evaluate_by_period(dates_test, y_test, lo_sc, hi_sc, alpha)
    print(crisis_df.to_string(index=False))

    # --- Conditional coverage by VIX ---
    _print_section("Objective 3d — Conditional Coverage by VIX Regime")

    # Need the test VIX — load it from the raw market data
    try:
        raw_path = PROJECT_ROOT / "data" / "raw" / "vix_daily.csv"
        vix_raw = pd.read_csv(raw_path, index_col="date", parse_dates=True)
        vix_col = "adj_close" if "adj_close" in vix_raw.columns else "close"
        vix_test = vix_raw.reindex(dates_test)[vix_col].to_numpy(dtype=float)

        # Fill any residual NaNs with column median
        if np.isnan(vix_test).any():
            median_val = float(np.nanmedian(vix_test))
            vix_test = np.where(np.isnan(vix_test), median_val, vix_test)

        cond_df = conditional_coverage_by_vix(dates_test, y_test, lo_sc, hi_sc, vix_test)
        print("\n  Split Conformal — conditional coverage by VIX:")
        print(cond_df.to_string(index=False))

        fig = plot_conditional_coverage(cond_df)
        _show_or_save(fig, "conditional_coverage_vix", save_plots)
    except Exception as exc:
        print(f"  (Skipping VIX conditional coverage: {exc})")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conformal Risk Management Pipeline")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild processed dataset from scratch.")
    parser.add_argument("--save-plots", action="store_true",
                        help="Save figures to figures/ directory.")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Target miscoverage rate (default: 0.05 -> 95%% coverage).")
    parser.add_argument("--gamma", type=float, default=0.005,
                        help="ACI adaptation step size (default: 0.005).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alpha = args.alpha
    gamma = args.gamma
    save_plots = args.save_plots

    if args.rebuild:
        # Remove cached processed files so they get rebuilt
        for f in ["features.csv", "targets.csv", "splits.json"]:
            p = PROJECT_ROOT / "data" / "processed" / f
            if p.exists():
                p.unlink()

    # Load data
    frame, subsets, feature_cols, target_col = _load_splits()

    print(f"\nDataset summary:")
    for split_name, subset in subsets.items():
        print(f"  {split_name:<14}: {len(subset):>5} rows  "
              f"({subset.index.min().date()} to {subset.index.max().date()})")

    # --- Objective 1: Split Conformal ---
    split_results = run_split_conformal(subsets, feature_cols, target_col, alpha, save_plots)

    # --- Objective 2: ACI ---
    aci_result = run_aci(subsets, feature_cols, target_col, alpha, gamma, save_plots)

    # --- Objective 3a: Portfolio ---
    run_portfolio(subsets, feature_cols, target_col, alpha, aci_result, save_plots)

    # --- Objective 3b-d: Comparison + Crisis + Conditional coverage ---
    run_comparison(subsets, feature_cols, target_col, alpha, aci_result, save_plots)

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    if save_plots:
        print(f"  Figures saved in: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
