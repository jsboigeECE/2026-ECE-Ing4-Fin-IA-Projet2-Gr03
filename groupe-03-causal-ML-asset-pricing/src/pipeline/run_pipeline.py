"""
End-to-End Causal Inference Pipeline
=====================================

CLI entry point that runs the full causal analysis pipeline:

    data → ols → dml → forest → dowhy → counterfactual → sensitivity → visualize

Usage:
    python -m src.pipeline.run_pipeline --steps all --output outputs/
    python -m src.pipeline.run_pipeline --steps data,ols,dml --output outputs/
    python -m src.pipeline.run_pipeline --data-source real --output outputs/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def _banner(text: str) -> None:
    """Print a formatted banner."""
    width = 70
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def step_data(args) -> pd.DataFrame:
    """Step 1: Generate or load data."""
    _banner("STEP 1: DATA GENERATION")

    if args.data_source == "synthetic":
        from src.data.generator import generate_synthetic_dataset, get_true_ate, get_variable_roles

        df = generate_synthetic_dataset(
            n_obs=args.n_obs,
            seed=args.seed,
            save_path=os.path.join(args.output, "data", "synthetic_data.csv"),
        )
        roles = get_variable_roles()
        true_ate = get_true_ate(df)

        print(f"\n  Dataset: Synthetic ({args.n_obs} observations)")
        print(f"  True ATE: {true_ate:.5f}")
        print(f"  Variable roles: {roles}")

    elif args.data_source == "real":
        from src.data.real_data import fetch_market_data

        df = fetch_market_data()
        os.makedirs(os.path.join(args.output, "data"), exist_ok=True)
        df.to_csv(os.path.join(args.output, "data", "real_market_data.csv"), index=False)
        print(f"\n  Dataset: Real market data ({len(df)} observations)")

    else:
        raise ValueError(f"Unknown data source: {args.data_source}")

    from src.data.preprocessor import compute_summary_statistics
    stats = compute_summary_statistics(df)
    print(f"\n  Summary statistics:\n{stats.to_string()}")

    return df


def step_ols(df: pd.DataFrame, args) -> dict:
    """Step 2: OLS Baseline."""
    _banner("STEP 2: OLS BASELINE REGRESSION")

    from src.models.baseline_ols import run_ols_baseline, compare_ols_specifications

    # Full OLS
    ols_result = run_ols_baseline(df)
    print(f"\n  OLS Coefficient (T→Y): {ols_result.coefficient:.5f} ± {ols_result.std_error:.5f}")
    print(f"  95% CI: [{ols_result.ci_lower:.5f}, {ols_result.ci_upper:.5f}]")
    print(f"  R²: {ols_result.r_squared:.4f}")

    if "true_cate" in df.columns:
        true_ate = df["true_cate"].mean()
        bias = ols_result.coefficient - true_ate
        print(f"\n  True ATE: {true_ate:.5f}")
        print(f"  OLS Bias: {bias:.5f} ({bias/true_ate*100:.1f}%)")

    # Specification comparison
    print("\n  --- OLS Specification Comparison ---")
    comp = compare_ols_specifications(df)
    print(comp.to_string(index=False))

    return {"ols_coefficient": ols_result.coefficient, "ols_ci": (ols_result.ci_lower, ols_result.ci_upper)}


def step_dml(df: pd.DataFrame, args) -> dict:
    """Step 3: Double Machine Learning."""
    _banner("STEP 3: DOUBLE MACHINE LEARNING (DML)")

    from src.data.preprocessor import prepare_causal_matrices
    from src.models.dml_estimator import run_linear_dml, compare_first_stages

    matrices = prepare_causal_matrices(df)

    # Linear DML with Random Forest
    dml_result = run_linear_dml(
        matrices["Y"], matrices["T"], matrices["W"], matrices["X"],
        first_stage="random_forest",
    )

    print(f"\n  DML ATE: {dml_result.ate:.5f} ± {dml_result.ate_std_error:.5f}")
    print(f"  95% CI: [{dml_result.ate_ci_lower:.5f}, {dml_result.ate_ci_upper:.5f}]")

    if "true_cate" in df.columns:
        true_ate = df["true_cate"].mean()
        print(f"  True ATE: {true_ate:.5f}")
        print(f"  DML Bias: {dml_result.ate - true_ate:.5f}")

    # Compare first stages
    print("\n  --- First-Stage Model Comparison ---")
    comp = compare_first_stages(matrices["Y"], matrices["T"], matrices["W"], matrices["X"])
    print(comp.to_string(index=False))

    return {
        "dml_ate": dml_result.ate,
        "dml_ci": (dml_result.ate_ci_lower, dml_result.ate_ci_upper),
        "matrices": matrices,
    }


def step_forest(df: pd.DataFrame, args, matrices: dict = None) -> dict:
    """Step 4: Causal Forest."""
    _banner("STEP 4: CAUSAL FOREST")

    from src.data.preprocessor import prepare_causal_matrices
    from src.models.causal_forest import (
        run_causal_forest, analyze_heterogeneity_by_group, cate_by_quantile,
    )

    if matrices is None:
        matrices = prepare_causal_matrices(df)

    cf_result = run_causal_forest(
        matrices["Y"], matrices["T"], matrices["W"], matrices["X"],
        X_names=matrices["X_names"],
        n_estimators=300,
    )

    print(f"\n  Causal Forest ATE: {cf_result.ate:.5f}")
    print(f"  95% CI: [{cf_result.ate_ci_lower:.5f}, {cf_result.ate_ci_upper:.5f}]")
    print(f"  CATE Std Dev: {cf_result.cate_std:.5f}")

    # Feature importance
    print("\n  Feature Importances:")
    for name, imp in sorted(cf_result.feature_importances.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        print(f"    {name:40s}: {imp:.4f} {bar}")

    # Heterogeneity by sector
    if "sector" in df.columns:
        print("\n  --- CATE by Sector ---")
        het_sector = analyze_heterogeneity_by_group(df, cf_result.cate_values, "sector")
        print(het_sector.to_string())

    # Heterogeneity by size
    if "market_cap_quintile" in df.columns:
        print("\n  --- CATE by Market Cap Quintile ---")
        het_size = cate_by_quantile(df, cf_result.cate_values, "log_market_cap")
        print(het_size.to_string())

    return {
        "cf_ate": cf_result.ate,
        "cf_ci": (cf_result.ate_ci_lower, cf_result.ate_ci_upper),
        "cf_result": cf_result,
        "matrices": matrices,
    }


def step_dowhy(df: pd.DataFrame, args) -> dict:
    """Step 5: DoWhy Pipeline."""
    _banner("STEP 5: DoWhy CAUSAL PIPELINE")

    from src.models.dowhy_pipeline import run_dowhy_pipeline, summarize_refutations

    results = run_dowhy_pipeline(df)

    print(f"\n  DoWhy Estimate: {results.estimate_value:.5f}")
    print(f"  95% CI: {results.estimate_ci}")
    print(f"  Estimation Method: {results.estimation_method}")

    print("\n  --- Refutation Tests ---")
    ref_summary = summarize_refutations(results)
    print(ref_summary.to_string(index=False))

    return {"dowhy_results": results}


def step_counterfactual(df: pd.DataFrame, args, cf_result=None, matrices=None) -> dict:
    """Step 6: Counterfactual Analysis."""
    _banner("STEP 6: COUNTERFACTUAL / WHAT-IF ANALYSIS")

    from src.data.preprocessor import prepare_causal_matrices
    from src.models.causal_forest import run_causal_forest
    from src.analysis.counterfactual import (
        compute_counterfactual_outcomes,
        compute_individual_treatment_effects,
        evaluate_trading_strategy,
        run_scenario_analysis,
    )

    if matrices is None:
        matrices = prepare_causal_matrices(df)

    if cf_result is None:
        cf_result = run_causal_forest(
            matrices["Y"], matrices["T"], matrices["W"], matrices["X"],
            X_names=matrices["X_names"],
        )

    estimator = cf_result.estimator

    # Counterfactual outcomes
    print("\n  --- Counterfactual Outcomes ---")
    cf_outcomes = compute_counterfactual_outcomes(estimator, matrices["X"])
    print(cf_outcomes.to_string(index=False))

    # Scenario analysis
    print("\n  --- What-If Scenarios ---")
    scenarios = run_scenario_analysis(estimator, matrices["X"], df)
    print(scenarios[["Scenario", "Treatment Shift", "Mean Return Impact", "% Stocks Positively Affected"]].to_string(index=False))

    # Trading strategy
    df_ite = compute_individual_treatment_effects(estimator, matrices["X"], df)
    strategy = evaluate_trading_strategy(df_ite)
    print("\n  --- CATE-Based Trading Strategy ---")
    for k, v in strategy.items():
        if isinstance(v, float):
            print(f"    {k:30s}: {v:+.5f}")
        else:
            print(f"    {k:30s}: {v}")

    return {"scenarios": scenarios, "strategy": strategy, "cf_outcomes": cf_outcomes}


def step_sensitivity(df: pd.DataFrame, args, matrices=None) -> dict:
    """Step 7: Sensitivity Analysis."""
    _banner("STEP 7: SENSITIVITY ANALYSIS")

    from src.data.preprocessor import prepare_causal_matrices
    from src.analysis.sensitivity import (
        sensitivity_to_unobserved_confounder,
        subsample_stability,
        random_cause_test,
    )

    if matrices is None:
        matrices = prepare_causal_matrices(df)

    # Sensitivity to unobserved confounders
    print("\n  --- Sensitivity to Unobserved Confounding ---")
    sens = sensitivity_to_unobserved_confounder(
        matrices["Y"], matrices["T"], matrices["W"], matrices["X"],
        effect_strengths=[0.0, 0.01, 0.05, 0.1, 0.2],
    )
    print(sens.to_string(index=False))

    # Subsample stability
    print("\n  --- Subsample Stability ---")
    stab = subsample_stability(
        matrices["Y"], matrices["T"], matrices["W"], matrices["X"],
        fractions=[0.6, 0.8, 1.0],
        n_reps=3,
    )
    print(stab.to_string(index=False))

    # Random cause test
    print("\n  --- Random Cause Test ---")
    rct = random_cause_test(
        matrices["Y"], matrices["T"], matrices["W"], matrices["X"],
        n_random_vars=2,
    )
    print(rct.to_string(index=False))

    return {"sensitivity": sens, "stability": stab, "random_cause": rct}


def step_visualize(df: pd.DataFrame, args, all_results: dict) -> None:
    """Step 8: Generate all visualizations."""
    _banner("STEP 8: GENERATING VISUALIZATIONS")

    fig_dir = os.path.join(args.output, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")

    from src.models.dowhy_pipeline import build_causal_dag
    from src.visualization.causal_graphs import plot_causal_dag
    from src.visualization.effects_plots import (
        plot_ate_comparison, plot_cate_distribution,
        plot_heterogeneous_effects_by_sector, plot_cate_by_size,
        plot_sensitivity, plot_refutation_summary, plot_ols_vs_dml_bias,
    )
    from src.visualization.financial_insights import (
        plot_trading_strategy_performance, plot_counterfactual_scenarios,
        plot_sector_allocation,
    )
    from src.analysis.heterogeneity import analyze_effects_by_sector, analyze_effects_by_size

    true_ate = df["true_cate"].mean() if "true_cate" in df.columns else None

    # 1. Causal DAG
    G = build_causal_dag()
    plot_causal_dag(G, save_path=os.path.join(fig_dir, "causal_dag.png"))

    # 2. ATE Comparison
    estimates = {}
    if "ols_coefficient" in all_results:
        estimates["OLS (Naive)"] = {"ate": all_results["ols_coefficient"], "ci_lower": all_results["ols_ci"][0], "ci_upper": all_results["ols_ci"][1]}
    if "dml_ate" in all_results:
        estimates["DML (Linear)"] = {"ate": all_results["dml_ate"], "ci_lower": all_results["dml_ci"][0], "ci_upper": all_results["dml_ci"][1]}
    if "cf_ate" in all_results:
        estimates["Causal Forest"] = {"ate": all_results["cf_ate"], "ci_lower": all_results["cf_ci"][0], "ci_upper": all_results["cf_ci"][1]}
    if "dowhy_results" in all_results:
        dr = all_results["dowhy_results"]
        estimates["DoWhy Pipeline"] = {"ate": dr.estimate_value, "ci_lower": dr.estimate_ci[0], "ci_upper": dr.estimate_ci[1]}

    if estimates:
        plot_ate_comparison(estimates, true_ate=true_ate,
                            save_path=os.path.join(fig_dir, "ate_comparison.png"))

    # 3. CATE distribution
    if "cf_result" in all_results:
        cf_r = all_results["cf_result"]
        true_cate_arr = df["true_cate"].values if "true_cate" in df.columns else None
        plot_cate_distribution(cf_r.cate_values, true_cate=true_cate_arr,
                               save_path=os.path.join(fig_dir, "cate_distribution.png"))

        # 4. Sector effects
        if "sector" in df.columns:
            sector_df = analyze_effects_by_sector(df, cf_r.cate_values)
            plot_heterogeneous_effects_by_sector(sector_df,
                save_path=os.path.join(fig_dir, "sector_effects.png"))

            # Sector allocation
            plot_sector_allocation(sector_df,
                save_path=os.path.join(fig_dir, "sector_allocation.png"))

        # 5. Size effects
        if "market_cap_quintile" in df.columns:
            from src.models.causal_forest import cate_by_quantile
            size_df = cate_by_quantile(df, cf_r.cate_values)
            plot_cate_by_size(size_df,
                save_path=os.path.join(fig_dir, "size_effects.png"))

    # 6. OLS vs DML bias
    if "ols_coefficient" in all_results and "dml_ate" in all_results and true_ate is not None:
        plot_ols_vs_dml_bias(
            all_results["ols_coefficient"], all_results["dml_ate"], true_ate,
            save_path=os.path.join(fig_dir, "ols_vs_dml_bias.png"),
        )

    # 7. Sensitivity plot
    if "sensitivity" in all_results:
        original_ate = all_results.get("dml_ate", all_results.get("cf_ate", 0))
        plot_sensitivity(all_results["sensitivity"], original_ate,
                         save_path=os.path.join(fig_dir, "sensitivity.png"))

    # 8. Refutation summary
    if "dowhy_results" in all_results:
        from src.models.dowhy_pipeline import summarize_refutations
        ref_df = summarize_refutations(all_results["dowhy_results"])
        plot_refutation_summary(ref_df,
                                save_path=os.path.join(fig_dir, "refutation_tests.png"))

    # 9. Scenario analysis
    if "scenarios" in all_results:
        plot_counterfactual_scenarios(all_results["scenarios"],
            save_path=os.path.join(fig_dir, "counterfactual_scenarios.png"))

    # 10. Trading strategy
    if "strategy" in all_results:
        plot_trading_strategy_performance(all_results["strategy"],
            save_path=os.path.join(fig_dir, "trading_strategy.png"))

    print(f"\n  All figures saved to {fig_dir}/")
    print(f"  Generated {len(os.listdir(fig_dir))} visualizations")


def main():
    parser = argparse.ArgumentParser(
        description="Causal ML for Asset Pricing — End-to-End Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--steps", type=str, default="all",
        help="Comma-separated steps to run: data,ols,dml,forest,dowhy,counterfactual,sensitivity,visualize\nOr 'all' to run everything.",
    )
    parser.add_argument("--data-source", type=str, default="synthetic",
                        choices=["synthetic", "real"],
                        help="Data source: 'synthetic' or 'real' (yfinance)")
    parser.add_argument("--n-obs", type=int, default=5000,
                        help="Number of observations for synthetic data")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", type=str, default="outputs",
                        help="Output directory")

    args = parser.parse_args()

    # Parse steps
    if args.steps == "all":
        steps = ["data", "ols", "dml", "forest", "dowhy", "counterfactual", "sensitivity", "visualize"]
    else:
        steps = [s.strip() for s in args.steps.split(",")]

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "data"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "figures"), exist_ok=True)

    print("\n" + "█" * 70)
    print("  CAUSAL ML FOR ASSET PRICING — PIPELINE")
    print(f"  Steps: {', '.join(steps)}")
    print(f"  Data source: {args.data_source}")
    print(f"  Output: {args.output}/")
    print("█" * 70)

    start_time = time.time()
    all_results = {}
    df = None
    matrices = None
    cf_result = None

    # Execute steps
    if "data" in steps:
        df = step_data(args)

    if df is None:
        # Load existing data
        data_path = os.path.join(args.output, "data", "synthetic_data.csv")
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            print(f"[INFO] Loaded existing data from {data_path}")
        else:
            df = step_data(args)

    if "ols" in steps:
        ols_out = step_ols(df, args)
        all_results.update(ols_out)

    if "dml" in steps:
        dml_out = step_dml(df, args)
        all_results.update(dml_out)
        matrices = dml_out.get("matrices")

    if "forest" in steps:
        forest_out = step_forest(df, args, matrices)
        all_results.update(forest_out)
        cf_result = forest_out.get("cf_result")
        if matrices is None:
            matrices = forest_out.get("matrices")

    if "dowhy" in steps:
        dowhy_out = step_dowhy(df, args)
        all_results.update(dowhy_out)

    if "counterfactual" in steps:
        cf_out = step_counterfactual(df, args, cf_result, matrices)
        all_results.update(cf_out)

    if "sensitivity" in steps:
        sens_out = step_sensitivity(df, args, matrices)
        all_results.update(sens_out)

    if "visualize" in steps:
        step_visualize(df, args, all_results)

    elapsed = time.time() - start_time
    print(f"\n{'█' * 70}")
    print(f"  PIPELINE COMPLETE — {elapsed:.1f}s")
    print(f"{'█' * 70}")

    # Save summary
    summary = {
        "data_source": args.data_source,
        "n_observations": len(df) if df is not None else 0,
        "elapsed_seconds": round(elapsed, 1),
        "steps_executed": steps,
    }
    if "true_cate" in (df.columns if df is not None else []):
        summary["true_ate"] = float(df["true_cate"].mean())
    for key in ["ols_coefficient", "dml_ate", "cf_ate"]:
        if key in all_results:
            summary[key] = float(all_results[key])
    if "dowhy_results" in all_results:
        summary["dowhy_ate"] = float(all_results["dowhy_results"].estimate_value)

    summary_path = os.path.join(args.output, "pipeline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
