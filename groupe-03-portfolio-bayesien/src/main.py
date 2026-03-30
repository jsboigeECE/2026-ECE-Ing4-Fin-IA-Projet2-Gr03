"""
Script principal du projet Black-Litterman.

Génère :
  - métriques de comparaison Markowitz / Black-Litterman
  - backtest out-of-sample
  - analyse de sensibilité sur les confiances
  - figures PNG et tableaux CSV/JSON dans results/
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.black_litterman import (
    black_litterman_posterior,
    market_implied_risk_aversion,
    views_contribution,
)
from src.config import (
    BACKTEST_INITIAL_WEALTH,
    COLORS,
    FIGURE_DPI,
    FIGURE_SIZE,
    FIGURES_DIR,
    RESULTS_DIR,
    SECTOR_MAX_WEIGHTS,
    SECTORS,
    SENSITIVITY_CONFIDENCE_GRID,
    SENSITIVITY_VIEW_INDEX,
    TICKERS,
    TRAIN_END,
    TEST_START,
    build_views,
)
from src.data import (
    compute_returns,
    download_prices,
    market_weights,
    prepare_all,
    split_train_test,
)
from src.markowitz import (
    efficient_frontier,
    max_sharpe,
    portfolio_stats,
    random_portfolios,
    sector_max_constraints,
)


def ensure_dirs() -> None:
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)


def format_weights(weights: np.ndarray, tickers: list[str]) -> pd.DataFrame:
    return (
        pd.DataFrame({"ticker": tickers, "weight": weights})
        .assign(sector=lambda df: df["ticker"].map(SECTORS))
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )


def performance_from_log_returns(
    log_returns: pd.Series, rf: float, label: str, start_value: float = BACKTEST_INITIAL_WEALTH
) -> tuple[dict, pd.Series]:
    wealth = start_value * np.exp(log_returns.cumsum())
    ann_return = float(np.exp(log_returns.mean() * 252) - 1)
    ann_vol = float(log_returns.std(ddof=1) * np.sqrt(252))
    sharpe = (ann_return - rf) / ann_vol if ann_vol > 0 else 0.0
    drawdown = wealth / wealth.cummax() - 1.0
    metrics = {
        "annual_return": ann_return,
        "annual_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "final_value": float(wealth.iloc[-1]),
    }
    return metrics, wealth


def plot_prior_posterior(tickers: list[str], pi: np.ndarray, mu_bl: np.ndarray) -> None:
    order = np.argsort(mu_bl)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    y = np.arange(len(tickers))
    ax.barh(y - 0.18, pi[order], height=0.35, label="Prior", color="#A7C7E7")
    ax.barh(y + 0.18, mu_bl[order], height=0.35, label="Posterior BL", color=COLORS["bl"])
    ax.set_yticks(y)
    ax.set_yticklabels(np.array(tickers)[order])
    ax.set_xlabel("Rendement annualisé attendu")
    ax.set_title("Prior d'équilibre vs posterior Black-Litterman")
    ax.legend()
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(Path(FIGURES_DIR) / "prior_vs_posterior.png", dpi=FIGURE_DPI)
    plt.close(fig)


def plot_allocations(weights_map: dict[str, np.ndarray], tickers: list[str]) -> None:
    frame = pd.DataFrame(weights_map, index=tickers)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    frame.plot(kind="bar", ax=ax, color=[COLORS["markowitz"], COLORS["bl"], COLORS["equal"], COLORS["market"]])
    ax.set_ylabel("Poids")
    ax.set_title("Comparaison des allocations")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(title="Stratégie")
    fig.tight_layout()
    fig.savefig(Path(FIGURES_DIR) / "allocations_comparison.png", dpi=FIGURE_DPI)
    plt.close(fig)


def plot_frontiers(random_ports: dict, frontier_mkz: dict, frontier_bl: dict, mkz_stats: dict, bl_stats: dict) -> None:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    scatter = ax.scatter(
        random_ports["vols"],
        random_ports["returns"],
        c=random_ports["sharpes"],
        cmap="viridis",
        s=8,
        alpha=0.35,
    )
    ax.plot(frontier_mkz["vols"], frontier_mkz["returns"], color=COLORS["markowitz"], lw=2, label="Frontière Markowitz")
    ax.plot(frontier_bl["vols"], frontier_bl["returns"], color=COLORS["bl"], lw=2, label="Frontière BL")
    ax.scatter(mkz_stats["volatility"], mkz_stats["return"], color=COLORS["markowitz"], s=120, marker="*", label="Max Sharpe MKZ")
    ax.scatter(bl_stats["volatility"], bl_stats["return"], color=COLORS["bl"], s=120, marker="*", label="Max Sharpe BL")
    ax.set_xlabel("Volatilité annualisée")
    ax.set_ylabel("Rendement annualisé")
    ax.set_title("Frontière efficiente : Markowitz vs Black-Litterman")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.colorbar(scatter, ax=ax, label="Sharpe")
    fig.tight_layout()
    fig.savefig(Path(FIGURES_DIR) / "efficient_frontier.png", dpi=FIGURE_DPI)
    plt.close(fig)


def plot_backtest(wealth_curves: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    wealth_curves.plot(ax=ax, lw=2, color=[COLORS["bl"], COLORS["markowitz"], COLORS["equal"], COLORS["market"]])
    ax.set_ylabel("Valeur du portefeuille (base 100)")
    ax.set_title("Backtesting out-of-sample")
    ax.grid(alpha=0.2)
    ax.legend(title="Stratégie")
    fig.tight_layout()
    fig.savefig(Path(FIGURES_DIR) / "backtest_equity_curves.png", dpi=FIGURE_DPI)
    plt.close(fig)


def plot_sensitivity(sensitivity: pd.DataFrame, base_confidence: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    axes[0].plot(sensitivity["confidence"], sensitivity["focus_weight"], color=COLORS["bl"], lw=2)
    axes[0].axvline(base_confidence, color="gray", ls="--", lw=1)
    axes[0].set_title("Poids de l'actif de la view absolue selon la confiance")
    axes[0].set_xlabel("Confiance")
    axes[0].set_ylabel("Poids")
    axes[0].grid(alpha=0.2)

    axes[1].plot(sensitivity["confidence"], sensitivity["out_sample_sharpe"], color=COLORS["markowitz"], lw=2)
    axes[1].axvline(base_confidence, color="gray", ls="--", lw=1)
    axes[1].set_title("Sharpe out-of-sample selon la confiance")
    axes[1].set_xlabel("Confiance")
    axes[1].set_ylabel("Sharpe")
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(Path(FIGURES_DIR) / "sensitivity_confidence.png", dpi=FIGURE_DPI)
    plt.close(fig)


def plot_view_contributions(tickers: list[str], contribs: np.ndarray, names: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(tickers))
    bottom = np.zeros(len(tickers))
    palette = ["#4C78A8", "#59A14F", "#F28E2B"]
    for i, name in enumerate(names):
        ax.bar(x, contribs[i], bottom=bottom, label=name, color=palette[i % len(palette)])
        bottom = bottom + contribs[i]
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=45, ha="right")
    ax.set_ylabel("Impact sur le rendement annualisé")
    ax.set_title("Contribution marginale des views au posterior")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(Path(FIGURES_DIR) / "view_contributions.png", dpi=FIGURE_DPI)
    plt.close(fig)


def make_summary_table(summary_map: dict[str, dict]) -> pd.DataFrame:
    df = pd.DataFrame(summary_map).T
    df.index.name = "strategy"
    return df.reset_index()


def run_project() -> dict:
    ensure_dirs()

    prices = download_prices()
    train_prices, test_prices = split_train_test(prices, train_end=TRAIN_END, test_start=TEST_START)
    data = prepare_all(train_prices)

    tickers = data["tickers"]
    rf = data["rf"]
    constraints = sector_max_constraints(tickers, SECTORS, SECTOR_MAX_WEIGHTS)
    views = build_views(tickers)
    market_log_returns = data["log_rets"] @ data["w_mkt"]
    lambda_mkt = market_implied_risk_aversion(market_log_returns.values, rf=rf)
    focus_asset_idx = tickers.index("NVDA")

    bl = black_litterman_posterior(data["cov"], data["w_mkt"], views, lambda_=lambda_mkt)
    pi = bl["pi"]
    mu_bl = bl["mu_bl"]
    cov_bl = bl["cov_bl"]

    w_markowitz = max_sharpe(data["mu"], data["cov"], rf=rf, extra_constraints=constraints)
    w_bl = max_sharpe(mu_bl, data["cov"], rf=rf, extra_constraints=constraints)
    w_equal = np.ones(len(tickers)) / len(tickers)
    w_market = market_weights(tickers)

    markowitz_stats = portfolio_stats(w_markowitz, data["mu"], data["cov"], rf=rf)
    bl_stats = portfolio_stats(w_bl, mu_bl, data["cov"], rf=rf)
    equal_stats = portfolio_stats(w_equal, data["mu"], data["cov"], rf=rf)
    market_stats = portfolio_stats(w_market, data["mu"], data["cov"], rf=rf)

    frontier_mkz = efficient_frontier(data["mu"], data["cov"], extra_constraints=constraints)
    frontier_bl = efficient_frontier(mu_bl, data["cov"], extra_constraints=constraints)
    random_ports = random_portfolios(data["mu"], data["cov"])

    test_log_returns = compute_returns(test_prices)
    backtest_metrics = {}
    wealth_curves = pd.DataFrame(index=test_log_returns.index)
    for label, weights in {
        "Black-Litterman": w_bl,
        "Markowitz": w_markowitz,
        "Equal Weight": w_equal,
        "Market Cap": w_market,
    }.items():
        port_log = test_log_returns @ weights
        metrics, wealth = performance_from_log_returns(port_log, rf=rf, label=label)
        backtest_metrics[label] = metrics
        wealth_curves[label] = wealth

    sensitivity_rows = []
    base_confidence = views[SENSITIVITY_VIEW_INDEX]["confidence"]
    for conf in SENSITIVITY_CONFIDENCE_GRID:
        scenario_views = copy.deepcopy(views)
        scenario_views[SENSITIVITY_VIEW_INDEX]["confidence"] = float(conf)
        scenario_bl = black_litterman_posterior(
            data["cov"], data["w_mkt"], scenario_views, lambda_=lambda_mkt
        )
        scenario_w = max_sharpe(
            scenario_bl["mu_bl"], data["cov"], rf=rf, extra_constraints=constraints
        )
        scenario_port_log = test_log_returns @ scenario_w
        scenario_metrics, _ = performance_from_log_returns(
            scenario_port_log, rf=rf, label=f"BL_conf_{conf:.2f}"
        )
        sensitivity_rows.append(
            {
                "confidence": float(conf),
                "focus_weight": float(scenario_w[focus_asset_idx]),
                "out_sample_sharpe": scenario_metrics["sharpe"],
                "out_sample_return": scenario_metrics["annual_return"],
            }
        )
    sensitivity_df = pd.DataFrame(sensitivity_rows)

    contribs = views_contribution(bl, data["mu"])

    summary = make_summary_table(
        {
            "Markowitz": markowitz_stats,
            "Black-Litterman": bl_stats,
            "Equal Weight": equal_stats,
            "Market Cap": market_stats,
        }
    )
    backtest_df = pd.DataFrame(backtest_metrics).T.reset_index(names="strategy")

    summary.to_csv(Path(RESULTS_DIR) / "in_sample_summary.csv", index=False)
    backtest_df.to_csv(Path(RESULTS_DIR) / "backtest_summary.csv", index=False)
    sensitivity_df.to_csv(Path(RESULTS_DIR) / "sensitivity.csv", index=False)
    format_weights(w_markowitz, tickers).to_csv(Path(RESULTS_DIR) / "weights_markowitz.csv", index=False)
    format_weights(w_bl, tickers).to_csv(Path(RESULTS_DIR) / "weights_black_litterman.csv", index=False)
    format_weights(w_market, tickers).to_csv(Path(RESULTS_DIR) / "weights_market.csv", index=False)
    pd.DataFrame({"ticker": tickers, "prior": pi, "posterior_bl": mu_bl}).to_csv(
        Path(RESULTS_DIR) / "prior_posterior_returns.csv", index=False
    )
    pd.DataFrame(contribs.T, columns=[view["name"] for view in views], index=tickers).reset_index(
        names="ticker"
    ).to_csv(Path(RESULTS_DIR) / "view_contributions.csv", index=False)
    wealth_curves.to_csv(Path(RESULTS_DIR) / "wealth_curves.csv")

    metadata = {
        "train_period": {"start": str(train_prices.index.min().date()), "end": str(train_prices.index.max().date())},
        "test_period": {"start": str(test_prices.index.min().date()), "end": str(test_prices.index.max().date())},
        "assets": tickers,
        "sector_constraints": SECTOR_MAX_WEIGHTS,
        "market_implied_risk_aversion": lambda_mkt,
        "views": [
            {"name": view["name"], "Q": view["Q"], "confidence": view["confidence"]}
            for view in views
        ],
    }
    with open(Path(RESULTS_DIR) / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    plot_prior_posterior(tickers, pi, mu_bl)
    plot_allocations(
        {
            "Markowitz": w_markowitz,
            "Black-Litterman": w_bl,
            "Equal Weight": w_equal,
            "Market Cap": w_market,
        },
        tickers,
    )
    plot_frontiers(random_ports, frontier_mkz, frontier_bl, markowitz_stats, bl_stats)
    plot_backtest(wealth_curves)
    plot_sensitivity(sensitivity_df, base_confidence=base_confidence)
    plot_view_contributions(tickers, contribs, [view["name"] for view in views])

    return {
        "summary": summary,
        "backtest": backtest_df,
        "sensitivity": sensitivity_df,
        "weights_bl": format_weights(w_bl, tickers),
        "weights_markowitz": format_weights(w_markowitz, tickers),
    }


if __name__ == "__main__":
    outputs = run_project()
    print("\n=== In-sample summary ===")
    print(outputs["summary"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== Backtest summary ===")
    print(outputs["backtest"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
